from __future__ import annotations

from collections import deque
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset

from profiling_workshop.common import nvtx_range
from profiling_workshop.pipeline.shared import (
    PrepareBatchFn,
    PipelineConfig,
    STAGE_HANDOFF,
    STAGE_IO,
    StageTimer,
    loader_batch_count,
)


def make_loader(dataset: Dataset[object], config: PipelineConfig, device: torch.device) -> DataLoader[object]:
    # Larger batches reduce per-sample launch and copy overhead. Pinned host memory
    # lets CUDA DMA engines pull batches without first staging them through a
    # temporary pinned buffer, which is what makes non_blocking=True meaningful.
    kwargs: dict[str, object] = {
        "batch_size": config.batch_size,
        "shuffle": False,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": True,
    }
    if config.num_workers > 0:
        # Persistent workers and DataLoader prefetch keep CPU-side batch
        # preparation ahead of the GPU instead of restarting workers each epoch.
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


class DeviceBatchIterator:
    def __init__(
        self,
        loader: DataLoader[object],
        prepare_batch: PrepareBatchFn,
        config: PipelineConfig,
        device: torch.device,
        timer: StageTimer,
    ) -> None:
        self.loader = loader
        self.prepare_batch = prepare_batch
        self.config = config
        self.device = device
        self.timer = timer

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        # CPU and non-prefetch runs still use the same interface, just without
        # CUDA stream overlap. This keeps the lab code structurally identical.
        if self.device.type != "cuda" or self.config.prefetch_batches <= 0:
            yield from self._iter_blocking()
            return
        yield from self._iter_prefetched_cuda()

    def __len__(self) -> int:
        return loader_batch_count(self.config)

    def _iter_blocking(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for raw_batch in self.loader:
            with self.timer.measure(STAGE_HANDOFF):
                x_cpu, y_cpu = self.prepare_batch(raw_batch, self.config)
            with self.timer.measure(STAGE_IO):
                with nvtx_range("issue_4_batch_io_non_blocking_h2d", self.config.nvtx_enabled):
                    # non_blocking=True can avoid synchronizing the training
                    # stream when the source tensors are pinned by the loader.
                    x = x_cpu.to(self.device, non_blocking=True)
                    y = y_cpu.to(self.device, non_blocking=True)
            yield x, y

    def _iter_prefetched_cuda(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        # H2D copies are issued on a side stream so the next batch can transfer
        # while kernels for the current batch are already running.
        stream = torch.cuda.Stream(device=self.device)
        source = iter(self.loader)
        queued: deque[tuple[torch.Tensor, torch.Tensor, torch.cuda.Event]] = deque()

        def enqueue_next() -> bool:
            try:
                raw_batch = next(source)
            except StopIteration:
                return False

            with self.timer.measure(STAGE_HANDOFF):
                x_cpu, y_cpu = self.prepare_batch(raw_batch, self.config)

            event = torch.cuda.Event()
            with self.timer.measure(STAGE_IO):
                with torch.cuda.stream(stream):
                    with nvtx_range("issue_4_batch_io_prefetched_h2d", self.config.nvtx_enabled):
                        x = x_cpu.to(self.device, non_blocking=True)
                        y = y_cpu.to(self.device, non_blocking=True)
                    # The event marks when this batch is safe for the training
                    # stream to consume. Without it, compute could race the copy.
                    event.record(stream)
            queued.append((x, y, event))
            return True

        # Fill a small queue before yielding the first batch so there is already
        # copy work in flight when the training loop starts computing.
        for _ in range(max(1, self.config.prefetch_batches)):
            if not enqueue_next():
                break

        current_stream = torch.cuda.current_stream(self.device)
        while queued:
            x, y, event = queued.popleft()
            current_stream.wait_event(event)
            # record_stream extends the tensors' lifetime on the training stream;
            # otherwise PyTorch may recycle their storage while kernels still use it.
            x.record_stream(current_stream)
            y.record_stream(current_stream)
            enqueue_next()
            yield x, y
