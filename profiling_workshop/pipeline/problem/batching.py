from __future__ import annotations

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
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
        drop_last=True,
    )


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
        for raw_batch in self.loader:
            with self.timer.measure(STAGE_HANDOFF):
                x_cpu, y_cpu = self.prepare_batch(raw_batch, self.config)

            with self.timer.measure(STAGE_IO):
                with nvtx_range("issue_4_batch_io_blocking_h2d", self.config.nvtx_enabled):
                    x = x_cpu.to(self.device)
                    y = y_cpu.to(self.device)

            yield x, y

    def __len__(self) -> int:
        return loader_batch_count(self.config)

