from __future__ import annotations

import torch
from torch.utils.data import Dataset

from profiling_workshop.common import nvtx_range
from profiling_workshop.pipeline.shared import AugmentedSyntheticDataset, PipelineConfig


def make_dataset(config: PipelineConfig) -> Dataset[tuple[torch.Tensor, torch.Tensor]]:
    # The problem pipeline performs feature engineering in the main training
    # loop. Moving it into the Dataset lets DataLoader workers prepare future
    # batches while the GPU trains on the current batch.
    return AugmentedSyntheticDataset(
        samples=config.samples,
        features=config.features,
        classes=config.classes,
        seed=config.seed,
        cpu_work=config.cpu_work,
    )


def prepare_batch(raw_batch: tuple[torch.Tensor, torch.Tensor], config: PipelineConfig) -> tuple[torch.Tensor, torch.Tensor]:
    with nvtx_range("issue_4_cpu_gpu_handoff_prefetched_cpu_batch", config.nvtx_enabled):
        x_cpu, y_cpu = raw_batch
        # Keep this function cheap: the expensive CPU work already happened in
        # the loader side of the pipeline, so the training loop avoids a CPU-only
        # gap before every GPU batch.
        return x_cpu.contiguous(), y_cpu.long()
