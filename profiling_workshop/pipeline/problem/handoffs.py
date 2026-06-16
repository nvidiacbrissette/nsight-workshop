from __future__ import annotations

import torch
from torch.utils.data import Dataset

from profiling_workshop.common import nvtx_range
from profiling_workshop.pipeline.shared import (
    RawSyntheticDataset,
    PipelineConfig,
    augment_features,
    labels_from_features,
)


def make_dataset(config: PipelineConfig) -> Dataset[torch.Tensor]:
    return RawSyntheticDataset(
        samples=config.samples,
        features=config.features,
        seed=config.seed,
    )


def prepare_batch(raw_batch: torch.Tensor, config: PipelineConfig) -> tuple[torch.Tensor, torch.Tensor]:
    with nvtx_range("issue_3_cpu_gpu_handoff_main_thread_preprocessing", config.nvtx_enabled):
        x_cpu = augment_features(raw_batch, config.cpu_work)
        y_cpu = labels_from_features(x_cpu, config.classes)
    return x_cpu, y_cpu

