from __future__ import annotations

import torch
from torch import nn

from profiling_workshop.common import nvtx_range
from profiling_workshop.pipeline.shared import (
    PipelineConfig,
    STAGE_KERNELS,
    STAGE_SYNC,
    StageTimer,
    autocast_for,
)


def train_batch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: object,
    x: torch.Tensor,
    y: torch.Tensor,
    config: PipelineConfig,
    device: torch.device,
    metric_tracker: object,
    timer: StageTimer,
) -> None:
    with timer.measure(STAGE_KERNELS):
        with nvtx_range("issue_2_full_batch_train_step", config.nvtx_enabled):
            # One optimizer step over the full batch gives each launch more work
            # to do. The timeline should show fewer launch clusters and a lower
            # ratio of launch overhead to kernel execution.
            optimizer.zero_grad(set_to_none=True)
            with autocast_for(device, config.amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    with timer.measure(STAGE_SYNC):
        metric_tracker.update(loss, logits, y)
