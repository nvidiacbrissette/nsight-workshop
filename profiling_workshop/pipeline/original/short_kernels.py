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
    micro_batches = max(1, min(config.micro_batches, x.shape[0]))
    for mb_x, mb_y in zip(torch.chunk(x, micro_batches), torch.chunk(y, micro_batches)):
        with timer.measure(STAGE_KERNELS):
            with nvtx_range("issue_3_short_lived_microbatch_train_step", config.nvtx_enabled):
                optimizer.zero_grad(set_to_none=False)
                with autocast_for(device, config.amp):
                    logits = model(mb_x)
                    loss = criterion(logits, mb_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        with timer.measure(STAGE_SYNC):
            metric_tracker.update(loss, logits, mb_y)
