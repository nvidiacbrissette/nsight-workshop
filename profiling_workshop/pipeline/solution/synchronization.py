from __future__ import annotations

import torch

from profiling_workshop.common import nvtx_range
from profiling_workshop.pipeline.shared import PipelineConfig


class MetricTracker:
    def __init__(self, device: torch.device, config: PipelineConfig) -> None:
        self.device = device
        self.config = config
        # Keep running metrics on the GPU. Pulling loss.item() or predictions to
        # the CPU every step forces synchronization and shows up as D2H traffic.
        self.loss_total = torch.zeros((), device=device)
        self.correct_total = torch.zeros((), device=device)
        self.seen = 0

    def update(self, loss: torch.Tensor, logits: torch.Tensor, y: torch.Tensor) -> None:
        with nvtx_range("issue_1_device_side_metric_accumulation", self.config.nvtx_enabled):
            batch_items = y.numel()
            # These tensor reductions can stay queued with the rest of the CUDA
            # work instead of blocking the host thread for a scalar result.
            self.loss_total += loss.detach() * batch_items
            self.correct_total += (logits.detach().argmax(dim=1) == y).sum()
            self.seen += batch_items

    def finalize(self) -> tuple[int, float, float]:
        # The final report needs CPU scalars, so synchronize once at the end
        # rather than once per micro-batch.
        loss = float(self.loss_total.detach().cpu().item()) / max(self.seen, 1)
        accuracy = float(self.correct_total.detach().cpu().item()) / max(self.seen, 1)
        return self.seen, loss, accuracy


def synchronize_phase(device: torch.device, config: PipelineConfig, label: str) -> None:
    # The matching problem function calls torch.cuda.synchronize() after H2D.
    # Leaving this as a no-op lets compute, copies, and metric work remain queued
    # unless a real data dependency requires ordering.
    return None
