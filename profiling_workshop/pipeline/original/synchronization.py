from __future__ import annotations

import torch

from profiling_workshop.common import nvtx_range
from profiling_workshop.pipeline.shared import PipelineConfig


class MetricTracker:
    def __init__(self, device: torch.device, config: PipelineConfig) -> None:
        self.device = device
        self.config = config
        self.loss_total = 0.0
        self.correct_total = 0
        self.seen = 0

    def update(self, loss: torch.Tensor, logits: torch.Tensor, y: torch.Tensor) -> None:
        with nvtx_range("issue_1_synchronizing_metrics_and_d2h", self.config.nvtx_enabled):
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

            batch_items = y.numel()
            self.loss_total += float(loss.detach().item()) * batch_items
            pred_cpu = logits.detach().argmax(dim=1).cpu()
            y_cpu = y.detach().cpu()
            self.correct_total += int((pred_cpu == y_cpu).sum().item())
            self.seen += batch_items

    def finalize(self) -> tuple[int, float, float]:
        loss = self.loss_total / max(self.seen, 1)
        accuracy = self.correct_total / max(self.seen, 1)
        return self.seen, loss, accuracy


def synchronize_phase(device: torch.device, config: PipelineConfig, label: str) -> None:
    if device.type != "cuda":
        return
    with nvtx_range(f"issue_1_extra_cuda_synchronize_{label}", config.nvtx_enabled):
        torch.cuda.synchronize(device)
