from __future__ import annotations

import argparse
import contextlib
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import torch
from torch import nn
from torch.utils.data import Dataset

from profiling_workshop.common import HEAD_CHOICES, maybe_sync, nvtx_range


STAGE_SYNC = "issue_1_synchronization"
STAGE_KERNELS = "issue_2_short_lived_kernels"
STAGE_HANDOFF = "issue_3_cpu_gpu_handoff"
STAGE_IO = "issue_4_batch_io"


@dataclass(frozen=True)
class PipelineDefaults:
    variant: str
    description: str
    batches: int
    batch_size: int
    micro_batches: int
    num_workers: int
    prefetch_batches: int
    head: str
    pin_memory: bool


@dataclass(frozen=True)
class PipelineConfig:
    variant: str
    device: str
    seed: int
    epochs: int
    samples: int
    batch_size: int
    features: int
    classes: int
    hidden: int
    depth: int
    head: str
    micro_batches: int
    lr: float
    num_workers: int
    prefetch_batches: int
    cpu_work: int
    log_every: int
    amp: bool
    no_nvtx: bool
    pin_memory: bool

    @property
    def nvtx_enabled(self) -> bool:
        return not self.no_nvtx


@dataclass(frozen=True)
class PipelineStats:
    variant: str
    device: torch.device
    seconds: float
    samples: int
    loss: float
    accuracy: float
    stage_seconds: dict[str, float]

    @property
    def samples_per_second(self) -> float:
        return self.samples / max(self.seconds, 1e-12)


class StageTimer:
    def __init__(self) -> None:
        self._totals: defaultdict[str, float] = defaultdict(float)

    @contextlib.contextmanager
    def measure(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self._totals[name] += time.perf_counter() - start

    def totals(self) -> dict[str, float]:
        return dict(self._totals)


class RawSyntheticDataset(Dataset[torch.Tensor]):
    """Returns deterministic raw samples; feature engineering happens elsewhere."""

    def __init__(self, samples: int, features: int, seed: int) -> None:
        self.samples = samples
        self.features = features
        self.seed = seed

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + index)
        return torch.randn(self.features, generator=generator, dtype=torch.float32)


class AugmentedSyntheticDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Returns fully prepared samples so DataLoader workers can overlap CPU work."""

    def __init__(self, samples: int, features: int, classes: int, seed: int, cpu_work: int) -> None:
        self.samples = samples
        self.features = features
        self.classes = classes
        self.seed = seed
        self.cpu_work = cpu_work

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + index)
        x = torch.randn(self.features, generator=generator, dtype=torch.float32)
        x = augment_features(x, self.cpu_work)
        y = labels_from_features(x, self.classes)
        return x, y


def augment_features(x: torch.Tensor, cpu_work: int) -> torch.Tensor:
    """CPU-side feature engineering that is deliberately plausible and nontrivial."""

    out = x.float()
    repeats = max(1, cpu_work)
    for step in range(repeats):
        scale_a = 1.4 + 0.08 * step
        scale_b = 0.7 + 0.05 * step
        out = torch.sin(out * scale_a) + torch.cos(out * scale_b)
        out = out + 0.05 * torch.roll(out, shifts=step + 1, dims=-1)
    mean = out.mean(dim=-1, keepdim=True)
    std = out.std(dim=-1, keepdim=True).clamp_min(1e-5)
    return ((out - mean) / std).contiguous()


def labels_from_features(x: torch.Tensor, classes: int) -> torch.Tensor:
    return torch.argmax(x[..., :classes], dim=-1).long()


def add_pipeline_args(parser: argparse.ArgumentParser, defaults: PipelineDefaults) -> None:
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--batches", type=int, default=defaults.batches)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--features", type=int, default=2048)
    parser.add_argument("--classes", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--head", default=defaults.head, choices=HEAD_CHOICES)
    parser.add_argument("--micro-batches", type=int, default=defaults.micro_batches)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--prefetch-batches", type=int, default=defaults.prefetch_batches)
    parser.add_argument("--cpu-work", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-nvtx", action="store_true")


def config_from_args(args: argparse.Namespace, defaults: PipelineDefaults) -> PipelineConfig:
    samples = args.samples if args.samples is not None else args.batches * args.batch_size
    return PipelineConfig(
        variant=defaults.variant,
        device=args.device,
        seed=args.seed,
        epochs=args.epochs,
        samples=samples,
        batch_size=args.batch_size,
        features=args.features,
        classes=args.classes,
        hidden=args.hidden,
        depth=args.depth,
        head=args.head,
        micro_batches=args.micro_batches,
        lr=args.lr,
        num_workers=args.num_workers,
        prefetch_batches=args.prefetch_batches,
        cpu_work=args.cpu_work,
        log_every=args.log_every,
        amp=args.amp,
        no_nvtx=args.no_nvtx,
        pin_memory=defaults.pin_memory,
    )


class NoopGradScaler:
    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None


def make_grad_scaler(device: torch.device, amp: bool) -> Any:
    enabled = amp and device.type == "cuda"
    if not enabled:
        return NoopGradScaler()
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


@contextlib.contextmanager
def autocast_for(device: torch.device, amp: bool) -> Iterator[None]:
    enabled = amp and device.type == "cuda"
    if enabled and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        with torch.amp.autocast("cuda", enabled=True):
            yield
    elif enabled:
        with torch.cuda.amp.autocast(enabled=True):
            yield
    else:
        yield


def make_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr)


def loader_batch_count(config: PipelineConfig) -> int:
    return max(1, config.samples // config.batch_size)


def print_result(stats: PipelineStats) -> None:
    print(
        f"RESULT pipeline={stats.variant} device={stats.device} samples={stats.samples} "
        f"seconds={stats.seconds:.3f} samples_per_second={stats.samples_per_second:.1f} "
        f"loss={stats.loss:.4f} accuracy={stats.accuracy:.3f}"
    )
    for name, seconds in sorted(stats.stage_seconds.items()):
        percent = 100.0 * seconds / max(stats.seconds, 1e-12)
        print(
            f"REGION pipeline={stats.variant} name={name} seconds={seconds:.6f} "
            f"percent_of_wall={percent:.1f}"
        )


def maybe_log_progress(config: PipelineConfig, epoch: int, batch_idx: int, total_batches: int) -> None:
    if config.log_every and (batch_idx + 1) % config.log_every == 0:
        print(f"pipeline={config.variant} epoch={epoch} batch={batch_idx + 1}/{total_batches}")


def final_synchronize(device: torch.device) -> None:
    with nvtx_range("pipeline_final_synchronize", enabled=True):
        maybe_sync(device)


PrepareBatchFn = Callable[[Any, PipelineConfig], tuple[torch.Tensor, torch.Tensor]]
