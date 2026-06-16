from __future__ import annotations

import contextlib
import random
import warnings
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn


warnings.filterwarnings("ignore", message="Can't initialize NVML")

HEAD_CHOICES = ("broadcast-distance", "matmul-distance", "linear")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if cuda_is_available_quietly() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not cuda_is_available_quietly():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def cuda_is_available_quietly() -> bool:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Can't initialize NVML")
        return torch.cuda.is_available()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda_is_available_quietly():
        torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def nvtx_range(name: str, enabled: bool = True) -> Iterator[None]:
    if enabled and cuda_is_available_quietly():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class RunStats:
    seconds: float
    samples: int
    loss: float
    accuracy: float

    @property
    def samples_per_second(self) -> float:
        return self.samples / max(self.seconds, 1e-12)


class FeatureMLP(nn.Module):
    def __init__(self, features: int, hidden: int, depth: int) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")

        layers: list[nn.Module] = []
        in_features = features
        for _ in range(depth):
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.GELU())
            in_features = hidden
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BroadcastDistanceHead(nn.Module):
    """Classify by computing all sample/prototype distances with broadcasting."""

    def __init__(self, embedding_features: int, classes: int) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(classes, embedding_features) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff = x[:, None, :] - self.prototypes[None, :, :]
        distances = diff.square().sum(dim=-1)
        return -distances


class MatmulDistanceHead(nn.Module):
    """Equivalent squared-distance classifier written to use a matrix multiply."""

    def __init__(self, embedding_features: int, classes: int) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(classes, embedding_features) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = x.square().sum(dim=1, keepdim=True)
        prototype_norm = self.prototypes.square().sum(dim=1).unsqueeze(0)
        distances = x_norm - 2.0 * (x @ self.prototypes.t()) + prototype_norm
        return -distances


class ProfiledClassifier(nn.Module):
    def __init__(
        self,
        features: int,
        hidden: int,
        classes: int,
        depth: int,
        head: str,
        nvtx_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.features = FeatureMLP(features, hidden, depth)
        self.head_name = head
        self.nvtx_enabled = nvtx_enabled

        if head == "broadcast-distance":
            self.head = BroadcastDistanceHead(hidden, classes)
        elif head == "matmul-distance":
            self.head = MatmulDistanceHead(hidden, classes)
        elif head == "linear":
            self.head = nn.Linear(hidden, classes)
        else:
            choices = ", ".join(HEAD_CHOICES)
            raise ValueError(f"Unknown head {head!r}. Choose one of: {choices}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with nvtx_range("feature_mlp", self.nvtx_enabled):
            x = self.features(x)
        range_name = self.head_name.replace("-", "_")
        with nvtx_range(f"classifier_head_{range_name}", self.nvtx_enabled):
            return self.head(x)


def make_model(
    features: int,
    hidden: int,
    classes: int,
    depth: int,
    head: str,
    nvtx_enabled: bool = True,
) -> ProfiledClassifier:
    return ProfiledClassifier(features, hidden, classes, depth, head, nvtx_enabled)
