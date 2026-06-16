from __future__ import annotations

import torch
from torch.utils.data import Dataset, TensorDataset


class SlowSyntheticDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """A deliberately expensive CPU-side dataset for profiling exercises."""

    def __init__(self, samples: int, features: int, classes: int, seed: int) -> None:
        self.samples = samples
        self.features = features
        self.classes = classes
        self.seed = seed

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + index)

        x = torch.randn(self.features, generator=generator)

        # Intentionally CPU-heavy feature engineering that runs once per sample.
        x = torch.sin(x * 1.7) + torch.cos(x * 0.6)
        x = (x - x.mean()) / (x.std() + 1e-5)

        teacher_slice = x[: self.classes]
        y = int(torch.argmax(teacher_slice).item())
        return x.float(), torch.tensor(y, dtype=torch.long)


def make_fast_dataset(
    samples: int,
    features: int,
    classes: int,
    seed: int,
) -> TensorDataset:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    x = torch.randn(samples, features, generator=generator)
    x = torch.sin(x * 1.7) + torch.cos(x * 0.6)
    x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-5)

    teacher = torch.randn(features, classes, generator=generator)
    y = torch.argmax(x @ teacher, dim=1).long()
    return TensorDataset(x.float(), y)

