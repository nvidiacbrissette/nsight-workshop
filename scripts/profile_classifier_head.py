#!/usr/bin/env python
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys
import time
from typing import ContextManager

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from profiling_workshop.common import (
    HEAD_CHOICES,
    BroadcastDistanceHead,
    MatmulDistanceHead,
    maybe_sync,
    nvtx_range,
    resolve_device,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Narrow Nsight Compute target for classifier-head kernel analysis."
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--head", default="broadcast-distance", choices=HEAD_CHOICES)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--classes", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-nvtx", action="store_true")
    return parser.parse_args()


def make_head(head: str, hidden: int, classes: int) -> nn.Module:
    if head == "broadcast-distance":
        return BroadcastDistanceHead(hidden, classes)
    if head == "matmul-distance":
        return MatmulDistanceHead(hidden, classes)
    if head == "linear":
        return nn.Linear(hidden, classes)
    choices = ", ".join(HEAD_CHOICES)
    raise ValueError(f"Unknown head {head!r}. Choose one of: {choices}.")


def autocast_context(device: torch.device, amp: bool) -> ContextManager[None]:
    if amp and device.type == "cuda" and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=True)
    if amp and device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


def run_head_once(
    head: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    amp: bool,
) -> torch.Tensor:
    with autocast_context(device, amp):
        logits = head(x)
    return logits


def main() -> None:
    args = parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")

    device = resolve_device(args.device)
    seed_everything(args.seed)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    x = torch.randn(args.batch_size, args.hidden, generator=generator).to(device)
    head = make_head(args.head, args.hidden, args.classes).to(device).eval()

    with torch.inference_mode():
        for _ in range(args.warmup):
            run_head_once(head, x, device, args.amp)
        maybe_sync(device)

        range_name = f"ncu_classifier_head_{args.head.replace('-', '_')}"
        start = time.perf_counter()
        with nvtx_range(range_name, enabled=not args.no_nvtx):
            for _ in range(args.iterations):
                logits = run_head_once(head, x, device, args.amp)
        maybe_sync(device)
        seconds = time.perf_counter() - start

    checksum = float(logits.float().mean().detach().cpu().item())
    samples = args.batch_size * args.iterations
    dtype_mode = "amp" if args.amp and device.type == "cuda" else "fp32"
    print(
        f"RESULT head={args.head} device={device} dtype={dtype_mode} iterations={args.iterations} "
        f"samples={samples} seconds={seconds:.6f} samples_per_second={samples / max(seconds, 1e-12):.1f} "
        f"checksum={checksum:.6f}"
    )


if __name__ == "__main__":
    main()
