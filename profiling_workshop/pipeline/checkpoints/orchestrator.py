from __future__ import annotations

import argparse
from dataclasses import dataclass
from types import ModuleType

from profiling_workshop.pipeline.original import (
    batching as original_batching,
    handoffs as original_handoffs,
    short_kernels as original_short_kernels,
)
from profiling_workshop.pipeline.solution import (
    batching as solution_batching,
    handoffs as solution_handoffs,
    short_kernels as solution_short_kernels,
    synchronization as solution_synchronization,
)
from profiling_workshop.pipeline.shared import PipelineDefaults
from profiling_workshop.pipeline.shared_runner import train_pipeline


@dataclass(frozen=True)
class Checkpoint:
    name: str
    description: str
    synchronization: ModuleType
    short_kernels: ModuleType
    handoffs: ModuleType
    batching: ModuleType
    defaults: PipelineDefaults


def checkpoint_defaults(name: str, description: str, *, pin_memory: bool = False) -> PipelineDefaults:
    return PipelineDefaults(
        variant=f"example_{name}",
        description=description,
        batches=128,
        batch_size=128,
        micro_batches=16,
        num_workers=0,
        prefetch_batches=0,
        head="broadcast-distance",
        pin_memory=pin_memory,
    )


CHECKPOINTS: dict[str, Checkpoint] = {
    "sync": Checkpoint(
        name="sync",
        description="Reference checkpoint with only the synchronization fix applied.",
        synchronization=solution_synchronization,
        short_kernels=original_short_kernels,
        handoffs=original_handoffs,
        batching=original_batching,
        defaults=checkpoint_defaults(
            "sync",
            "Reference checkpoint with only the synchronization fix applied.",
        ),
    ),
    "io": Checkpoint(
        name="io",
        description="Reference checkpoint with synchronization and host/device IO fixes applied.",
        synchronization=solution_synchronization,
        short_kernels=original_short_kernels,
        handoffs=original_handoffs,
        batching=solution_batching,
        defaults=checkpoint_defaults(
            "io",
            "Reference checkpoint with synchronization and host/device IO fixes applied.",
            pin_memory=True,
        ),
    ),
    "kernels": Checkpoint(
        name="kernels",
        description="Reference checkpoint with synchronization, IO, and kernel-launch fixes applied.",
        synchronization=solution_synchronization,
        short_kernels=solution_short_kernels,
        handoffs=original_handoffs,
        batching=solution_batching,
        defaults=checkpoint_defaults(
            "kernels",
            "Reference checkpoint with synchronization, IO, and kernel-launch fixes applied.",
            pin_memory=True,
        ),
    ),
    "handoff": Checkpoint(
        name="handoff",
        description="Reference checkpoint with all four Nsight Systems lab fixes applied.",
        synchronization=solution_synchronization,
        short_kernels=solution_short_kernels,
        handoffs=solution_handoffs,
        batching=solution_batching,
        defaults=checkpoint_defaults(
            "handoff",
            "Reference checkpoint with all four Nsight Systems lab fixes applied.",
            pin_memory=True,
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    checkpoint_parser = argparse.ArgumentParser(add_help=False)
    checkpoint_parser.add_argument("--checkpoint", choices=CHECKPOINTS, default="sync")
    checkpoint_args, _ = checkpoint_parser.parse_known_args()
    checkpoint = CHECKPOINTS[checkpoint_args.checkpoint]

    parser = argparse.ArgumentParser(description=checkpoint.description)
    parser.add_argument("--checkpoint", choices=CHECKPOINTS, default=checkpoint.name)

    from profiling_workshop.pipeline.shared import add_pipeline_args

    add_pipeline_args(parser, checkpoint.defaults)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = CHECKPOINTS[args.checkpoint]
    train_pipeline(
        args,
        checkpoint.defaults,
        synchronization=checkpoint.synchronization,
        short_kernels=checkpoint.short_kernels,
        handoffs=checkpoint.handoffs,
        batching=checkpoint.batching,
    )


if __name__ == "__main__":
    main()
