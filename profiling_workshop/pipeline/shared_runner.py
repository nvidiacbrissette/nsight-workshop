from __future__ import annotations

import argparse
import time
from types import ModuleType

from torch import nn

from profiling_workshop.common import make_model, resolve_device, seed_everything
from profiling_workshop.pipeline.shared import (
    PipelineDefaults,
    PipelineStats,
    STAGE_SYNC,
    StageTimer,
    config_from_args,
    final_synchronize,
    make_grad_scaler,
    make_optimizer,
    maybe_log_progress,
    print_result,
)


def train_pipeline(
    args: argparse.Namespace,
    defaults: PipelineDefaults,
    *,
    synchronization: ModuleType,
    short_kernels: ModuleType,
    handoffs: ModuleType,
    batching: ModuleType,
) -> PipelineStats:
    config = config_from_args(args, defaults)
    device = resolve_device(config.device)
    seed_everything(config.seed)

    dataset = handoffs.make_dataset(config)
    loader = batching.make_loader(dataset, config, device)
    model = make_model(
        config.features,
        config.hidden,
        config.classes,
        config.depth,
        config.head,
        nvtx_enabled=config.nvtx_enabled,
    ).to(device)
    optimizer = make_optimizer(model, config.lr)
    criterion = nn.CrossEntropyLoss()
    scaler = make_grad_scaler(device, config.amp)
    metrics = synchronization.MetricTracker(device, config)
    timer = StageTimer()

    final_synchronize(device)
    wall_start = time.perf_counter()

    total_batches = len(loader)
    for epoch in range(config.epochs):
        device_batches = batching.DeviceBatchIterator(
            loader,
            handoffs.prepare_batch,
            config,
            device,
            timer,
        )
        for batch_idx, (x, y) in enumerate(device_batches):
            with timer.measure(STAGE_SYNC):
                synchronization.synchronize_phase(device, config, "after_h2d")
            short_kernels.train_batch(
                model,
                optimizer,
                criterion,
                scaler,
                x,
                y,
                config,
                device,
                metrics,
                timer,
            )
            maybe_log_progress(config, epoch, batch_idx, total_batches)

    final_synchronize(device)
    wall_seconds = time.perf_counter() - wall_start
    seen, loss, accuracy = metrics.finalize()

    stats = PipelineStats(
        variant=config.variant,
        device=device,
        seconds=wall_seconds,
        samples=seen,
        loss=loss,
        accuracy=accuracy,
        stage_seconds=timer.totals(),
    )
    print_result(stats)
    return stats
