# Profiling Workshop

Two-hour workshop materials for profiling a PyTorch training pipeline with NVIDIA Nsight Systems, plus an optional Nsight Compute kernel-analysis section.

The workshop now has two notebooks:

- `notebooks/01_nsys_trace_guided_optimization.ipynb` introduces the synthetic training pipeline, profiles paired problem/reference implementations, and guides attendees through one focused fix at a time.
- `notebooks/02_ncu_kernel_analysis.ipynb` uses Nsight Compute to inspect selected classifier-head kernels after the Nsight Systems workflow has narrowed the search.

## Repository Layout

```text
notebooks/
  01_nsys_trace_guided_optimization.ipynb
  02_ncu_kernel_analysis.ipynb
profiling_workshop/
  common.py
  data.py
  pipeline/
    shared.py
    problem/
      batching.py
      handoffs.py
      short_kernels.py
      synchronization.py
      orchestrator.py
    solution/
      batching.py
      handoffs.py
      short_kernels.py
      synchronization.py
      orchestrator.py
scripts/
  train_baseline.py
  train_optimized.py
  run_problem_pipeline.py
  run_solution_pipeline.py
  profile_classifier_head.py
traces/
  generated Nsight Systems and Nsight Compute reports go here
```

## Setup

Create an environment with PyTorch and Jupyter:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the profiling notebook, run on a machine with an NVIDIA GPU and Nsight Systems available on `PATH`:

```bash
nsys --version
```

For the optional Nsight Compute notebook, also make sure `ncu` is available:

```bash
ncu --version
```

The notebooks still run on CPU for code exploration, but the Nsight Systems and Nsight Compute exercises are intended for CUDA.

## Run the Notebooks

```bash
jupyter lab
```

Open the notebooks in order:

1. `notebooks/01_nsys_trace_guided_optimization.ipynb`
2. `notebooks/02_ncu_kernel_analysis.ipynb` if you want the optional kernel-level section.

## Run the Scripts Directly

Editable problem workload:

```bash
python3 scripts/run_problem_pipeline.py --device cuda --samples 8192 --batch-size 128 --micro-batches 16 --features 2048 --hidden 4096 --depth 4 --head broadcast-distance
```

Reference solution workload:

```bash
python3 scripts/run_solution_pipeline.py --device cuda --samples 8192 --batch-size 1024 --micro-batches 1 --features 2048 --hidden 4096 --depth 4 --head matmul-distance --num-workers 4 --prefetch-batches 2
```

The older `scripts/train_baseline.py` and `scripts/train_optimized.py` entry points are kept as aliases for the problem and solution pipelines.

Narrow classifier-head target for Nsight Compute:

```bash
python3 scripts/profile_classifier_head.py --device cuda --head broadcast-distance --batch-size 1024 --hidden 4096 --classes 64 --iterations 20
python3 scripts/profile_classifier_head.py --device cuda --head matmul-distance --batch-size 1024 --hidden 4096 --classes 64 --iterations 20
```

Capture a reference Nsight Systems trace:

```bash
nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --sample=process-tree \
  --cpuctxsw=process-tree \
  --backtrace=fp \
  --cuda-memory-usage=true \
  --stats=false \
  -o traces/baseline \
  python3 scripts/run_problem_pipeline.py --device cuda --samples 8192 --batch-size 128 --micro-batches 16 --features 2048 --hidden 4096 --depth 4 --head broadcast-distance
```

If your GPU supports Nsight Systems HW metrics, add:

```bash
--gpu-metrics-device=<gpu-id> --gpu-metrics-frequency=10000
```

Use `nsys profile --gpu-metrics-device=help --duration=1 --trace=none true` to check whether your GPU exposes those counters. The notebook auto-skips HW metrics when they are unavailable. Add `python-gil` to `--trace` if you want Python GIL events. Open the generated `traces/baseline.nsys-rep` in the Nsight Systems UI.

## Workshop Arc

Notebook 01 starts by introducing the workflow and profiling hooks:

- CPU-side data generation and feature engineering.
- Host-to-device transfer.
- A dense MLP and classifier head large enough for L40S/A100-class GPUs.
- Micro-batched training that creates short-lived kernels.
- Metric logging that creates synchronization and device-to-host traffic.

Then it uses the paired modules to inspect and fix the trace one pattern at a time:

- `synchronization.py`: remove unnecessary synchronizes and per-step D2H metric reads.
- `short_kernels.py`: replace tiny micro-batch optimizer steps with fuller-batch work.
- `handoffs.py`: move CPU preprocessing into the input side so it can overlap GPU work.
- `batching.py`: use larger batches, pinned memory, non-blocking copies, and CUDA prefetching.

Each script prints a `RESULT` line for overall throughput and `REGION` lines for the four issue areas. The notebook compares those quick measurements, then captures Nsight Systems traces with NVTX ranges so attendees can verify the timeline pattern directly.

## Optional Nsight Compute Section

Notebook 02 uses Nsight Compute as a kernel-level microscope after Nsight Systems has identified a GPU range worth inspecting. It intentionally does not ask attendees to rewrite PyTorch or cuBLAS kernels. Instead, it compares two equivalent classifier-head formulations:

- `broadcast-distance`: plausible tensor code that creates more elementwise/reduction work and a larger intermediate.
- `matmul-distance`: an algebraic rewrite that lets PyTorch route the dominant work into matrix-multiply kernels.

The intended teaching point is that Nsight Compute often tells PyTorch users which high-level formulation maps to better kernels. If the remaining hot kernel is already a strong library GEMM, the right next step may be to stop kernel tuning and return to model shape, batch size, dtype, or pipeline decisions.

Example NCU capture:

```bash
ncu \
  --force-overwrite \
  --target-processes all \
  --set full \
  --launch-count 10 \
  --nvtx \
  --nvtx-include "ncu_classifier_head_broadcast_distance/" \
  -o traces/ncu_broadcast_head \
  python3 scripts/profile_classifier_head.py --device cuda --head broadcast-distance --batch-size 1024 --hidden 4096 --classes 64 --iterations 20
```
