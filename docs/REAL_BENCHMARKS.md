# Real RTX 2050 Benchmark Protocol

This benchmark is the reproducible path for actual hardware results. It does
not use the synthetic benchmark table in `run_all_phases.py`.

## What It Measures

- Actual CUDA peak allocated and reserved memory.
- Actual tokens per second and milliseconds per step.
- Process RSS memory on the host.
- OOM status for each method.
- NeuroCache activation offload statistics when enabled.

## Methods

- `baseline`: ordinary CUDA training.
- `checkpoint`: PyTorch gradient checkpointing over Transformer blocks.
- `neurocache_cpu`: saved autograd tensors are offloaded to CPU.
- `neurocache_sync_cpu`: saved autograd tensors are offloaded to CPU with
  synchronous pinned-buffer copies.
- `neurocache_cpu_fp16`: saved autograd tensors are offloaded to CPU as FP16.
- `neurocache_cpu_int8`: saved autograd tensors are offloaded to CPU as INT8.
- `neurocache_checkpoint_cpu`: combines block checkpointing with CPU
  saved-tensor offload.
- `neurocache_sync_checkpoint_cpu`: same policy using synchronous pinned-buffer
  copies.
- `neurocache_checkpoint_fp16`: combines block checkpointing with FP16
  saved-tensor offload.
- `neurocache_sync_checkpoint_fp16`: same policy using synchronous pinned-buffer
  copies for comparison against stream-based transfers.
- `neurocache_heuristic_checkpoint_fp16`: keeps frequently reused tensors on GPU.
- `neurocache_predictor_checkpoint_fp16`: uses the balanced LSTM predictor to
  choose keep/offload actions.
- `checkpoint_cpuadam`: checkpointing with AdamW optimizer state on CPU.
- `checkpoint_hybridadam`: checkpointing with Adam first moment on GPU and
  second moment on CPU.
- `checkpoint_bf16adam`: checkpointing with Adam moments stored as BF16 CUDA
  tensors.
- `neurocache_sync_checkpoint_cpu_hybridadam`: synchronous NeuroCache activation
  offload plus HybridAdam optimizer-state placement.
- `neurocache_budget5_checkpoint_cpu_bf16adam`: synchronous NeuroCache
  activation offload capped at five tensors per step plus BF16 CUDA optimizer
  moments. This is the current best RTX 2050 target-envelope variant.

The NeuroCache variants use `torch.autograd.graph.saved_tensors_hooks`, so they
move real training activations instead of simulating tensor placement.

## Run

```powershell
.venv\Scripts\python.exe experiments\rtx2050_real_benchmark.py
```

Outputs:

- `results/metrics/rtx2050_real_benchmark.json`
- `results/metrics/rtx2050_real_benchmark.csv`

The result files include both baseline-relative and checkpoint-relative fields
when those variants are present:

- `vram_reduction_vs_baseline_pct`
- `throughput_vs_baseline_pct`
- `vram_reduction_vs_checkpoint_pct`
- `throughput_vs_checkpoint_pct`

Train the balanced scheduling predictor:

```powershell
.venv\Scripts\python.exe experiments\train_balanced_predictor.py --output-dir results\metrics\balanced_predictor_real
```

Run the publication validation suite:

```powershell
.venv\Scripts\python.exe experiments\validate_neurocache_rtx2050.py --steps 10 --output-dir results\validation\rtx2050_publication
```

Regenerate tables and plots from existing raw validation data:

```powershell
.venv\Scripts\python.exe experiments\validate_neurocache_rtx2050.py --steps 10 --output-dir results\validation\rtx2050_publication --postprocess-only
```

Validation outputs:

- `raw_runs.csv` and `raw_runs.json`
- `main_result_table.csv`
- `multi_seed_stats.csv`
- `budget_sweep_table.csv`
- `plots/memory_vs_budget.png` and `.svg`
- `plots/throughput_vs_budget.png` and `.svg`
- `plots/memory_throughput_tradeoff.png` and `.svg`
- `plots/checkpoint_vs_neurocache_bar.png` and `.svg`
- `PAPER_RESULTS_DRAFT.md`

## Notes

The model uses deterministic random token batches. That is intentional: it
isolates systems behavior from dataset download speed, tokenization, and model
hub availability. For paper-grade claims, run at least three seeds and report
mean plus standard deviation.

Use `--warmup-steps 1` or higher for optimizer-state comparisons. With zero
warmup steps, optimizer state allocation can make peak-memory comparisons less
representative.
