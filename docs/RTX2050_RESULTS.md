# RTX 2050 Real Benchmark Results

Last updated: 2026-05-05

Hardware/runtime:

- GPU: NVIDIA GeForce RTX 2050, 4096 MiB reported by `nvidia-smi`
- Driver: 577.05
- Python: 3.11.14
- PyTorch: 2.11.0+cu128
- CUDA runtime in PyTorch: 12.8
- System RAM: 12.6 GB

Benchmark model:

- GPT-like decoder trained on deterministic random token batches
- Parameters: 97.93M
- Layers: 12
- Hidden size: 768
- Heads: 12
- Sequence length: 768
- Batch size: 7 for the original activation-offload result; batch size 8 for
  the current optimizer-plus-activation result
- Warmup steps: 1
- Measured steps: 3

## Current Best Measured Configuration

The strongest result so far is synchronous, budgeted NeuroCache activation
offload combined with BF16 optimizer-state placement:

- Transformer blocks use PyTorch checkpointing.
- Five large saved autograd tensors per step are offloaded to pinned CPU memory
  with real `saved_tensors_hooks`.
- Adam moments stay on CUDA but are stored in BF16.
- No synthetic placement scores or fabricated timing are used.

Command:

```powershell
.venv\Scripts\python.exe experiments\rtx2050_real_benchmark.py --warmup-steps 1 --measure-steps 3 --d-model 768 --n-layers 12 --n-heads 12 --seq-len 768 --batch-size 8 --min-tensor-kb 4096 --variants checkpoint,checkpoint_bf16adam,neurocache_budget4_checkpoint_cpu_bf16adam,neurocache_budget5_checkpoint_cpu_bf16adam --output-dir results\metrics\rtx2050_bf16adam_budget5_b8_multistep
```

| Variant | Peak CUDA allocated MB | Tokens/sec | ms/step | VRAM reduction vs checkpoint | Throughput vs checkpoint |
|---|---:|---:|---:|---:|---:|
| checkpoint | 2894.5 | 1904.3 | 3226.3 | 0.0% | 0.0% |
| checkpoint_bf16adam | 2546.4 | 1910.8 | 3215.3 | 12.0% | +0.3% |
| neurocache_budget4_checkpoint_cpu_bf16adam | 2474.4 | 1903.0 | 3228.6 | 14.5% | -0.1% |
| neurocache_budget5_checkpoint_cpu_bf16adam | 2456.4 | 1905.7 | 3224.1 | 15.1% | +0.1% |

Three-seed check for `neurocache_budget5_checkpoint_cpu_bf16adam`
(`seed=42,7,123`):

| Metric | Mean | Std |
|---|---:|---:|
| VRAM reduction vs checkpoint | 15.14% | 0.00 |
| Throughput vs checkpoint | +0.48% | 0.35 |
| NeuroCache tokens/sec | 1906.1 | 3.9 |
| Checkpoint tokens/sec | 1897.0 | 7.5 |

Interpretation:

- This configuration reaches the target envelope in real CUDA measurements:
  peak CUDA allocation is 15.1% lower than checkpointing and throughput is
  slightly higher in the three-seed average.
- The BF16 optimizer-state path is not bit-identical to FP32 AdamW. Losses were
  finite and close across the short benchmark, but longer convergence tests are
  still required before making training-quality claims.
- The budgeted activation policy matters: four offloads per step almost reaches
  the memory target, while five crosses it with little throughput cost.
- DeepSpeed is not yet included in these measured results on this Windows RTX
  2050 environment.

Output artifacts:

- `results/metrics/rtx2050_bf16adam_budget5_b8_multistep/rtx2050_real_benchmark.json`
- `results/metrics/rtx2050_bf16adam_budget5_b8_seed7/rtx2050_real_benchmark.json`
- `results/metrics/rtx2050_bf16adam_budget5_b8_seed123/rtx2050_real_benchmark.json`

## Publication Validation Run

The publication validation suite was run with five seeds, 10 measured steps per
run, and the locked budget-5 NeuroCache configuration.

Command:

```powershell
.venv\Scripts\python.exe experiments\validate_neurocache_rtx2050.py --steps 10 --output-dir results\validation\rtx2050_publication
```

Main five-seed result:

| Variant | Peak CUDA MB mean | Tokens/sec mean | VRAM vs checkpoint | Throughput vs checkpoint | Final loss mean |
|---|---:|---:|---:|---:|---:|
| checkpoint | 2895.8 | 1890.0 | 0.0% | 0.0% | 9.7987 |
| checkpoint_bf16adam | 2545.6 | 1891.8 | 12.1% | +0.1% | 9.7659 |
| neurocache_budget5_checkpoint_cpu_bf16adam | 2455.6 | 1885.0 | 15.2% | -0.3% | 9.7659 |

The run produced 30 real CUDA measurements with zero non-finite losses. The
locked NeuroCache configuration satisfies the memory target but does not strictly
prove a mean throughput win in this 10-step validation; it is 0.26% slower on
average and within normal run-to-run noise.

To reduce timing noise, a 20-step locked-configuration repeat compared only
`checkpoint` and `neurocache_budget5_checkpoint_cpu_bf16adam` across the same
five seeds:

| Variant | Peak CUDA MB mean | Tokens/sec mean | VRAM vs checkpoint | Throughput vs checkpoint | Final loss mean |
|---|---:|---:|---:|---:|---:|
| checkpoint | 2894.5 | 1890.2 | 0.0% | 0.0% | 9.7729 |
| neurocache_budget5_checkpoint_cpu_bf16adam | 2456.4 | 1887.6 | 15.1% | -0.1% | 9.7299 |

Interpretation:

- The memory result is stable: budget-5 NeuroCache repeatedly gives about 15.1%
  lower peak CUDA allocation than checkpointing.
- Throughput is statistically close, but the strict equal-or-better throughput
  claim is not fully proven by the longer validation because the mean remains
  0.13% below checkpointing.
- The validation supports a careful claim of "checkpoint-level throughput" or
  "near-equal throughput," not an unqualified throughput win.

Budget sweep, seed 42, 10 measured steps:

| Budget | Peak CUDA MB | Tokens/sec | VRAM vs checkpoint | Throughput vs checkpoint | Offloaded tensors |
|---:|---:|---:|---:|---:|---:|
| 0 | 2545.6 | 1898.2 | 12.1% | +1.6% | 0 |
| 1 | 2527.6 | 1888.1 | 12.7% | +1.0% | 11 |
| 2 | 2509.6 | 1860.4 | 13.3% | -0.4% | 22 |
| 3 | 2491.6 | 1888.5 | 14.0% | +1.1% | 33 |
| 4 | 2473.6 | 1866.3 | 14.6% | -0.1% | 44 |
| 5 | 2455.6 | 1878.6 | 15.2% | +0.5% | 55 |
| 6 | 2437.6 | 1866.0 | 15.8% | -0.1% | 66 |
| 8 | 2401.6 | 1872.1 | 17.1% | +0.2% | 88 |
| 10 | 2365.6 | 1885.4 | 18.3% | +0.9% | 110 |

The sweep shows a clean memory/budget trend and a noisy throughput curve. Budget
5 is the first budget that crosses the 15% memory target in the fixed
configuration. Larger budgets can reduce memory further, but they increase
transfer and packing work and need multi-seed validation before replacing the
locked setting.

Publication artifacts:

- `results/validation/rtx2050_publication/raw_runs.csv`
- `results/validation/rtx2050_publication/raw_runs.json`
- `results/validation/rtx2050_publication/main_result_table.csv`
- `results/validation/rtx2050_publication/multi_seed_stats.csv`
- `results/validation/rtx2050_publication/budget_sweep_table.csv`
- `results/validation/rtx2050_publication/plots/*.png`
- `results/validation/rtx2050_publication/plots/*.svg`
- `results/validation/rtx2050_publication/PAPER_RESULTS_DRAFT.md`
- `results/validation/rtx2050_publication_20step/locked_20step_summary.csv`

## HybridAdam Negative Result

Before the BF16-state optimizer, the best memory result was synchronous
NeuroCache activation offload plus HybridAdam second-moment CPU placement.
It reduced peak CUDA allocation by 21.6%, but remained slower than checkpointing
over the longer run. That path is useful evidence, but not the current best
configuration.

## Main Multistep Result

Command:

```powershell
.venv\Scripts\python.exe experiments\rtx2050_real_benchmark.py --warmup-steps 1 --measure-steps 3 --d-model 768 --n-layers 12 --n-heads 12 --seq-len 768 --batch-size 7 --min-tensor-kb 4096 --variants baseline,checkpoint,neurocache_checkpoint_fp16 --output-dir results\metrics\rtx2050_pressure_768x12_s768_b7_multistep
```

| Variant | Peak CUDA allocated MB | Peak CUDA reserved MB | Tokens/sec | ms/step | VRAM reduction vs baseline | Throughput vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 5745.6 | 6394.0 | 639.7 | 8403.8 | 0.0% | 0.0% |
| checkpoint | 2676.5 | 3472.0 | 2202.9 | 2440.4 | 53.4% | +244.4% |
| neurocache_checkpoint_fp16 | 2454.3 | 3508.0 | 1714.5 | 3135.6 | 57.3% | +168.0% |

Interpretation:

- The baseline exceeds nominal 4GB VRAM and appears to rely on Windows/WDDM memory paging, which makes it much slower.
- Gradient checkpointing is the strongest baseline in this run.
- NeuroCache plus checkpointing reduces peak allocated CUDA memory by an additional 222.3 MB versus checkpointing alone, about 8.3% relative to checkpointing.
- NeuroCache plus checkpointing is slower than checkpointing alone in this implementation because CPU FP16 packing/unpacking costs about 6.1 seconds over the measured run.
- Loss values matched across variants for this synthetic benchmark, so the FP16 offload did not visibly destabilize this short run.

## Batch-8 Stress Notes

At batch size 8, baseline again ran by exceeding nominal VRAM, while checkpointing stayed near 2265.7 MB allocated. Pure `neurocache_cpu_fp16` OOMed during the offload copy. That is a real limitation of this implementation: the copy/pack path can still require temporary GPU working memory under extreme pressure.

The stable batch-8 command excluding the known-OOM pure offload variant was:

```powershell
.venv\Scripts\python.exe experiments\rtx2050_real_benchmark.py --warmup-steps 0 --measure-steps 1 --d-model 768 --n-layers 12 --n-heads 12 --seq-len 768 --batch-size 8 --min-tensor-kb 4096 --variants baseline,checkpoint,neurocache_checkpoint_fp16 --output-dir results\metrics\rtx2050_pressure_768x12_s768_b8_no_oom_order
```

Output artifacts:

- `results/metrics/rtx2050_pressure_768x12_s768_b7_multistep/rtx2050_real_benchmark.json`
- `results/metrics/rtx2050_pressure_768x12_s768_b7_multistep/rtx2050_real_benchmark.csv`
- `results/metrics/rtx2050_pressure_768x12_s768_b8_no_oom_order/rtx2050_real_benchmark.json`
- `results/metrics/rtx2050_pressure_768x12_s768_b8_no_oom_order/rtx2050_real_benchmark.csv`
- `results/metrics/rtx2050_pressure_768x12_s768_b8_cpu_fp16_oom/rtx2050_real_benchmark.json`
- `results/metrics/rtx2050_pressure_768x12_s768_b8_cpu_fp16_oom/rtx2050_real_benchmark.csv`

## Research Direction

The current honest claim is not "NeuroCache beats all techniques." The honest
claim is narrower:

> A real saved-tensor NeuroCache path can provide extra activation-memory
> reduction on top of gradient checkpointing under RTX 2050 memory pressure,
> but the current CPU packing path trades that memory headroom for throughput.

The next research step is to reduce offload overhead with selective prediction,
larger-tensor-only policies, asynchronous host transfer, and possibly CPU pinned
buffer reuse.

## Async + Predictor Follow-Up

After adding CUDA transfer streams, reusable pinned host buffers, transfer
throttling, and a balanced LSTM predictor, the measured batch-7 command was:

```powershell
.venv\Scripts\python.exe experiments\rtx2050_real_benchmark.py --warmup-steps 1 --measure-steps 2 --d-model 768 --n-layers 12 --n-heads 12 --seq-len 768 --batch-size 7 --min-tensor-kb 4096 --max-pending-d2h 2 --predictor-path results\metrics\balanced_predictor_real\balanced_predictor.pt --variants baseline,checkpoint,neurocache_sync_checkpoint_fp16,neurocache_checkpoint_fp16,neurocache_predictor_checkpoint_fp16 --output-dir results\metrics\rtx2050_throttled_async_predictor_b7
```

| Variant | Peak CUDA allocated MB | Tokens/sec | Transfers | Transfer stream ms | Policy keep/offload |
|---|---:|---:|---:|---:|---:|
| baseline | 5745.6 | 572.0 | 0 | 0.0 | - |
| checkpoint | 2676.5 | 1753.7 | 0 | 0.0 | - |
| sync NeuroCache + checkpoint | 2454.3 | 1494.6 | 51 D2H + 51 H2D | 0.0 | 0 / 51 |
| async NeuroCache + checkpoint | 2456.1 | 668.3 | 51 D2H + 51 H2D | 3780.5 | 0 / 51 |
| predictor NeuroCache + checkpoint | 2660.8 | 592.3 | 10 D2H + 10 H2D | 1098.4 | 41 / 10 |

Balanced predictor training:

- Samples: 12,000 total, 3,000 per class
- Classes: KEEP, OFFLOAD_CPU, OFFLOAD_SSD, PREFETCH
- Final accuracy: 0.9796
- Final macro F1: 0.9795
- Artifact: `results/metrics/balanced_predictor_real/balanced_predictor_eval.json`

Interpretation:

- Reusable pinned buffers worked: sync NeuroCache allocated 17 host buffers and reused them 34 times.
- The predictor reduced offloaded tensors from 51 to 10, proving smarter scheduling behavior.
- The predictor variant recovered fewer memory savings because it kept 41 tensors resident.
- The async stream path was slower than the synchronous pinned-buffer path on this Windows/WDDM RTX 2050 run. It raised reserved memory and did not hide enough latency to beat sync. This is a real negative result and should guide the next optimization pass.
- The current best measured tradeoff remains sync NeuroCache + checkpointing: about 8.3% lower peak allocated CUDA memory than checkpointing alone, with lower throughput.
