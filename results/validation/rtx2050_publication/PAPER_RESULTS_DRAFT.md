# NeuroCache RTX 2050 Validation Draft

## Proof of Achievement

Configuration: checkpointed transformer training with BF16 CUDA Adam moments and
budgeted NeuroCache activation offload through PyTorch saved-tensor hooks. The
locked NeuroCache budget is 5 tensors per step. The GPU is
NVIDIA GeForce RTX 2050 with 4096.0
MiB reported by `nvidia-smi`.

Across the five-seed validation, checkpointing used
2895.8 MB peak CUDA allocation and
achieved 1890.0 tokens/sec on average.
NeuroCache used 2455.6 MB and achieved
1885.0 tokens/sec. This corresponds to
15.20% lower peak CUDA memory and
0.26% lower throughput relative to gradient checkpointing. The strict mean-throughput criterion is not fully satisfied in this dataset; throughput is statistically close but slightly lower in the mean.

## Results

We evaluated a 97.93M-parameter GPT-like decoder at sequence length 768 and
batch size 8 on an RTX 2050. Each run used one warmup step and
10 measured training steps. We compare ordinary training,
gradient checkpointing, checkpointing with BF16 optimizer moments, and the
locked NeuroCache configuration.

The main result shows that NeuroCache reaches the target memory envelope with
near-equal throughput. Peak CUDA allocation is reduced by more than 15% relative
to gradient checkpointing. No run reported CUDA OOM or non-finite final loss.

A longer 20-step repeat was run for the two most important variants,
checkpointing and locked budget-5 NeuroCache, across the same five seeds. In
that repeat, checkpointing averaged 2894.5 MB and 1890.2 tokens/sec, while
NeuroCache averaged 2456.4 MB and 1887.6 tokens/sec. This gives 15.14% lower
peak CUDA memory and 0.13% lower mean throughput. The longer repeat confirms
the memory result and shows that the throughput gap is very small, but it still
does not prove a strict throughput win.

## Analysis

The budget sweep shows a clear memory/offload relationship and a noisy
throughput relationship. Budget 5 is the first budget that crosses the 15%
memory-reduction target in the fixed configuration. Budget 10 gave the strongest
single-seed point in the sweep, but it was not promoted to the main claim
because the locked validation configuration is budget 5 and larger budgets need
their own multi-seed validation. Too little offloading, especially budget 0-4,
does not remove enough activation memory to satisfy the 15% target. Too much
offloading reduces memory further but increases host transfer work and packing
time. The tradeoff curve therefore has a knee near budget 5: it crosses the
memory target while keeping transfer count low.

The result also explains why unbounded activation offload was slower in earlier
experiments. Moving every eligible saved tensor shifts too much work to CPU
memory transfers. Budgeting keeps the offload path selective and turns
activation movement into a small memory-pressure relief mechanism instead of a
dominant cost.

## Conclusion

These experiments provide real CUDA evidence that a budgeted NeuroCache policy
can reduce peak GPU memory by at least 15% relative to gradient checkpointing
while maintaining statistically comparable throughput on low-VRAM hardware. The
claim should remain scoped: the strict mean-throughput win must be treated as
not yet proven when the measured mean is negative, the optimizer moments use
BF16 rather than exact FP32 AdamW state, and longer convergence experiments are
needed before making final model quality claims. DeepSpeed comparison remains a
separate pending validation.

## Reproducibility

Run the validation suite:

```powershell
.venv\Scripts\python.exe experiments\validate_neurocache_rtx2050.py --steps 10 --output-dir results\validation\rtx2050_publication
```

Primary outputs are in `results/validation/rtx2050_publication`:

- `raw_runs.csv` and `raw_runs.json`
- `main_result_table.csv`
- `multi_seed_stats.csv`
- `budget_sweep_table.csv`
- `plots/*.png` and `plots/*.svg`
