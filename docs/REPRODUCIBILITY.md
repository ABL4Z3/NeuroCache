# Reproducibility Guide

This document explains how to independently verify all results reported in the NeuroCache paper.

## Prerequisites

```bash
pip install -r requirements.txt
```

Required packages: PyTorch 2.0+, Transformers, scikit-learn, numpy, pandas, matplotlib, psutil

## Step-by-Step Verification

### 1. Phase 1: Predictor Validation (Table 3)

```bash
cd experiments
python phase1_predictor_validation.py
```

This script:
- Generates **balanced** data (50,000 samples, equal class distribution)
- Generates **degenerate** data (97% SSD class, 1.5% each for KEEP/CPU)
- Trains a neural predictor (3-layer MLP) on balanced data for 10 epochs
- Compares against a rule-based predictor
- Reports accuracy, macro F1, and per-class F1 for both data types

**Expected results:**
- Balanced: neural accuracy ~85%, macro F1 ~0.85
- Degenerate: 97% accuracy but only 0.25 macro F1 (accuracy paradox)

All random seeds are fixed (`np.random.seed(42)`) for reproducibility.

### 2. Full Pipeline (Tables 1, 2, 4)

```bash
python run_all_phases.py
```

This runs all 10 phases sequentially, producing:
- `results/metrics/benchmark_results.csv` (Table 1 data)
- `results/metrics/ablation_results.csv` (Table 2 data)
- `results/metrics/quantization_results.csv` (Table 4 data)
- `results/metrics/all_phase_results.json` (complete run data)

### 3. Verify CSV Results

All result CSVs are included in `results/metrics/`. You can verify:
- Each row matches the numbers reported in the paper
- Running the experiments produces the same values (deterministic seeds)

### 4. Visualizations

The `results/visualizations/` directory contains:
- `neurocache_dashboard.png` — Memory usage over time
- `predictor_training.png` — Training/validation curves
- `quantization_comparison.png` — Compression vs. error tradeoff

## Key Design Decisions for Reproducibility

| Decision | Why It Matters |
|----------|---------------|
| Fixed random seeds (42) | Same data split, same initialization, same results every run |
| Balanced training data | Avoids the accuracy paradox (97% accuracy = useless predictor) |
| Multiple baselines | DeepSpeed, gradient checkpointing, rule-based — not cherry-picked |
| Ablation study | Each component tested independently |
| Standard metrics | F1, precision, recall — not just accuracy |
| Open data | All CSVs and JSONs included — verify without running |

## Environment Notes

- Experiments were run on CPU-simulated CUDA (8.68GB RAM, 4 cores)
- PyTorch 2.11.0, Transformers 5.7.0
- Results may vary slightly on different hardware due to floating-point differences
- The scheduling logic and decision quality are hardware-independent

## Reporting Issues

If you cannot reproduce the results, please open an issue at:
https://github.com/abl4z3/NeuroCache/issues
