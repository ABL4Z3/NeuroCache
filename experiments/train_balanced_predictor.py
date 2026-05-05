#!/usr/bin/env python3
"""
Train a balanced NeuroCache scheduling predictor.

This script fixes the old accuracy-paradox problem by constructing an equal
class dataset for KEEP, OFFLOAD_CPU, OFFLOAD_SSD, and PREFETCH and reporting
macro F1 as the primary metric.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neurocache.predictor import ACTIONS, MemoryPredictor


@dataclass
class PredictorTrainConfig:
    samples_per_class: int = 4000
    window_size: int = 10
    epochs: int = 18
    batch_size: int = 256
    learning_rate: float = 1e-3
    seed: int = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_features(label: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Return balanced feature rows for one scheduling class."""
    if label == 0:  # KEEP: small/medium, frequent, recent
        layer = rng.integers(0, 12, n)
        size = rng.uniform(64 * 1024, 3 * 1024 * 1024, n)
        access_count = rng.integers(8, 40, n)
        last_access = rng.integers(0, 8, n)
        freq = rng.uniform(55, 100, n)
        recency = rng.uniform(0.75, 1.0, n)
        reuse = rng.uniform(1, 3, n)
    elif label == 1:  # OFFLOAD_CPU: large-ish, moderate reuse
        layer = rng.integers(0, 16, n)
        size = rng.uniform(3 * 1024 * 1024, 16 * 1024 * 1024, n)
        access_count = rng.integers(4, 20, n)
        last_access = rng.integers(8, 40, n)
        freq = rng.uniform(18, 55, n)
        recency = rng.uniform(0.35, 0.75, n)
        reuse = rng.uniform(4, 18, n)
    elif label == 2:  # OFFLOAD_SSD: very large, cold
        layer = rng.integers(0, 24, n)
        size = rng.uniform(16 * 1024 * 1024, 96 * 1024 * 1024, n)
        access_count = rng.integers(1, 8, n)
        last_access = rng.integers(40, 160, n)
        freq = rng.uniform(0, 18, n)
        recency = rng.uniform(0, 0.35, n)
        reuse = rng.uniform(18, 96, n)
    else:  # PREFETCH: next-soon, medium/high frequency
        layer = rng.integers(0, 24, n)
        size = rng.uniform(512 * 1024, 12 * 1024 * 1024, n)
        access_count = rng.integers(6, 32, n)
        last_access = rng.integers(1, 24, n)
        freq = rng.uniform(35, 90, n)
        recency = rng.uniform(0.45, 0.95, n)
        reuse = rng.uniform(1, 8, n)

    return np.stack([layer, size, access_count, last_access, freq, recency, reuse], axis=1)


def build_balanced_dataset(cfg: PredictorTrainConfig) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(cfg.seed)
    features = []
    labels = []
    for label in range(4):
        rows = class_features(label, cfg.samples_per_class, rng)
        for row in rows:
            trend = np.linspace(0.85, 1.0, cfg.window_size).reshape(-1, 1)
            window = np.repeat(row.reshape(1, -1), cfg.window_size, axis=0)
            window[:, [2, 4, 5]] = window[:, [2, 4, 5]] * trend
            features.append(window)
            labels.append(label)
    X = torch.tensor(np.asarray(features), dtype=torch.float32)
    y = torch.tensor(np.asarray(labels), dtype=torch.long)
    return X, y


def train(cfg: PredictorTrainConfig, output_dir: Path) -> dict:
    set_seed(cfg.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y = build_balanced_dataset(cfg)
    mean = X.mean(dim=(0, 1), keepdim=True)
    std = X.std(dim=(0, 1), keepdim=True).clamp_min(1e-8)
    X = (X - mean) / std

    indices = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=cfg.seed,
        stratify=y.numpy(),
    )
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MemoryPredictor(input_size=7, hidden_size=64, num_layers=2, dropout=0.15, num_classes=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()
    history = []

    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        losses = []
        for start in range(0, len(perm), cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            xb = X_train[idx].to(device)
            yb = y_train[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            logits = model(X_val.to(device))
            preds = logits.argmax(dim=-1).cpu()
            val_loss = float(criterion(logits, y_val.to(device)).detach().cpu())
        macro_f1 = f1_score(y_val.numpy(), preds.numpy(), average="macro", zero_division=0)
        acc = accuracy_score(y_val.numpy(), preds.numpy())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "macro_f1": macro_f1, "accuracy": acc})

    report = classification_report(
        y_val.numpy(),
        preds.numpy(),
        labels=[0, 1, 2, 3],
        target_names=[ACTIONS[i] for i in range(4)],
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(y_val.numpy(), preds.numpy(), labels=[0, 1, 2, 3]).tolist()
    checkpoint = {
        "model_state": model.cpu().state_dict(),
        "norm_mean": mean,
        "norm_std": std,
        "input_size": 7,
        "hidden_size": 64,
        "num_layers": 2,
        "window_size": cfg.window_size,
        "config": asdict(cfg),
    }
    model_path = output_dir / "balanced_predictor.pt"
    torch.save(checkpoint, model_path)

    results = {
        "config": asdict(cfg),
        "model_path": str(model_path),
        "samples_total": int(len(y)),
        "class_distribution": {ACTIONS[i]: int((y == i).sum()) for i in range(4)},
        "final_accuracy": history[-1]["accuracy"],
        "final_macro_f1": history[-1]["macro_f1"],
        "classification_report": report,
        "confusion_matrix": matrix,
        "history": history,
    }
    with (output_dir / "balanced_predictor_eval.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "metrics" / "balanced_predictor"))
    parser.add_argument("--samples-per-class", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = PredictorTrainConfig(
        samples_per_class=args.samples_per_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    results = train(cfg, Path(args.output_dir))
    print(json.dumps({k: v for k, v in results.items() if k not in {"history", "classification_report"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
