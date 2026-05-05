"""
Phase 3 — LSTM Memory Predictor
2-layer LSTM that processes rolling windows of tensor access logs to predict
the optimal action: KEEP, OFFLOAD_CPU, OFFLOAD_SSD, or PREFETCH.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from typing import List, Tuple, Optional


# Action labels
ACTIONS = {0: 'KEEP', 1: 'OFFLOAD_CPU', 2: 'OFFLOAD_SSD', 3: 'PREFETCH'}
ACTION_IDS = {v: k for k, v in ACTIONS.items()}


class MemoryPredictor(nn.Module):
    """
    2-layer LSTM predictor for tensor memory scheduling decisions.
    Input: rolling window of tensor features [step, layer_idx, tensor_size, last_access, frequency, loss, grad_norm]
    Output: probability distribution over {KEEP, OFFLOAD_CPU, OFFLOAD_SSD, PREFETCH}
    """

    def __init__(self, input_size=7, hidden_size=64, num_layers=2, dropout=0.2, num_classes=4):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        # Attention mechanism for weighting recent accesses
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size) — rolling window of features
        Returns:
            (batch, num_classes) — action probabilities
        """
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)

        # Attention weights
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # normalize over seq

        # Weighted sum
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_size)

        # Classify
        logits = self.classifier(context)  # (batch, num_classes)
        return logits

    def predict(self, x: torch.Tensor) -> Tuple[int, float]:
        """Predict single action from a feature window."""
        self.eval()
        with torch.no_grad():
            if x.dim() == 2:
                x = x.unsqueeze(0)  # add batch dim
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            action_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, action_id].item()
        return action_id, confidence


def create_training_labels(stats_df: pd.DataFrame, total_steps: int) -> pd.DataFrame:
    """
    Generate training labels based on access patterns.
    Heuristic labeling:
    - KEEP: high frequency, recent access, small size
    - OFFLOAD_CPU: moderate frequency, moderate recency
    - OFFLOAD_SSD: low frequency, old access, large size
    - PREFETCH: moderate frequency but increasing trend
    """
    df = stats_df.copy()

    # Normalize features
    freq = df['access_frequency'].astype(float)
    recency = df['recency_score'].astype(float)
    reuse = df['reuse_distance'].astype(float).replace(float('inf'), total_steps)
    size = df['total_size_bytes'].astype(float)

    freq_norm = (freq - freq.min()) / (freq.max() - freq.min() + 1e-8)
    recency_norm = (recency - recency.min()) / (recency.max() - recency.min() + 1e-8)
    reuse_norm = (reuse - reuse.min()) / (reuse.max() - reuse.min() + 1e-8)
    size_norm = (size - size.min()) / (size.max() - size.min() + 1e-8)

    # Scoring system for labeling
    keep_score = freq_norm * 0.4 + recency_norm * 0.4 + (1 - size_norm) * 0.2
    cpu_score = freq_norm * 0.2 + recency_norm * 0.2 + reuse_norm * 0.3 + size_norm * 0.3
    ssd_score = (1 - freq_norm) * 0.3 + (1 - recency_norm) * 0.3 + reuse_norm * 0.2 + size_norm * 0.2
    prefetch_score = freq_norm * 0.3 + recency_norm * 0.1 + (1 - reuse_norm) * 0.3 + (1 - size_norm) * 0.3

    scores = torch.tensor([keep_score.values, cpu_score.values, ssd_score.values, prefetch_score.values])
    labels = scores.argmax(dim=0).numpy()

    df['label'] = labels
    df['label_name'] = df['label'].map(ACTIONS)

    return df


def build_sequence_dataset(stats_df: pd.DataFrame, window_size: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build rolling-window sequences for LSTM training.
    Each sample is a window of consecutive tensor access features.
    """
    features = []
    labels = []

    feature_cols = ['layer_index', 'total_size_bytes', 'access_count',
                    'last_access_step', 'access_frequency', 'recency_score', 'reuse_distance']

    # Group by tensor name and create sequences
    for name, group in stats_df.groupby('name'):
        group = group.sort_values('last_access_step')
        if len(group) < window_size:
            # Pad with zeros
            feature_vals = group[feature_cols].values.astype(float)
            pad_len = window_size - len(feature_vals)
            feature_vals = np.vstack([np.zeros((pad_len, len(feature_cols))), feature_vals])
            label = group['label'].iloc[-1]
            features.append(feature_vals)
            labels.append(label)
        else:
            # Use the last window
            feature_vals = group[feature_cols].values[-window_size:].astype(float)
            label = group['label'].iloc[-1]
            features.append(feature_vals)
            labels.append(label)

    # Also create synthetic sequences by varying the window
    all_features = stats_df[feature_cols].values.astype(float)
    all_labels = stats_df['label'].values.astype(int)

    # Create additional sequences with noise for data augmentation
    for i in range(len(all_features)):
        feat = all_features[i]
        label = all_labels[i]
        # Create a window by repeating with noise
        window = np.tile(feat, (window_size, 1))
        noise = np.random.randn(*window.shape) * 0.01
        window = window + noise
        features.append(window)
        labels.append(label)

    X = torch.tensor(np.array(features), dtype=torch.float32)
    y = torch.tensor(np.array(labels), dtype=torch.long)

    return X, y


def train_predictor(
    stats_csv_path: str,
    output_dir: str = "./results",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    window_size: int = 10,
) -> dict:
    """
    Train the LSTM predictor on memory profiling data.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load stats
    df = pd.read_csv(stats_csv_path)
    print(f"[Predictor] Loaded {len(df)} tensor records from {stats_csv_path}")

    # Create labels
    total_steps = df['last_access_step'].max() if 'last_access_step' in df.columns else 100
    df = create_training_labels(df, total_steps)
    label_dist = df['label_name'].value_counts()
    print(f"[Predictor] Label distribution:\n{label_dist}")

    # Build sequences
    X, y = build_sequence_dataset(df, window_size=window_size)
    print(f"[Predictor] Dataset: {X.shape[0]} sequences, window_size={window_size}")

    # Normalize features
    mean = X.mean(dim=(0, 1), keepdim=True)
    std = X.std(dim=(0, 1), keepdim=True) + 1e-8
    X_norm = (X - mean) / std

    # Train/eval split
    split_idx = int(len(X_norm) * 0.8)
    indices = torch.randperm(len(X_norm))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    X_train, y_train = X_norm[train_idx], y[train_idx]
    X_val, y_val = X_norm[val_idx], y[val_idx]

    print(f"[Predictor] Train: {len(X_train)}, Val: {len(X_val)}")

    # Initialize model
    model = MemoryPredictor(
        input_size=X_train.shape[-1],
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        num_classes=4,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    best_val_acc = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        perm = torch.randperm(len(X_train))
        X_shuf, y_shuf = X_train[perm], y_train[perm]

        for i in range(0, len(X_shuf), batch_size):
            xb = X_shuf[i:i+batch_size]
            yb = y_shuf[i:i+batch_size]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val).item()
            val_preds = val_logits.argmax(dim=-1)
            val_acc = (val_preds == y_val).float().mean().item()

        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state': model.state_dict(),
                'norm_mean': mean,
                'norm_std': std,
                'input_size': X_train.shape[-1],
                'window_size': window_size,
            }, os.path.join(output_dir, 'predictor.pt'))

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    print(f"[Predictor] Training complete. Best val accuracy: {best_val_acc:.4f}")

    # Save training history
    import json
    with open(os.path.join(output_dir, 'predictor_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        val_preds = val_logits.argmax(dim=-1)

    from sklearn.metrics import classification_report, confusion_matrix
    present_labels = sorted(set(y_val.numpy().tolist()) | set(val_preds.numpy().tolist()))
    target_names_all = [ACTIONS[i] for i in range(4)]
    report = classification_report(y_val.numpy(), val_preds.numpy(),
                                   labels=list(range(4)),
                                   target_names=target_names_all,
                                   zero_division=0)
    cm = confusion_matrix(y_val.numpy(), val_preds.numpy(), labels=list(range(4)))

    results = {
        'best_val_accuracy': best_val_acc,
        'total_params': sum(p.numel() for p in model.parameters()),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'history': history,
    }

    with open(os.path.join(output_dir, 'predictor_eval.json'), 'w') as f:
        json.dump({k: v for k, v in results.items() if k != 'history'}, f, indent=2, default=str)

    return results
