#!/usr/bin/env python3
"""
Phase 1: Neural Predictor Validation with Mixed-Label Data
Generates balanced multi-class data and trains LSTM + rule-based predictors.
"""
import numpy as np
import json
import os
import warnings
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score, precision_recall_fscore_support)
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')
OUTPUT_DIR = './results'
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(42)

LABEL_NAMES = ['Keep_GPU', 'Offload_CPU', 'Offload_SSD']

def generate_balanced_data(n_samples=50000):
    records = []
    for i in range(n_samples // 3):
        layer_idx = np.random.randint(0, 12)
        tensor_type = np.random.choice([0, 1])
        tensor_size_mb = np.random.uniform(2, 60)
        last_used = np.random.randint(0, 5)
        freq = np.random.uniform(0.55, 1.0)
        phase = np.random.choice([0, 1])
        records.append([layer_idx, tensor_type, tensor_size_mb, last_used, freq, phase, 0])
    for i in range(n_samples // 3):
        layer_idx = np.random.randint(0, 12)
        tensor_type = np.random.randint(0, 3)
        tensor_size_mb = np.random.uniform(40, 150)
        last_used = np.random.randint(3, 20)
        freq = np.random.uniform(0.15, 0.55)
        phase = np.random.choice([0, 1, 2])
        records.append([layer_idx, tensor_type, tensor_size_mb, last_used, freq, phase, 1])
    for i in range(n_samples // 3):
        layer_idx = np.random.randint(0, 12)
        tensor_type = 2
        tensor_size_mb = np.random.uniform(100, 400)
        last_used = np.random.randint(15, 100)
        freq = np.random.uniform(0.0, 0.15)
        phase = np.random.choice([0, 2])
        records.append([layer_idx, tensor_type, tensor_size_mb, last_used, freq, phase, 2])
    np.random.shuffle(records)
    arr = np.array(records, dtype=np.float64)
    return arr[:, :6], arr[:, 6].astype(np.int64)

def generate_degenerate_data(n_samples=50000):
    records = []
    for i in range(int(n_samples * 0.97)):
        records.append([np.random.randint(0,12), np.random.randint(0,3), np.random.uniform(5,200),
                       np.random.randint(30,100), np.random.uniform(0.0,0.05), 0, 2])
    for i in range(int(n_samples * 0.015)):
        records.append([np.random.randint(0,12), 0, np.random.uniform(2,30),
                       np.random.randint(0,3), np.random.uniform(0.7,1.0), 0, 0])
    for i in range(int(n_samples * 0.015)):
        records.append([np.random.randint(0,12), 1, np.random.uniform(30,80),
                       np.random.randint(3,10), np.random.uniform(0.3,0.6), 1, 1])
    np.random.shuffle(records)
    arr = np.array(records, dtype=np.float64)
    return arr[:, :6], arr[:, 6].astype(np.int64)

class NeuralPredictor:
    def __init__(self, input_size=6, hidden1=128, hidden2=64, num_classes=3, lr=0.01):
        self.lr = lr
        s1 = np.sqrt(2.0 / input_size)
        s2 = np.sqrt(2.0 / hidden1)
        s3 = np.sqrt(2.0 / hidden2)
        self.W1 = np.random.randn(hidden1, input_size) * s1
        self.b1 = np.zeros(hidden1)
        self.W2 = np.random.randn(hidden2, hidden1) * s2
        self.b2 = np.zeros(hidden2)
        self.W3 = np.random.randn(num_classes, hidden2) * s3
        self.b3 = np.zeros(num_classes)

    def _relu(self, x): return np.maximum(0, x)
    def _softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / (e.sum() + 1e-12)

    def forward(self, x):
        self.z1 = self.W1 @ x + self.b1; self.a1 = self._relu(self.z1)
        self.z2 = self.W2 @ self.a1 + self.b2; self.a2 = self._relu(self.z2)
        self.z3 = self.W3 @ self.a2 + self.b3; self.probs = self._softmax(self.z3)
        return self.probs

    def train_step(self, x, target):
        probs = self.forward(x)
        loss = -np.log(probs[target] + 1e-12)
        grad_z3 = probs.copy(); grad_z3[target] -= 1.0
        grad_W3 = np.outer(grad_z3, self.a2); grad_b3 = grad_z3
        grad_a2 = self.W3.T @ grad_z3; grad_z2 = grad_a2 * (self.z2 > 0)
        grad_W2 = np.outer(grad_z2, self.a1); grad_b2 = grad_z2
        grad_a1 = self.W2.T @ grad_z2; grad_z1 = grad_a1 * (self.z1 > 0)
        grad_W1 = np.outer(grad_z1, x); grad_b1 = grad_z1
        clip = 5.0
        for g in [grad_W1, grad_b1, grad_W2, grad_b2, grad_W3, grad_b3]:
            np.clip(g, -clip, clip, out=g)
        self.W3 -= self.lr * grad_W3; self.b3 -= self.lr * grad_b3
        self.W2 -= self.lr * grad_W2; self.b2 -= self.lr * grad_b2
        self.W1 -= self.lr * grad_W1; self.b1 -= self.lr * grad_b1
        return loss, probs

    def predict_batch(self, X):
        return np.array([np.argmax(self.forward(x)) for x in X])

class RuleBasedPredictor:
    def predict_batch(self, X):
        preds = []
        for f in X:
            _, ttype, size_mb, last_used, freq, phase = f
            if phase == 2 and ttype == 2: preds.append(2)
            elif freq > 0.5 and size_mb < 60: preds.append(0)
            elif freq > 0.15 or last_used < 10: preds.append(1)
            else: preds.append(2)
        return np.array(preds)

def evaluate_predictor(y_true, y_pred, label_names):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0,1,2], zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    return {'accuracy': float(acc), 'macro_f1': float(macro_f1),
            'per_class_f1': {label_names[i]: float(f1[i]) for i in range(3)},
            'confusion_matrix': cm.tolist()}

if __name__ == '__main__':
    print("Phase 1: Predictor Validation")
    X_bal, y_bal = generate_balanced_data(50000)
    X_deg, y_deg = generate_degenerate_data(50000)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_bal)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_bal, test_size=0.2, random_state=42, stratify=y_bal)
    predictor = NeuralPredictor(input_size=6, hidden1=128, hidden2=64, num_classes=3, lr=0.01)
    for epoch in range(10):
        total_loss = 0; correct = 0
        indices = np.random.permutation(len(X_train))[:5000]
        for idx in indices:
            loss, probs = predictor.train_step(X_train[idx], y_train[idx])
            total_loss += loss
            if np.argmax(probs) == y_train[idx]: correct += 1
        print(f"  Epoch {epoch+1}/10: loss={total_loss/5000:.4f}, acc={correct/5000:.4f}")
    neural_preds = predictor.predict_batch(X_test)
    rule_preds = RuleBasedPredictor().predict_batch(scaler.inverse_transform(X_test))
    print(f"Neural: Acc={accuracy_score(y_test, neural_preds):.4f}, F1={f1_score(y_test, neural_preds, average='macro'):.4f}")
    print(f"Rule:   Acc={accuracy_score(y_test, rule_preds):.4f}, F1={f1_score(y_test, rule_preds, average='macro'):.4f}")
    print("Done!")
