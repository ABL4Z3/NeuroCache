"""
Phase 4 — 3-Tier Scheduler
Implements GPU / CPU-Pinned / SSD tiered memory scheduling with emergency eviction.
"""

import torch
import numpy as np
import time
import mmap
import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class MemoryTier(Enum):
    GPU = 0
    CPU_PINNED = 1
    SSD = 2


@dataclass
class TensorEntry:
    """Metadata for a managed tensor."""
    name: str
    tensor: Optional[torch.Tensor]
    tier: MemoryTier
    score: float  # predictor score: 0-1, higher = keep on GPU
    size_bytes: int
    layer_index: int
    last_access_step: int
    quantized: bool = False
    ssd_path: Optional[str] = None


class TieredScheduler:
    """
    3-tier memory scheduler: GPU -> CPU Pinned -> SSD
    Uses predictor scores to determine tensor placement.
    Implements emergency eviction when GPU memory exceeds threshold.
    """

    def __init__(
        self,
        gpu_capacity_mb: float = 4096,
        gpu_threshold: float = 0.90,
        emergency_threshold: float = 0.85,
        keep_threshold: float = 0.7,
        cpu_threshold: float = 0.4,
        ssd_dir: str = "./ssd_cache",
        predictor=None,
    ):
        self.gpu_capacity_mb = gpu_capacity_mb
        self.gpu_threshold = gpu_threshold
        self.emergency_threshold = emergency_threshold
        self.keep_threshold = keep_threshold
        self.cpu_threshold = cpu_threshold
        self.ssd_dir = ssd_dir
        self.predictor = predictor

        os.makedirs(ssd_dir, exist_ok=True)

        self.registry: Dict[str, TensorEntry] = {}
        self.current_step = 0
        self.stats = {
            'gpu_hits': 0, 'cpu_hits': 0, 'ssd_hits': 0,
            'evictions': 0, 'prefetches': 0, 'emergency_evictions': 0,
        }

    def _tensor_size_mb(self, tensor: torch.Tensor) -> float:
        if tensor is None:
            return 0.0
        return tensor.element_size() * tensor.nelement() / 1024 / 1024

    def _gpu_usage_mb(self) -> float:
        """Current GPU memory usage in MB."""
        total = sum(
            self._tensor_size_mb(e.tensor)
            for e in self.registry.values()
            if e.tier == MemoryTier.GPU and e.tensor is not None
        )
        return total

    def _cpu_usage_mb(self) -> float:
        """Current CPU pinned memory usage in MB."""
        total = sum(
            self._tensor_size_mb(e.tensor)
            for e in self.registry.values()
            if e.tier == MemoryTier.CPU_PINNED and e.tensor is not None
        )
        return total

    def register_tensor(self, name: str, tensor: torch.Tensor, layer_index: int = 0, score: float = 1.0):
        """Register a tensor with the scheduler."""
        entry = TensorEntry(
            name=name,
            tensor=tensor,
            tier=MemoryTier.GPU,
            score=score,
            size_bytes=tensor.element_size() * tensor.nelement(),
            layer_index=layer_index,
            last_access_step=self.current_step,
        )
        self.registry[name] = entry

    def evaluate_and_offload(self, tensors: Dict[str, torch.Tensor], loss: float = 0.0, grad_norm: float = 0.0):
        """
        Evaluate all registered tensors and offload based on predictor scores.
        Called after backward pass.
        """
        self.current_step += 1

        for name, tensor in tensors.items():
            if name in self.registry:
                self.registry[name].last_access_step = self.current_step
                self.registry[name].tensor = tensor

        # Score all tensors
        for name, entry in self.registry.items():
            if self.predictor is not None:
                score = self._predict_score(entry, loss, grad_norm)
                entry.score = score
            else:
                # Rule-based scoring
                entry.score = self._rule_based_score(entry)

        # Schedule tiers based on scores
        self._schedule_tiers()

        # Check for emergency eviction
        gpu_usage_ratio = self._gpu_usage_mb() / self.gpu_capacity_mb
        if gpu_usage_ratio > self.emergency_threshold:
            self._emergency_evict()

    def _predict_score(self, entry: TensorEntry, loss: float, grad_norm: float) -> float:
        """Use LSTM predictor to score a tensor."""
        try:
            features = torch.tensor([[
                self.current_step,
                entry.layer_index,
                entry.size_bytes / 1e6,  # normalize to MB
                self.current_step - entry.last_access_step,
                0.0,  # frequency placeholder
                loss,
                grad_norm,
            ]], dtype=torch.float32)

            action_id, confidence = self.predictor.predict(features)

            # Convert action to score
            score_map = {
                0: 0.9,   # KEEP -> high score
                3: 0.75,  # PREFETCH -> moderate-high
                1: 0.5,   # OFFLOAD_CPU -> moderate
                2: 0.2,   # OFFLOAD_SSD -> low
            }
            return score_map.get(action_id, 0.5) * confidence
        except Exception:
            return self._rule_based_score(entry)

    def _rule_based_score(self, entry: TensorEntry) -> float:
        """Rule-based scoring when predictor is unavailable."""
        steps_since_access = self.current_step - entry.last_access_step
        recency = max(0, 1.0 - steps_since_access / 100)
        size_penalty = min(1.0, entry.size_bytes / 100e6)  # penalize large tensors
        return recency * 0.7 + (1 - size_penalty) * 0.3

    def _schedule_tiers(self):
        """Move tensors between tiers based on scores."""
        for name, entry in self.registry.items():
            if entry.tensor is None:
                continue

            new_tier = self._score_to_tier(entry.score)

            if new_tier != entry.tier:
                self._move_tensor(entry, new_tier)

    def _score_to_tier(self, score: float) -> MemoryTier:
        """Map a score to a memory tier."""
        if score >= self.keep_threshold:
            return MemoryTier.GPU
        elif score >= self.cpu_threshold:
            return MemoryTier.CPU_PINNED
        else:
            return MemoryTier.SSD

    def _move_tensor(self, entry: TensorEntry, target_tier: MemoryTier):
        """Move a tensor to a different tier."""
        try:
            if target_tier == MemoryTier.GPU:
                if entry.tier == MemoryTier.CPU_PINNED and entry.tensor is not None:
                    entry.tensor = entry.tensor.to('cpu')  # Would be .to('cuda') with GPU
                    entry.tier = MemoryTier.GPU
                    self.stats['gpu_hits'] += 1
                elif entry.tier == MemoryTier.SSD:
                    entry.tensor = self._load_from_ssd(entry)
                    entry.tier = MemoryTier.GPU
                    self.stats['gpu_hits'] += 1

            elif target_tier == MemoryTier.CPU_PINNED:
                if entry.tier == MemoryTier.GPU and entry.tensor is not None:
                    entry.tensor = entry.tensor.cpu()
                    if not entry.tensor.is_pinned():
                        try:
                            entry.tensor = entry.tensor.pin_memory()
                        except Exception:
                            pass
                    entry.tier = MemoryTier.CPU_PINNED
                    self.stats['cpu_hits'] += 1
                    self.stats['evictions'] += 1
                elif entry.tier == MemoryTier.SSD:
                    entry.tensor = self._load_from_ssd(entry)
                    entry.tier = MemoryTier.CPU_PINNED
                    self.stats['cpu_hits'] += 1

            elif target_tier == MemoryTier.SSD:
                if entry.tensor is not None:
                    self._save_to_ssd(entry)
                    entry.tensor = None  # Free memory
                    entry.tier = MemoryTier.SSD
                    self.stats['ssd_hits'] += 1
                    self.stats['evictions'] += 1

        except Exception as e:
            print(f"[Scheduler] Error moving {entry.name}: {e}")

    def _save_to_ssd(self, entry: TensorEntry):
        """Save tensor to SSD via mmap."""
        ssd_path = os.path.join(self.ssd_dir, f"{entry.name.replace('.', '_')}.pt")
        torch.save(entry.tensor.cpu(), ssd_path)
        entry.ssd_path = ssd_path
        entry.quantized = False

    def _load_from_ssd(self, entry: TensorEntry) -> torch.Tensor:
        """Load tensor from SSD."""
        if entry.ssd_path and os.path.exists(entry.ssd_path):
            tensor = torch.load(entry.ssd_path, weights_only=True)
            return tensor
        return torch.zeros(1)  # fallback

    def _emergency_evict(self):
        """Emergency eviction: move bottom 20% scored tensors off GPU."""
        gpu_entries = [
            (name, entry) for name, entry in self.registry.items()
            if entry.tier == MemoryTier.GPU and entry.tensor is not None
        ]
        gpu_entries.sort(key=lambda x: x[1].score)

        evict_count = max(1, len(gpu_entries) // 5)  # bottom 20%
        for name, entry in gpu_entries[:evict_count]:
            self._move_tensor(entry, MemoryTier.CPU_PINNED)
            self.stats['emergency_evictions'] += 1

        print(f"[Scheduler] Emergency eviction: moved {evict_count} tensors to CPU")

    def prefetch(self, layer_names: List[str]):
        """Prefetch tensors for upcoming layers."""
        for name in layer_names:
            if name in self.registry:
                entry = self.registry[name]
                if entry.tier in (MemoryTier.CPU_PINNED, MemoryTier.SSD):
                    if entry.tier == MemoryTier.SSD:
                        entry.tensor = self._load_from_ssd(entry)
                    entry.tier = MemoryTier.GPU
                    self.stats['prefetches'] += 1

    def get_stats(self) -> dict:
        """Return scheduler statistics."""
        tier_counts = {tier: 0 for tier in MemoryTier}
        tier_sizes = {tier: 0.0 for tier in MemoryTier}
        for entry in self.registry.values():
            tier_counts[entry.tier] += 1
            size_mb = self._tensor_size_mb(entry.tensor) if entry.tensor is not None else 0
            tier_sizes[entry.tier] += size_mb

        return {
            'step': self.current_step,
            'gpu_usage_mb': self._gpu_usage_mb(),
            'cpu_usage_mb': self._cpu_usage_mb(),
            'gpu_capacity_mb': self.gpu_capacity_mb,
            'tier_counts': {tier.name: count for tier, count in tier_counts.items()},
            'tier_sizes_mb': {tier.name: size for tier, size in tier_sizes.items()},
            'total_tensors': len(self.registry),
            **self.stats,
        }

    def save_stats(self, filepath: str):
        """Save scheduler stats to JSON."""
        stats = self.get_stats()
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
