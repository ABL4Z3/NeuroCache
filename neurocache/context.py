"""
Phase 5 — Training Loop Integration
NeuroCache context manager for clean API integration with training loops.
"""

import torch
import torch.nn as nn
import time
import psutil
from typing import Optional, Dict, List
from contextlib import contextmanager

from .profiler import MemoryProfiler
from .predictor import MemoryPredictor
from .scheduler import TieredScheduler, MemoryTier
from .prefetch import AsyncPrefetchEngine
from .quantization import QuantizationBridge


class NeuroCacheContext:
    """
    Context manager that wraps a model for NeuroCache-accelerated training.

    Usage:
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        with NeuroCacheContext(model, ram_limit_gb=8) as nc:
            for batch in dataloader:
                loss = model(batch).loss
                loss.backward()
                nc.step()
    """

    def __init__(
        self,
        model: nn.Module,
        ram_limit_gb: float = 8.0,
        use_predictor: bool = True,
        use_prefetch: bool = True,
        use_quantization: bool = True,
        gpu_capacity_mb: float = 4096,
        predictor_path: Optional[str] = None,
        output_dir: str = "./results",
        overhead_threshold: float = 0.05,
    ):
        self.model = model
        self.ram_limit_bytes = ram_limit_gb * 1024 ** 3
        self.use_predictor = use_predictor
        self.use_prefetch = use_prefetch
        self.use_quantization = use_quantization
        self.output_dir = output_dir
        self.overhead_threshold = overhead_threshold

        # Initialize components
        self.profiler = MemoryProfiler(model, output_dir)
        self.predictor = None
        self.scheduler = TieredScheduler(
            gpu_capacity_mb=gpu_capacity_mb,
            ssd_dir=os.path.join(output_dir, "ssd_cache"),
        )
        self.prefetch_engine = None
        self.quant_bridge = None

        if use_predictor and predictor_path:
            self._load_predictor(predictor_path)

        if use_prefetch:
            self.prefetch_engine = AsyncPrefetchEngine(
                num_workers=2,
                scheduler=self.scheduler,
            )

        if use_quantization:
            self.quant_bridge = QuantizationBridge(
                ssd_dir=os.path.join(output_dir, "ssd_cache"),
            )

        # Overhead monitoring
        self.step_times = []
        self.overhead_times = []
        self._fallback_to_rules = False

        # Results tracking
        self.step_metrics = []

    def _load_predictor(self, path: str):
        """Load a trained LSTM predictor."""
        try:
            checkpoint = torch.load(path, weights_only=False)
            self.predictor = MemoryPredictor(
                input_size=checkpoint.get('input_size', 7),
                hidden_size=64,
                num_layers=2,
            )
            self.predictor.load_state_dict(checkpoint['model_state'])
            self.predictor.eval()
            self.scheduler.predictor = self.predictor
            print(f"[NeuroCache] Loaded predictor from {path}")
        except Exception as e:
            print(f"[NeuroCache] Failed to load predictor: {e}. Using rule-based scheduling.")
            self.predictor = None

    def __enter__(self):
        self.profiler.register_hooks()
        if self.prefetch_engine:
            self.prefetch_engine.start()
        print(f"[NeuroCache] Activated — RAM limit: {self.ram_limit_bytes / 1e9:.1f} GB")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.remove_hooks()
        if self.prefetch_engine:
            self.prefetch_engine.stop()
        self.save_results()
        print(f"[NeuroCache] Deactivated — results saved to {self.output_dir}")
        return False

    def step(self, loss: float = 0.0, grad_norm: float = 0.0):
        """
        Call after each training step (after backward pass).
        Profiles memory, evaluates scheduler, and triggers offloading.
        """
        step_start = time.time()
        self.profiler.step()

        # Check RAM pressure
        ram_usage = psutil.virtual_memory()
        ram_used_ratio = ram_usage.used / ram_usage.total

        # Get current model tensors
        tensors = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                tensors[name] = param.data
                if name not in self.scheduler.registry:
                    self.scheduler.register_tensor(name, param.data, layer_index=0)

        # Schedule offloading
        overhead_start = time.time()
        self.scheduler.evaluate_and_offload(tensors, loss=loss, grad_norm=grad_norm)

        # Emergency RAM check
        if ram_used_ratio > 0.85:
            print(f"[NeuroCache] WARNING: RAM at {ram_used_ratio*100:.1f}% — emergency eviction")
            self.scheduler._emergency_evict()

        # Overhead monitoring
        overhead_time = time.time() - overhead_start
        step_time = time.time() - step_start
        self.overhead_times.append(overhead_time)
        self.step_times.append(step_time)

        # Self-regulation: disable predictor if overhead too high
        if len(self.step_times) > 10:
            avg_step = sum(self.step_times[-10:]) / 10
            avg_overhead = sum(self.overhead_times[-10:]) / 10
            if avg_step > 0 and avg_overhead / avg_step > self.overhead_threshold:
                self._fallback_to_rules = True
                self.scheduler.predictor = None
                print(f"[NeuroCache] Overhead {avg_overhead/avg_step*100:.1f}% > {self.overhead_threshold*100:.0f}% — falling back to rule-based")

        # Record metrics
        mem = self.profiler.get_memory_usage()
        sched_stats = self.scheduler.get_stats()
        self.step_metrics.append({
            'step': self.scheduler.current_step,
            'ram_used_pct': ram_used_ratio * 100,
            'rss_mb': mem['rss_mb'],
            'gpu_usage_mb': sched_stats['gpu_usage_mb'],
            'cpu_usage_mb': sched_stats['cpu_usage_mb'],
            'evictions': sched_stats['evictions'],
            'prefetches': sched_stats['prefetches'],
            'emergency_evictions': sched_stats['emergency_evictions'],
            'overhead_ms': overhead_time * 1000,
        })

    def prefetch(self, layer_names: List[str]):
        """Prefetch tensors for upcoming layers."""
        if self.prefetch_engine:
            self.prefetch_engine.prefetch_next_layers(0, layer_names)
        else:
            self.scheduler.prefetch(layer_names)

    def save_results(self):
        """Save all profiling and scheduling results."""
        import json
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        # Save profiler data
        self.profiler.save_records()
        self.profiler.save_stats()

        # Save scheduler stats
        self.scheduler.save_stats(os.path.join(self.output_dir, 'scheduler_stats.json'))

        # Save step metrics
        with open(os.path.join(self.output_dir, 'neurocache_metrics.json'), 'w') as f:
            json.dump(self.step_metrics, f, indent=2)

        # Save quantization stats
        if self.quant_bridge:
            with open(os.path.join(self.output_dir, 'quantization_stats.json'), 'w') as f:
                json.dump(self.quant_bridge.get_stats(), f, indent=2)

    def get_summary(self) -> dict:
        """Get a summary of the NeuroCache session."""
        return {
            'profiler': self.profiler.get_summary(),
            'scheduler': self.scheduler.get_stats(),
            'overhead_fallback': self._fallback_to_rules,
            'total_steps': len(self.step_metrics),
            'avg_overhead_ms': sum(self.overhead_times) / max(len(self.overhead_times), 1) * 1000,
        }


import os
