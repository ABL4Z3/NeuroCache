"""
Phase 2 — Memory Profiler
Registers forward/backward hooks on model layers to track tensor-level memory access patterns.
Outputs: memory_dataset.csv with per-tensor access statistics.
"""

import torch
import torch.nn as nn
import time
import csv
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class TensorRecord:
    """Record for a single tensor access event."""
    name: str
    layer_index: int
    size_bytes: int
    step: int
    access_type: str  # 'read' (forward) or 'write' (backward)
    timestamp: float = 0.0


@dataclass
class TensorStats:
    """Aggregated statistics for a tensor across training."""
    name: str
    layer_index: int
    total_size_bytes: int
    access_count: int = 0
    read_count: int = 0
    write_count: int = 0
    last_access_step: int = 0
    first_access_step: int = -1
    access_steps: list = field(default_factory=list)

    @property
    def access_frequency(self) -> float:
        """Accesses per 100 steps."""
        if not self.access_steps:
            return 0.0
        span = max(self.access_steps) - min(self.access_steps) + 1
        return (self.access_count / max(span, 1)) * 100

    @property
    def recency_score(self) -> float:
        """How recently was this tensor accessed (1.0 = just now, 0.0 = long ago)."""
        if not self.access_steps:
            return 0.0
        total_steps = max(self.access_steps)
        if total_steps == 0:
            return 1.0
        return max(self.access_steps) / total_steps

    @property
    def reuse_distance(self) -> float:
        """Average steps between consecutive accesses."""
        if len(self.access_steps) < 2:
            return float('inf')
        sorted_steps = sorted(self.access_steps)
        distances = [sorted_steps[i+1] - sorted_steps[i] for i in range(len(sorted_steps)-1)]
        return sum(distances) / len(distances)


class MemoryProfiler:
    """
    Hooks into a PyTorch model to profile tensor-level memory access patterns.
    Generates the dataset needed to train the LSTM predictor.
    """

    def __init__(self, model: nn.Module, output_dir: str = "./results"):
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.hooks = []
        self.records: List[TensorRecord] = []
        self.tensor_stats: Dict[str, TensorStats] = {}
        self.current_step = 0
        self.layer_index_map = {}
        self._build_layer_index()

    def _build_layer_index(self):
        """Assign each named module a layer index."""
        for idx, (name, _) in enumerate(self.model.named_modules()):
            self.layer_index_map[name] = idx

    def _get_tensor_size(self, tensor: torch.Tensor) -> int:
        """Calculate tensor size in bytes."""
        if tensor is None:
            return 0
        return tensor.element_size() * tensor.nelement()

    def _forward_hook(self, name: str, layer_idx: int):
        """Create a forward hook for a module."""
        def hook(module, input, output):
            step = self.current_step
            # Profile input tensors
            for i, inp in enumerate(input):
                if isinstance(inp, torch.Tensor):
                    rec = TensorRecord(
                        name=f"{name}.input_{i}",
                        layer_index=layer_idx,
                        size_bytes=self._get_tensor_size(inp),
                        step=step,
                        access_type='read',
                        timestamp=time.time()
                    )
                    self.records.append(rec)
                    self._update_stats(rec)

            # Profile output tensors
            if isinstance(output, torch.Tensor):
                rec = TensorRecord(
                    name=f"{name}.output",
                    layer_index=layer_idx,
                    size_bytes=self._get_tensor_size(output),
                    step=step,
                    access_type='read',
                    timestamp=time.time()
                )
                self.records.append(rec)
                self._update_stats(rec)
            elif isinstance(output, tuple):
                for i, out in enumerate(output):
                    if isinstance(out, torch.Tensor):
                        rec = TensorRecord(
                            name=f"{name}.output_{i}",
                            layer_index=layer_idx,
                            size_bytes=self._get_tensor_size(out),
                            step=step,
                            access_type='read',
                            timestamp=time.time()
                        )
                        self.records.append(rec)
                        self._update_stats(rec)

            # Profile parameters
            for pname, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    rec = TensorRecord(
                        name=f"{name}.{pname}",
                        layer_index=layer_idx,
                        size_bytes=self._get_tensor_size(param),
                        step=step,
                        access_type='read',
                        timestamp=time.time()
                    )
                    self.records.append(rec)
                    self._update_stats(rec)
        return hook

    def _backward_hook(self, name: str, layer_idx: int):
        """Create a backward hook for a module."""
        def hook(module, grad_input, grad_output):
            step = self.current_step
            for i, grad in enumerate(grad_output):
                if isinstance(grad, torch.Tensor):
                    rec = TensorRecord(
                        name=f"{name}.grad_output_{i}",
                        layer_index=layer_idx,
                        size_bytes=self._get_tensor_size(grad),
                        step=step,
                        access_type='write',
                        timestamp=time.time()
                    )
                    self.records.append(rec)
                    self._update_stats(rec)
        return hook

    def _update_stats(self, record: TensorRecord):
        """Update aggregated statistics for a tensor."""
        key = record.name
        if key not in self.tensor_stats:
            self.tensor_stats[key] = TensorStats(
                name=record.name,
                layer_index=record.layer_index,
                total_size_bytes=record.size_bytes,
                first_access_step=record.step,
            )
        stats = self.tensor_stats[key]
        stats.access_count += 1
        if record.access_type == 'read':
            stats.read_count += 1
        else:
            stats.write_count += 1
        stats.last_access_step = max(stats.last_access_step, record.step)
        stats.access_steps.append(record.step)

    def register_hooks(self):
        """Register forward and backward hooks on all model modules."""
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                layer_idx = self.layer_index_map.get(name, 0)
                h1 = module.register_forward_hook(self._forward_hook(name, layer_idx))
                h2 = module.register_full_backward_hook(self._backward_hook(name, layer_idx))
                self.hooks.extend([h1, h2])
        print(f"[Profiler] Registered {len(self.hooks)} hooks on {len(self.layer_index_map)} modules")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        print(f"[Profiler] Removed all hooks")

    def step(self):
        """Increment the training step counter."""
        self.current_step += 1

    def get_memory_usage(self):
        """Get current memory usage snapshot."""
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        return {
            'rss_mb': mem_info.rss / 1024 / 1024,
            'vms_mb': mem_info.vms / 1024 / 1024,
            'step': self.current_step,
        }

    def save_records(self, filename: str = "memory_trace.json"):
        """Save raw access records to JSON."""
        import json
        filepath = os.path.join(self.output_dir, filename)
        data = [
            {
                'name': r.name,
                'layer_index': r.layer_index,
                'size_bytes': r.size_bytes,
                'step': r.step,
                'access_type': r.access_type,
                'timestamp': r.timestamp,
            }
            for r in self.records
        ]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[Profiler] Saved {len(data)} records to {filepath}")

    def save_stats(self, filename: str = "memory_dataset.csv"):
        """Save aggregated tensor statistics to CSV — training data for predictor."""
        filepath = os.path.join(self.output_dir, filename)
        fieldnames = [
            'name', 'layer_index', 'total_size_bytes', 'access_count',
            'read_count', 'write_count', 'last_access_step', 'first_access_step',
            'access_frequency', 'recency_score', 'reuse_distance'
        ]
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for key, stats in sorted(self.tensor_stats.items()):
                writer.writerow({
                    'name': stats.name,
                    'layer_index': stats.layer_index,
                    'total_size_bytes': stats.total_size_bytes,
                    'access_count': stats.access_count,
                    'read_count': stats.read_count,
                    'write_count': stats.write_count,
                    'last_access_step': stats.last_access_step,
                    'first_access_step': stats.first_access_step,
                    'access_frequency': f"{stats.access_frequency:.4f}",
                    'recency_score': f"{stats.recency_score:.4f}",
                    'reuse_distance': f"{stats.reuse_distance:.2f}",
                })
        print(f"[Profiler] Saved {len(self.tensor_stats)} tensor stats to {filepath}")
        return filepath

    def get_summary(self) -> dict:
        """Return a summary of the profiling session."""
        total_reads = sum(s.read_count for s in self.tensor_stats.values())
        total_writes = sum(s.write_count for s in self.tensor_stats.values())
        total_bytes = sum(s.total_size_bytes for s in self.tensor_stats.values())
        return {
            'total_steps': self.current_step,
            'total_records': len(self.records),
            'unique_tensors': len(self.tensor_stats),
            'total_reads': total_reads,
            'total_writes': total_writes,
            'total_memory_bytes': total_bytes,
            'total_memory_mb': total_bytes / 1024 / 1024,
            'avg_access_frequency': sum(s.access_frequency for s in self.tensor_stats.values()) / max(len(self.tensor_stats), 1),
            'avg_reuse_distance': sum(s.reuse_distance for s in self.tensor_stats.values() if s.reuse_distance != float('inf')) / max(len(self.tensor_stats), 1),
        }
