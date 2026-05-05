"""
Phase 6 — Async Prefetch Engine
Background thread pool for async I/O operations with priority queue.
Overlaps data movement with compute using concurrent execution.
"""

import torch
import threading
import queue
import time
import os
from typing import Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum


class PrefetchAction(Enum):
    OFFLOAD_CPU = 0
    OFFLOAD_SSD = 1
    PREFETCH_GPU = 2
    QUANTIZE = 3


@dataclass
class PrefetchTask:
    """A task for the prefetch engine."""
    action: PrefetchAction
    tensor_name: str
    priority: float  # higher = more urgent
    callback: Optional[Callable] = None
    created_at: float = field(default_factory=time.time)

    def __lt__(self, other):
        """Higher priority = processed first."""
        return self.priority > other.priority


class AsyncPrefetchEngine:
    """
    Background thread pool for async memory operations.
    Implements priority-based prefetch queue with configurable workers.
    """

    def __init__(
        self,
        num_workers: int = 2,
        scheduler=None,
        prefetch_ahead_layers: int = 3,
    ):
        self.num_workers = num_workers
        self.scheduler = scheduler
        self.prefetch_ahead_layers = prefetch_ahead_layers

        self.task_queue = queue.PriorityQueue()
        self.result_queue = queue.Queue()
        self.workers = []
        self.running = False
        self.stats = {
            'tasks_completed': 0,
            'tasks_pending': 0,
            'total_wait_time_ms': 0,
            'avg_wait_time_ms': 0,
            'offload_cpu_count': 0,
            'offload_ssd_count': 0,
            'prefetch_gpu_count': 0,
            'quantize_count': 0,
        }
        self._lock = threading.Lock()

    def start(self):
        """Start the worker threads."""
        self.running = True
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop, name=f"PrefetchWorker-{i}", daemon=True)
            worker.start()
            self.workers.append(worker)
        print(f"[PrefetchEngine] Started {self.num_workers} workers")

    def stop(self):
        """Stop the worker threads."""
        self.running = False
        # Poison pills — use a special PrefetchTask with lowest priority
        poison = PrefetchTask(action=PrefetchAction.OFFLOAD_CPU, tensor_name='__POISON__', priority=-999)
        for _ in self.workers:
            self.task_queue.put(poison)
        for worker in self.workers:
            worker.join(timeout=5)
        self.workers.clear()
        print(f"[PrefetchEngine] Stopped all workers")

    def submit(self, task: PrefetchTask):
        """Submit a task to the prefetch engine."""
        if not self.running:
            self.start()
        self.task_queue.put(task)
        with self._lock:
            self.stats['tasks_pending'] += 1

    def submit_offload_cpu(self, tensor_name: str, priority: float = 0.5):
        """Submit a CPU offload task."""
        self.submit(PrefetchTask(
            action=PrefetchAction.OFFLOAD_CPU,
            tensor_name=tensor_name,
            priority=priority,
        ))

    def submit_offload_ssd(self, tensor_name: str, priority: float = 0.3):
        """Submit an SSD offload task."""
        self.submit(PrefetchTask(
            action=PrefetchAction.OFFLOAD_SSD,
            tensor_name=tensor_name,
            priority=priority,
        ))

    def submit_prefetch_gpu(self, tensor_name: str, priority: float = 0.8):
        """Submit a GPU prefetch task (high priority)."""
        self.submit(PrefetchTask(
            action=PrefetchAction.PREFETCH_GPU,
            tensor_name=tensor_name,
            priority=priority,
        ))

    def submit_quantize(self, tensor_name: str, priority: float = 0.4):
        """Submit a quantization task."""
        self.submit(PrefetchTask(
            action=PrefetchAction.QUANTIZE,
            tensor_name=tensor_name,
            priority=priority,
        ))

    def prefetch_next_layers(self, current_layer: int, layer_names: list):
        """
        Prefetch tensors for layers ahead of current layer.
        Called before forward pass.
        """
        for i in range(1, self.prefetch_ahead_layers + 1):
            target_idx = current_layer + i
            if target_idx < len(layer_names):
                name = layer_names[target_idx]
                self.submit_prefetch_gpu(name, priority=1.0 - (i * 0.2))

    def _worker_loop(self):
        """Worker thread main loop."""
        while self.running:
            try:
                task = self.task_queue.get(timeout=1.0)
                if task is None or task.tensor_name == '__POISON__':
                    break

                start_time = time.time()
                result = self._execute_task(task)
                elapsed_ms = (time.time() - start_time) * 1000

                with self._lock:
                    self.stats['tasks_completed'] += 1
                    self.stats['tasks_pending'] = max(0, self.stats['tasks_pending'] - 1)
                    self.stats['total_wait_time_ms'] += elapsed_ms
                    self.stats['avg_wait_time_ms'] = self.stats['total_wait_time_ms'] / self.stats['tasks_completed']

                if task.callback:
                    task.callback(result)

                self.task_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[PrefetchWorker] Error: {e}")

    def _execute_task(self, task: PrefetchTask) -> dict:
        """Execute a single prefetch task."""
        result = {
            'action': task.action.name,
            'tensor_name': task.tensor_name,
            'success': False,
            'time_ms': 0,
        }

        start = time.time()

        try:
            if self.scheduler is None:
                result['error'] = 'No scheduler attached'
                return result

            if task.tensor_name not in self.scheduler.registry:
                result['error'] = f'Tensor {task.tensor_name} not found in registry'
                return result

            entry = self.scheduler.registry[task.tensor_name]

            if task.action == PrefetchAction.OFFLOAD_CPU:
                if entry.tier.value == 0:  # GPU
                    self.scheduler._move_tensor(entry, __import__('neurocache.scheduler', fromlist=['MemoryTier']).MemoryTier.CPU_PINNED)
                    with self._lock:
                        self.stats['offload_cpu_count'] += 1
                result['success'] = True

            elif task.action == PrefetchAction.OFFLOAD_SSD:
                if entry.tier.value <= 1:  # GPU or CPU
                    self.scheduler._move_tensor(entry, __import__('neurocache.scheduler', fromlist=['MemoryTier']).MemoryTier.SSD)
                    with self._lock:
                        self.stats['offload_ssd_count'] += 1
                result['success'] = True

            elif task.action == PrefetchAction.PREFETCH_GPU:
                if entry.tier.value > 0:  # Not on GPU
                    self.scheduler._move_tensor(entry, __import__('neurocache.scheduler', fromlist=['MemoryTier']).MemoryTier.GPU)
                    with self._lock:
                        self.stats['prefetch_gpu_count'] += 1
                result['success'] = True

            elif task.action == PrefetchAction.QUANTIZE:
                # Quantization handled by QuantizationBridge
                with self._lock:
                    self.stats['quantize_count'] += 1
                result['success'] = True

        except Exception as e:
            result['error'] = str(e)

        result['time_ms'] = (time.time() - start) * 1000
        return result

    def wait_all(self, timeout: float = 30.0):
        """Wait for all pending tasks to complete."""
        self.task_queue.join()

    def get_stats(self) -> dict:
        """Return engine statistics."""
        with self._lock:
            return dict(self.stats)

    def get_prefetch_wait_time(self) -> float:
        """Average prefetch wait time in ms."""
        with self._lock:
            return self.stats['avg_wait_time_ms']
