"""
Runtime activation offload for real CUDA training.

This module uses PyTorch saved-tensor hooks to move autograd-saved tensors out
of GPU memory during the forward pass and restore them during backward. It is a
real measurement target for low-VRAM experiments, separate from the older
simulation-oriented scheduler.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class ActivationOffloadConfig:
    """Configuration for saved activation offload."""

    mode: str = "cpu_fp16"
    min_tensor_bytes: int = 256 * 1024
    pin_memory: bool = True
    enabled: bool = True
    async_transfer: bool = True
    reuse_host_buffers: bool = True
    policy: str = "all"
    predictor_path: Optional[str] = None
    predictor_confidence: float = 0.45
    max_pending_d2h: int = 2
    max_offloads_per_context: int = 0


@dataclass
class ActivationOffloadStats:
    """Aggregated runtime statistics."""

    tensors_packed: int = 0
    tensors_unpacked: int = 0
    original_bytes: int = 0
    stored_bytes: int = 0
    pack_time_ms: float = 0.0
    unpack_time_ms: float = 0.0
    d2h_transfers: int = 0
    h2d_transfers: int = 0
    d2h_bytes: int = 0
    h2d_bytes: int = 0
    host_buffer_allocations: int = 0
    host_buffer_reuses: int = 0
    policy_keep_count: int = 0
    policy_offload_count: int = 0
    predictor_calls: int = 0
    predictor_keep_count: int = 0
    predictor_offload_count: int = 0
    budget_keep_count: int = 0
    d2h_throttle_waits: int = 0
    d2h_event_pairs: list = field(default_factory=list)
    h2d_event_pairs: list = field(default_factory=list)

    @property
    def compression_ratio(self) -> float:
        if self.stored_bytes <= 0:
            return 1.0
        return self.original_bytes / self.stored_bytes

    @property
    def saved_bytes(self) -> int:
        return max(0, self.original_bytes - self.stored_bytes)

    def to_dict(self) -> dict:
        d2h_stream_ms = _elapsed_event_ms(self.d2h_event_pairs)
        h2d_stream_ms = _elapsed_event_ms(self.h2d_event_pairs)
        transfer_stream_ms = d2h_stream_ms + h2d_stream_ms
        transfer_mb = (self.d2h_bytes + self.h2d_bytes) / 1024 / 1024
        return {
            "tensors_packed": self.tensors_packed,
            "tensors_unpacked": self.tensors_unpacked,
            "original_mb": self.original_bytes / 1024 / 1024,
            "stored_mb": self.stored_bytes / 1024 / 1024,
            "estimated_saved_mb": self.saved_bytes / 1024 / 1024,
            "compression_ratio": self.compression_ratio,
            "pack_time_ms": self.pack_time_ms,
            "unpack_time_ms": self.unpack_time_ms,
            "d2h_transfers": self.d2h_transfers,
            "h2d_transfers": self.h2d_transfers,
            "d2h_mb": self.d2h_bytes / 1024 / 1024,
            "h2d_mb": self.h2d_bytes / 1024 / 1024,
            "host_buffer_allocations": self.host_buffer_allocations,
            "host_buffer_reuses": self.host_buffer_reuses,
            "policy_keep_count": self.policy_keep_count,
            "policy_offload_count": self.policy_offload_count,
            "predictor_calls": self.predictor_calls,
            "predictor_keep_count": self.predictor_keep_count,
            "predictor_offload_count": self.predictor_offload_count,
            "budget_keep_count": self.budget_keep_count,
            "d2h_throttle_waits": self.d2h_throttle_waits,
            "d2h_stream_time_ms": d2h_stream_ms,
            "h2d_stream_time_ms": h2d_stream_ms,
            "transfer_stream_time_ms": transfer_stream_ms,
            "effective_transfer_bandwidth_gbps": (transfer_mb / 1024) / (transfer_stream_ms / 1000)
            if transfer_stream_ms > 0
            else 0.0,
        }


@dataclass
class _CachedActivation:
    payload: torch.Tensor
    device: torch.device
    dtype: torch.dtype
    shape: torch.Size
    mode: str
    owner: Optional["ActivationOffloadContext"] = None
    ready_event: Optional[torch.cuda.Event] = None
    pool_key: Optional[tuple] = None
    pooled: bool = False
    scale: Optional[float] = None
    offset: Optional[float] = None


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.element_size() * tensor.nelement()


def _elapsed_event_ms(event_pairs: list) -> float:
    total = 0.0
    remaining = []
    for start_event, end_event in event_pairs:
        try:
            if end_event.query():
                total += start_event.elapsed_time(end_event)
            else:
                remaining.append((start_event, end_event))
        except RuntimeError:
            remaining.append((start_event, end_event))
    event_pairs[:] = remaining
    return total


class ActivationOffloadContext:
    """
    Saved-tensor hook context for activation offload.

    Supported modes:
    - cpu: move saved tensors to CPU without dtype conversion.
    - cpu_fp16: move floating saved tensors to CPU float16.
    - cpu_int8: move floating saved tensors to CPU uint8 with per-tensor scale.
    """

    def __init__(self, config: Optional[ActivationOffloadConfig] = None):
        self.config = config or ActivationOffloadConfig()
        self.stats = ActivationOffloadStats()
        self._hook_cm: Any = None
        self._transfer_stream: Optional[torch.cuda.Stream] = None
        self._restore_stream: Optional[torch.cuda.Stream] = None
        self._host_pool: dict[tuple, list[torch.Tensor]] = defaultdict(list)
        self._access_step = 0
        self._signature_stats: dict[tuple, dict] = {}
        self._predictor = None
        self._predictor_norm_mean = None
        self._predictor_norm_std = None
        self._predictor_window_size = 10
        self._pending_d2h: list[torch.cuda.Event] = []
        self._context_offload_count = 0

    def __enter__(self) -> "ActivationOffloadContext":
        if not self.config.enabled:
            return self
        self._context_offload_count = 0
        if torch.cuda.is_available() and self.config.async_transfer:
            self._transfer_stream = torch.cuda.Stream()
            self._restore_stream = torch.cuda.Stream()
        self._load_predictor_if_needed()
        self._hook_cm = torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)
        self._hook_cm.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._hook_cm is not None:
            self._hook_cm.__exit__(exc_type, exc_val, exc_tb)
        return False

    def _pool_key(self, shape: torch.Size, dtype: torch.dtype) -> tuple:
        return tuple(shape), dtype

    def _load_predictor_if_needed(self):
        if self.config.policy != "predictor" or self._predictor is not None or not self.config.predictor_path:
            return
        try:
            from .predictor import MemoryPredictor

            checkpoint = torch.load(self.config.predictor_path, map_location="cpu", weights_only=False)
            model = MemoryPredictor(
                input_size=checkpoint.get("input_size", 7),
                hidden_size=checkpoint.get("hidden_size", 64),
                num_layers=checkpoint.get("num_layers", 2),
                num_classes=4,
            )
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            self._predictor = model
            self._predictor_norm_mean = checkpoint.get("norm_mean")
            self._predictor_norm_std = checkpoint.get("norm_std")
            self._predictor_window_size = checkpoint.get("window_size", 10)
        except Exception as exc:
            print(f"[ActivationOffload] Predictor load failed: {exc}. Falling back to heuristic policy.")
            self.config.policy = "heuristic"

    def _get_host_buffer(self, shape: torch.Size, dtype: torch.dtype) -> tuple[torch.Tensor, tuple, bool]:
        key = self._pool_key(shape, dtype)
        if self.config.reuse_host_buffers and self._host_pool[key]:
            self.stats.host_buffer_reuses += 1
            return self._host_pool[key].pop(), key, True

        try:
            payload = torch.empty(tuple(shape), dtype=dtype, device="cpu", pin_memory=self.config.pin_memory)
        except RuntimeError:
            payload = torch.empty(tuple(shape), dtype=dtype, device="cpu")
        self.stats.host_buffer_allocations += 1
        return payload, key, self.config.reuse_host_buffers

    def _release_host_buffer(self, key: Optional[tuple], payload: torch.Tensor, pooled: bool):
        if key is not None and pooled and self.config.reuse_host_buffers:
            self._host_pool[key].append(payload)

    def _signature(self, tensor: torch.Tensor) -> tuple:
        return tuple(tensor.shape), str(tensor.dtype)

    def _features_for_tensor(self, tensor: torch.Tensor, original_bytes: int) -> tuple[torch.Tensor, dict]:
        self._access_step += 1
        sig = self._signature(tensor)
        stats = self._signature_stats.setdefault(
            sig,
            {
                "access_count": 0,
                "first_step": self._access_step,
                "last_step": self._access_step,
                "prev_step": None,
                "reuse_distance": float(self._predictor_window_size),
            },
        )
        prev_step = stats["last_step"]
        stats["access_count"] += 1
        stats["prev_step"] = prev_step
        stats["last_step"] = self._access_step
        stats["reuse_distance"] = max(1, self._access_step - prev_step)
        span = max(1, self._access_step - stats["first_step"] + 1)
        access_frequency = stats["access_count"] / span * 100.0
        recency_score = 1.0 / (1.0 + stats["reuse_distance"])
        features = torch.tensor(
            [
                0.0,
                original_bytes,
                stats["access_count"],
                self._access_step,
                access_frequency,
                recency_score,
                stats["reuse_distance"],
            ],
            dtype=torch.float32,
        )
        return features, stats

    def _record_offload_request(self) -> bool:
        if (
            self.config.max_offloads_per_context > 0
            and self._context_offload_count >= self.config.max_offloads_per_context
        ):
            self.stats.budget_keep_count += 1
            self.stats.policy_keep_count += 1
            return False
        self._context_offload_count += 1
        self.stats.policy_offload_count += 1
        return True

    def _should_offload(self, tensor: torch.Tensor, original_bytes: int) -> bool:
        features, stats = self._features_for_tensor(tensor, original_bytes)
        policy = self.config.policy
        if policy == "all":
            return self._record_offload_request()

        if policy == "heuristic":
            keep = stats["access_count"] > 2 and stats["reuse_distance"] <= 2
            if keep:
                self.stats.policy_keep_count += 1
                return False
            return self._record_offload_request()

        if policy == "predictor" and self._predictor is not None:
            self.stats.predictor_calls += 1
            window = features.repeat(self._predictor_window_size, 1).unsqueeze(0)
            if self._predictor_norm_mean is not None and self._predictor_norm_std is not None:
                window = (window - self._predictor_norm_mean) / (self._predictor_norm_std + 1e-8)
            with torch.no_grad():
                logits = self._predictor(window)
                probs = torch.softmax(logits, dim=-1)
                action_id = int(torch.argmax(probs, dim=-1).item())
                confidence = float(probs[0, action_id].item())
            # KEEP and PREFETCH stay resident; CPU/SSD labels are offload decisions.
            keep = action_id in (0, 3) and confidence >= self.config.predictor_confidence
            if keep:
                self.stats.predictor_keep_count += 1
                self.stats.policy_keep_count += 1
                return False
            self.stats.predictor_offload_count += 1
            return self._record_offload_request()

        return self._record_offload_request()

    def _pack(self, tensor: torch.Tensor):
        if (
            not self.config.enabled
            or not tensor.is_cuda
            or tensor.numel() == 0
            or _tensor_nbytes(tensor) < self.config.min_tensor_bytes
        ):
            return tensor

        start = time.perf_counter()
        original_bytes = _tensor_nbytes(tensor)
        if not self._should_offload(tensor, original_bytes):
            return tensor
        mode = self.config.mode

        with torch.no_grad():
            source = tensor.detach()
            if mode == "cpu":
                target_dtype = source.dtype
                scale = None
                offset = None
            elif mode == "cpu_fp16" and tensor.is_floating_point():
                target_dtype = torch.float16
                scale = None
                offset = None
            elif mode == "cpu_int8" and tensor.is_floating_point():
                t_min = source.amin()
                t_max = source.amax()
                span = (t_max - t_min).clamp_min(torch.finfo(torch.float32).eps)
                source = torch.round((source - t_min) * (255.0 / span)).clamp_(0, 255).to(torch.uint8)
                target_dtype = torch.uint8
                scale = (span / 255.0).item()
                offset = t_min.item()
            else:
                target_dtype = source.dtype
                scale = None
                offset = None

            payload, pool_key, pooled = self._get_host_buffer(source.shape, target_dtype)
            ready_event = None
            if self.config.async_transfer and self._transfer_stream is not None:
                self._throttle_d2h_if_needed()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(self._transfer_stream):
                    start_event.record(self._transfer_stream)
                    payload.copy_(source, non_blocking=True)
                    end_event.record(self._transfer_stream)
                    ready_event = end_event
                self.stats.d2h_event_pairs.append((start_event, end_event))
                self._pending_d2h.append(end_event)
            else:
                payload.copy_(source, non_blocking=False)

        stored_bytes = _tensor_nbytes(payload)
        self.stats.tensors_packed += 1
        self.stats.original_bytes += original_bytes
        self.stats.stored_bytes += stored_bytes
        self.stats.d2h_transfers += 1
        self.stats.d2h_bytes += stored_bytes
        self.stats.pack_time_ms += (time.perf_counter() - start) * 1000

        return _CachedActivation(
            payload=payload,
            device=tensor.device,
            dtype=tensor.dtype,
            shape=tensor.shape,
            mode=mode,
            owner=self,
            ready_event=ready_event,
            pool_key=pool_key,
            pooled=pooled,
            scale=scale,
            offset=offset,
        )

    def _unpack(self, packed):
        if not isinstance(packed, _CachedActivation):
            return packed

        start = time.perf_counter()
        if packed.ready_event is not None:
            torch.cuda.current_stream(packed.device).wait_event(packed.ready_event)

        if packed.mode == "cpu_int8" and packed.scale is not None and packed.offset is not None:
            tensor = self._copy_to_device(packed.payload, packed.device, torch.uint8).float()
            tensor = (tensor * packed.scale + packed.offset).to(packed.dtype)
        else:
            tensor = self._copy_to_device(packed.payload, packed.device, packed.dtype)

        tensor = tensor.reshape(packed.shape)
        self.stats.tensors_unpacked += 1
        self.stats.h2d_transfers += 1
        self.stats.h2d_bytes += _tensor_nbytes(packed.payload)
        self.stats.unpack_time_ms += (time.perf_counter() - start) * 1000
        self._release_host_buffer(packed.pool_key, packed.payload, packed.pooled)
        return tensor

    def _copy_to_device(self, payload: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.config.async_transfer and self._restore_stream is not None:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(self._restore_stream):
                start_event.record(self._restore_stream)
                tensor = payload.to(device, dtype=dtype, non_blocking=True)
                end_event.record(self._restore_stream)
            torch.cuda.current_stream(device).wait_event(end_event)
            self.stats.h2d_event_pairs.append((start_event, end_event))
            return tensor
        return payload.to(device, dtype=dtype, non_blocking=True)

    def _throttle_d2h_if_needed(self):
        self._pending_d2h = [event for event in self._pending_d2h if not event.query()]
        if self.config.max_pending_d2h <= 0:
            return
        while len(self._pending_d2h) >= self.config.max_pending_d2h:
            event = self._pending_d2h.pop(0)
            event.synchronize()
            self.stats.d2h_throttle_waits += 1
            self._pending_d2h = [item for item in self._pending_d2h if not item.query()]
