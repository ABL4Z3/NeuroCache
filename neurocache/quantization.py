"""
Phase 8 — Quantization Bridge
INT8/INT4 quantization for offloaded tensors to reduce I/O bandwidth.
"""

import torch
import torch.nn as nn
import os
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


SAFE_TORCH_DTYPES = {
    'torch.float16': torch.float16,
    'torch.float32': torch.float32,
    'torch.float64': torch.float64,
    'torch.bfloat16': torch.bfloat16,
    'torch.int8': torch.int8,
    'torch.uint8': torch.uint8,
    'torch.int16': torch.int16,
    'torch.int32': torch.int32,
    'torch.int64': torch.int64,
    'torch.bool': torch.bool,
}


@dataclass
class QuantizationResult:
    """Result of a quantization operation."""
    original_size_bytes: int
    quantized_size_bytes: int
    compression_ratio: float
    quantize_time_ms: float
    dequantize_time_ms: float = 0.0
    method: str = ''


class QuantizationBridge:
    """
    Handles quantization of tensors before offloading to SSD.
    - Activations: INT8 dynamic quantization (4x reduction)
    - Frozen weights: INT4 via bitsandbytes (8x reduction)
    - Gradient buffers: FP16 (2x reduction)
    """

    def __init__(self, ssd_dir: str = "./ssd_cache"):
        self.ssd_dir = ssd_dir
        os.makedirs(ssd_dir, exist_ok=True)
        self.stats = {
            'tensors_quantized': 0,
            'total_original_bytes': 0,
            'total_quantized_bytes': 0,
            'total_io_bandwidth_saved_mb': 0.0,
        }
        self._quantized_tensors: Dict[str, dict] = {}

    def quantize_int8(self, tensor: torch.Tensor, name: str) -> Tuple[torch.Tensor, QuantizationResult]:
        """
        Apply INT8 dynamic quantization to a tensor.
        Reduces FP32 -> INT8 (4x reduction).
        """
        start = time.time()
        original_size = tensor.element_size() * tensor.nelement()

        # Dynamic quantization: scale + zero_point + int8 values
        if tensor.is_floating_point():
            # Compute scale and zero point
            t_min = tensor.min().item()
            t_max = tensor.max().item()

            scale = (t_max - t_min) / 255.0 if t_max != t_min else 1.0
            zero_point = int(round(-t_min / scale)) if scale != 0 else 128
            zero_point = max(0, min(255, zero_point))

            quantized = torch.round((tensor - t_min) / scale).to(torch.uint8)
            quantized_size = quantized.element_size() * quantized.nelement()

            # Store metadata for dequantization
            self._quantized_tensors[name] = {
                'scale': scale,
                'zero_point': zero_point,
                't_min': t_min,
                't_max': t_max,
                'original_dtype': str(tensor.dtype),
                'original_shape': list(tensor.shape),
                'method': 'int8',
            }
        else:
            quantized = tensor
            quantized_size = original_size
            scale = 1.0
            self._quantized_tensors[name] = {'method': 'int8_skip', 'original_dtype': str(tensor.dtype)}

        elapsed = (time.time() - start) * 1000
        result = QuantizationResult(
            original_size_bytes=original_size,
            quantized_size_bytes=quantized_size,
            compression_ratio=original_size / max(quantized_size, 1),
            quantize_time_ms=elapsed,
            method='int8_dynamic',
        )

        self._update_stats(result)
        return quantized, result

    def quantize_int4(self, tensor: torch.Tensor, name: str) -> Tuple[torch.Tensor, QuantizationResult]:
        """
        Simulate INT4 quantization (8x reduction from FP32).
        Packs two 4-bit values per byte.
        """
        start = time.time()
        original_size = tensor.element_size() * tensor.nelement()

        if tensor.is_floating_point():
            t_min = tensor.min().item()
            t_max = tensor.max().item()

            scale = (t_max - t_min) / 15.0 if t_max != t_min else 1.0
            zero_point = int(round(-t_min / scale)) if scale != 0 else 8
            zero_point = max(0, min(15, zero_point))

            # Quantize to 4-bit (0-15)
            q4 = torch.round((tensor - t_min) / scale).clamp(0, 15).to(torch.uint8)

            # Pack two 4-bit values into one byte
            if q4.numel() % 2 != 0:
                q4 = torch.cat([q4, torch.zeros(1, dtype=torch.uint8)])
            q4_reshaped = q4.reshape(-1, 2)
            packed = (q4_reshaped[:, 0] << 4) | q4_reshaped[:, 1]
            packed_size = packed.element_size() * packed.nelement()

            self._quantized_tensors[name] = {
                'scale': scale,
                'zero_point': zero_point,
                't_min': t_min,
                't_max': t_max,
                'original_dtype': str(tensor.dtype),
                'original_shape': list(tensor.shape),
                'padded': q4.numel() != tensor.nelement(),
                'method': 'int4',
            }
        else:
            packed = tensor
            packed_size = original_size
            self._quantized_tensors[name] = {'method': 'int4_skip', 'original_dtype': str(tensor.dtype)}

        elapsed = (time.time() - start) * 1000
        result = QuantizationResult(
            original_size_bytes=original_size,
            quantized_size_bytes=packed_size,
            compression_ratio=original_size / max(packed_size, 1),
            quantize_time_ms=elapsed,
            method='int4_packed',
        )

        self._update_stats(result)
        return packed, result

    def quantize_fp16(self, tensor: torch.Tensor, name: str) -> Tuple[torch.Tensor, QuantizationResult]:
        """Convert FP32 tensor to FP16 (2x reduction)."""
        start = time.time()
        original_size = tensor.element_size() * tensor.nelement()

        if tensor.dtype == torch.float32:
            quantized = tensor.half()
        else:
            quantized = tensor

        quantized_size = quantized.element_size() * quantized.nelement()

        self._quantized_tensors[name] = {
            'original_dtype': str(tensor.dtype),
            'original_shape': list(tensor.shape),
            'method': 'fp16',
        }

        elapsed = (time.time() - start) * 1000
        result = QuantizationResult(
            original_size_bytes=original_size,
            quantized_size_bytes=quantized_size,
            compression_ratio=original_size / max(quantized_size, 1),
            quantize_time_ms=elapsed,
            method='fp16',
        )

        self._update_stats(result)
        return quantized, result

    def dequantize(self, quantized: torch.Tensor, name: str) -> torch.Tensor:
        """Dequantize a tensor back to its original format."""
        start = time.time()
        meta = self._quantized_tensors.get(name, {})

        method = meta.get('method', '')

        if method == 'int8':
            t_min = meta['t_min']
            scale = meta['scale']
            original_dtype = SAFE_TORCH_DTYPES.get(meta.get('original_dtype', ''), torch.float32)
            dequantized = quantized.float() * scale + t_min
            if original_dtype != torch.float32:
                dequantized = dequantized.to(original_dtype)

        elif method == 'int4':
            t_min = meta['t_min']
            scale = meta['scale']
            original_shape = meta.get('original_shape', None)
            # Unpack — quantized is flat 1D packed array
            flat = quantized.reshape(-1)
            low = (flat & 0x0F).float()
            high = ((flat >> 4) & 0x0F).float()
            # Interleave: for each byte, high nibble first, then low nibble
            unpacked = torch.stack([high, low], dim=1).reshape(-1)
            if meta.get('padded', False):
                target_numel = 1
                for s in original_shape:
                    target_numel *= s
                unpacked = unpacked[:target_numel]
            dequantized = unpacked * scale + t_min
            if original_shape:
                dequantized = dequantized.reshape(original_shape)

        elif method == 'fp16':
            dequantized = quantized.float()

        else:
            dequantized = quantized

        elapsed = (time.time() - start) * 1000
        if name in self._quantized_tensors:
            self._quantized_tensors[name]['dequantize_time_ms'] = elapsed

        return dequantized

    def _update_stats(self, result: QuantizationResult):
        """Update internal statistics."""
        self.stats['tensors_quantized'] += 1
        self.stats['total_original_bytes'] += result.original_size_bytes
        self.stats['total_quantized_bytes'] += result.quantized_size_bytes
        saved_mb = (result.original_size_bytes - result.quantized_size_bytes) / 1024 / 1024
        self.stats['total_io_bandwidth_saved_mb'] += max(0, saved_mb)

    def get_stats(self) -> dict:
        """Return quantization statistics."""
        total = max(self.stats['total_original_bytes'], 1)
        return {
            **self.stats,
            'overall_compression_ratio': total / max(self.stats['total_quantized_bytes'], 1),
            'bandwidth_reduction_pct': (1 - self.stats['total_quantized_bytes'] / total) * 100,
        }
