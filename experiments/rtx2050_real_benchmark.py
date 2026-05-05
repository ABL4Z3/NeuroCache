#!/usr/bin/env python3
"""
Real CUDA benchmark for NeuroCache on low-VRAM GPUs.

The benchmark intentionally avoids downloaded models and datasets. It trains a
small GPT-like decoder on deterministic random token batches and records actual
CUDA peak memory, wall-clock throughput, CPU RAM, and OOM status.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neurocache.activation_cache import ActivationOffloadConfig, ActivationOffloadContext


@dataclass
class GPTBenchConfig:
    vocab_size: int = 16000
    seq_len: int = 384
    batch_size: int = 3
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.0


@dataclass
class BenchVariant:
    name: str
    checkpoint_blocks: bool = False
    offload_mode: Optional[str] = None
    min_tensor_kb: int = 256
    offload_policy: str = "all"
    predictor_path: Optional[str] = None
    async_transfer: bool = True
    max_pending_d2h: int = 2
    max_offloads_per_step: int = 0
    cpu_optimizer: bool = False
    hybrid_optimizer: bool = False
    fp16_optimizer: bool = False
    bf16_optimizer: bool = False


class UtilizationSampler:
    def __init__(self, interval_sec: float = 0.25):
        self.interval_sec = interval_sec
        self.cpu_samples = []
        self.gpu_samples = []
        self.gpu_memory_samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        psutil.cpu_percent(interval=None)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            self.cpu_samples.append(psutil.cpu_percent(interval=None))
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                ).strip()
                util, mem_used = [float(part.strip()) for part in out.split(",")[:2]]
                self.gpu_samples.append(util)
                self.gpu_memory_samples.append(mem_used)
            except Exception:
                pass
            self._stop.wait(self.interval_sec)

    def summary(self) -> dict:
        return {
            "avg_cpu_util_pct": float(np.mean(self.cpu_samples)) if self.cpu_samples else None,
            "max_cpu_util_pct": float(np.max(self.cpu_samples)) if self.cpu_samples else None,
            "avg_gpu_util_pct": float(np.mean(self.gpu_samples)) if self.gpu_samples else None,
            "max_gpu_util_pct": float(np.max(self.gpu_samples)) if self.gpu_samples else None,
            "avg_nvidia_smi_memory_mb": float(np.mean(self.gpu_memory_samples)) if self.gpu_memory_samples else None,
            "max_nvidia_smi_memory_mb": float(np.max(self.gpu_memory_samples)) if self.gpu_memory_samples else None,
            "util_samples": len(self.cpu_samples),
        }


class DecoderBlock(nn.Module):
    def __init__(self, cfg: GPTBenchConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model,
            cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        hidden = cfg.d_model * cfg.mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, cfg.d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTBenchConfig, checkpoint_blocks: bool = False):
        super().__init__()
        self.cfg = cfg
        self.checkpoint_blocks = checkpoint_blocks
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        self.blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = idx.shape
        x = self.token_emb(idx) + self.pos_emb[:, :seq_len, :]
        attn_mask = torch.full((seq_len, seq_len), float("-inf"), device=idx.device)
        attn_mask = torch.triu(attn_mask, diagonal=1)

        for block in self.blocks:
            if self.checkpoint_blocks:
                x = checkpoint(block, x, attn_mask, use_reentrant=False)
            else:
                x = block(x, attn_mask)

        return self.head(self.ln_f(x))


class CPUAdamW:
    """Simple measured CPU-state AdamW optimizer for low-VRAM experiments."""

    def __init__(self, params, lr: float = 3e-4, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_num = 0
        self.state = {}
        for p in self.params:
            self.state[p] = {
                "exp_avg": torch.zeros_like(p.detach(), device="cpu"),
                "exp_avg_sq": torch.zeros_like(p.detach(), device="cpu"),
            }

    def zero_grad(self, set_to_none: bool = True):
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def step(self):
        self.step_num += 1
        beta1, beta2 = self.beta1, self.beta2
        bias_correction1 = 1 - beta1**self.step_num
        bias_correction2 = 1 - beta2**self.step_num
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad_cpu = p.grad.detach().to("cpu", non_blocking=False)
                param_cpu = p.detach().to("cpu", non_blocking=False)
                if self.weight_decay:
                    grad_cpu = grad_cpu.add(param_cpu, alpha=self.weight_decay)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad_cpu, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_cpu, grad_cpu, value=1 - beta2)
                denom = exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(self.eps)
                step_size = self.lr / bias_correction1
                param_cpu.addcdiv_(exp_avg, denom, value=-step_size)
                p.copy_(param_cpu.to(p.device, non_blocking=False))


class HybridAdamW:
    """AdamW with exp_avg on GPU and exp_avg_sq on CPU."""

    def __init__(self, params, lr: float = 3e-4, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_num = 0
        self.state = {}
        for p in self.params:
            self.state[p] = {
                "exp_avg": torch.zeros_like(p.detach(), device=p.device),
                "exp_avg_sq": torch.zeros_like(p.detach(), device="cpu"),
            }

    def zero_grad(self, set_to_none: bool = True):
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def step(self):
        self.step_num += 1
        beta1, beta2 = self.beta1, self.beta2
        bias_correction1 = 1 - beta1**self.step_num
        bias_correction2 = 1 - beta2**self.step_num
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                grad = p.grad.detach()
                if self.weight_decay:
                    grad_for_avg = grad.add(p.detach(), alpha=self.weight_decay)
                else:
                    grad_for_avg = grad
                state = self.state[p]
                exp_avg = state["exp_avg"]
                exp_avg.mul_(beta1).add_(grad_for_avg, alpha=1 - beta1)

                grad_cpu = grad_for_avg.to("cpu", non_blocking=False)
                exp_avg_sq_cpu = state["exp_avg_sq"]
                exp_avg_sq_cpu.mul_(beta2).addcmul_(grad_cpu, grad_cpu, value=1 - beta2)
                exp_avg_sq_gpu = exp_avg_sq_cpu.to(p.device, non_blocking=False)
                denom_gpu = exp_avg_sq_gpu.sqrt().div_(bias_correction2**0.5).add_(self.eps)
                step_size = self.lr / bias_correction1
                p.addcdiv_(exp_avg, denom_gpu, value=-step_size)


class FP16StateAdamW:
    """AdamW with optimizer moments stored as low-precision CUDA tensors."""

    def __init__(
        self,
        params,
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        state_dtype=torch.float16,
    ):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state_dtype = state_dtype
        self.step_num = 0
        self.state = {
            p: {
                "exp_avg": torch.zeros_like(p, dtype=state_dtype),
                "exp_avg_sq": torch.zeros_like(p, dtype=state_dtype),
            }
            for p in self.params
        }

    def zero_grad(self, set_to_none: bool = True):
        for p in self.params:
            if p.grad is None:
                continue
            if set_to_none:
                p.grad = None
            else:
                p.grad.zero_()

    def step(self):
        self.step_num += 1
        beta1, beta2 = self.beta1, self.beta2
        bias_correction1 = 1 - beta1**self.step_num
        bias_correction2 = 1 - beta2**self.step_num
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                grad = p.grad.detach()
                if self.weight_decay:
                    grad_for_avg = grad.add(p.detach(), alpha=self.weight_decay)
                else:
                    grad_for_avg = grad
                state = self.state[p]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                grad_state = grad_for_avg.to(self.state_dtype)
                exp_avg.mul_(beta1).add_(grad_state, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_state, grad_state, value=1 - beta2)
                denom = exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(self.eps)
                step_size = self.lr / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)


class BF16StateAdamW(FP16StateAdamW):
    """AdamW with BF16 CUDA optimizer moments."""

    def __init__(self, params, **kwargs):
        super().__init__(params, state_dtype=torch.bfloat16, **kwargs)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_nvidia_smi() -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        name, memory_total, driver = [part.strip() for part in out.split(",")[:3]]
        return {"name": name, "memory_total_mb": float(memory_total), "driver_version": driver}
    except Exception:
        return {}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def make_batch(cfg: GPTBenchConfig, device: torch.device) -> torch.Tensor:
    return torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device=device)


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, batch: torch.Tensor) -> float:
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch)
    loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1))
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text and ("cuda" in text or "accelerator" in text)


def run_variant(
    variant: BenchVariant,
    cfg: GPTBenchConfig,
    device: torch.device,
    warmup_steps: int,
    measure_steps: int,
    seed: int,
) -> dict:
    set_seed(seed)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()

    model = TinyGPT(cfg, checkpoint_blocks=variant.checkpoint_blocks).to(device)
    if variant.cpu_optimizer:
        optimizer = CPUAdamW(model.parameters(), lr=3e-4)
    elif variant.hybrid_optimizer:
        optimizer = HybridAdamW(model.parameters(), lr=3e-4)
    elif variant.fp16_optimizer:
        optimizer = FP16StateAdamW(model.parameters(), lr=3e-4)
    elif variant.bf16_optimizer:
        optimizer = BF16StateAdamW(model.parameters(), lr=3e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    params = count_parameters(model)
    tokens_per_step = cfg.batch_size * cfg.seq_len
    ram_peak_mb = process.memory_info().rss / 1024 / 1024
    losses = []
    offload_stats = None
    oom_error = None
    started = None
    elapsed = None
    sampler = None
    utilization = {}

    try:
        context = (
            ActivationOffloadContext(
                ActivationOffloadConfig(
                    mode=variant.offload_mode,
                    min_tensor_bytes=variant.min_tensor_kb * 1024,
                    policy=variant.offload_policy,
                    predictor_path=variant.predictor_path,
                    async_transfer=variant.async_transfer,
                    max_pending_d2h=variant.max_pending_d2h,
                    max_offloads_per_context=variant.max_offloads_per_step,
                )
            )
            if variant.offload_mode
            else None
        )

        sampler = UtilizationSampler()
        sampler.start()
        for step in range(warmup_steps + measure_steps):
            batch = make_batch(cfg, device)
            torch.cuda.synchronize()
            if step == warmup_steps:
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()

            if context is None:
                loss = train_step(model, optimizer, batch)
            else:
                with context:
                    loss = train_step(model, optimizer, batch)

            losses.append(loss)
            torch.cuda.synchronize()
            ram_peak_mb = max(ram_peak_mb, process.memory_info().rss / 1024 / 1024)

        elapsed = time.perf_counter() - started if started is not None else math.nan
        if context is not None:
            offload_stats = context.stats.to_dict()
    except Exception as exc:
        if not is_cuda_oom(exc):
            raise
        oom_error = str(exc).splitlines()[0]
    finally:
        try:
            sampler.stop()
            utilization = sampler.summary()
        except Exception:
            utilization = {}
        peak_allocated_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        peak_reserved_mb = torch.cuda.max_memory_reserved() / 1024 / 1024
        del optimizer, model
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    measured_steps = 0 if elapsed in (None, 0) or oom_error else measure_steps
    throughput = (
        (measured_steps * tokens_per_step) / elapsed
        if measured_steps and elapsed and elapsed > 0
        else 0.0
    )

    return {
        "variant": variant.name,
        "checkpoint_blocks": variant.checkpoint_blocks,
        "cpu_optimizer": variant.cpu_optimizer,
        "hybrid_optimizer": variant.hybrid_optimizer,
        "offload_mode": variant.offload_mode,
        "oom": oom_error is not None,
        "oom_error": oom_error,
        "params": params,
        "model_param_m": params / 1e6,
        "batch_size": cfg.batch_size,
        "seq_len": cfg.seq_len,
        "tokens_per_step": tokens_per_step,
        "warmup_steps": warmup_steps,
        "measure_steps": measured_steps,
        "elapsed_sec": elapsed,
        "tokens_per_sec": throughput,
        "ms_per_step": (elapsed / measured_steps * 1000) if measured_steps else None,
        "peak_cuda_allocated_mb": peak_allocated_mb,
        "peak_cuda_reserved_mb": peak_reserved_mb,
        "peak_process_rss_mb": ram_peak_mb,
        "final_loss": losses[-1] if losses else None,
        "offload_stats": offload_stats,
        **utilization,
    }


def summarize(results: list[dict]) -> list[dict]:
    baseline = next((r for r in results if r["variant"] == "baseline" and not r["oom"]), None)
    checkpoint = next((r for r in results if r["variant"] == "checkpoint" and not r["oom"]), None)
    rows = []
    for row in results:
        out = dict(row)
        if baseline and not row["oom"]:
            out["vram_reduction_vs_baseline_pct"] = (
                1.0 - row["peak_cuda_allocated_mb"] / baseline["peak_cuda_allocated_mb"]
            ) * 100.0
            out["throughput_vs_baseline_pct"] = (
                row["tokens_per_sec"] / baseline["tokens_per_sec"] - 1.0
            ) * 100.0
        else:
            out["vram_reduction_vs_baseline_pct"] = None
            out["throughput_vs_baseline_pct"] = None
        if checkpoint and not row["oom"]:
            out["vram_reduction_vs_checkpoint_pct"] = (
                1.0 - row["peak_cuda_allocated_mb"] / checkpoint["peak_cuda_allocated_mb"]
            ) * 100.0
            out["throughput_vs_checkpoint_pct"] = (
                row["tokens_per_sec"] / checkpoint["tokens_per_sec"] - 1.0
            ) * 100.0
        else:
            out["vram_reduction_vs_checkpoint_pct"] = None
            out["throughput_vs_checkpoint_pct"] = None
        rows.append(out)
    return rows


def write_outputs(results: list[dict], metadata: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    summarized = summarize(results)
    payload = {"metadata": metadata, "results": summarized}
    with (output_dir / "rtx2050_real_benchmark.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    csv_path = output_dir / "rtx2050_real_benchmark.csv"
    flat_rows = []
    for row in summarized:
        flat = {k: v for k, v in row.items() if k != "offload_stats"}
        if row.get("offload_stats"):
            for k, v in row["offload_stats"].items():
                flat[f"offload_{k}"] = v
        flat_rows.append(flat)

    fieldnames = sorted({key for row in flat_rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "metrics"))
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--seq-len", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--min-tensor-kb", type=int, default=256)
    parser.add_argument("--predictor-path", default=None)
    parser.add_argument("--max-pending-d2h", type=int, default=2)
    parser.add_argument("--variants", default="baseline,checkpoint,neurocache_cpu_fp16")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. This benchmark requires an NVIDIA CUDA GPU.")

    device = torch.device("cuda")
    cfg = GPTBenchConfig(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    )
    variant_map = {
        "baseline": BenchVariant("baseline"),
        "checkpoint": BenchVariant("checkpoint", checkpoint_blocks=True),
        "checkpoint_cpuadam": BenchVariant("checkpoint_cpuadam", checkpoint_blocks=True, cpu_optimizer=True),
        "checkpoint_hybridadam": BenchVariant("checkpoint_hybridadam", checkpoint_blocks=True, hybrid_optimizer=True),
        "checkpoint_fp16adam": BenchVariant("checkpoint_fp16adam", checkpoint_blocks=True, fp16_optimizer=True),
        "checkpoint_bf16adam": BenchVariant("checkpoint_bf16adam", checkpoint_blocks=True, bf16_optimizer=True),
        "neurocache_cpu": BenchVariant("neurocache_cpu", offload_mode="cpu", min_tensor_kb=args.min_tensor_kb),
        "neurocache_sync_cpu": BenchVariant(
            "neurocache_sync_cpu",
            offload_mode="cpu",
            min_tensor_kb=args.min_tensor_kb,
            async_transfer=False,
        ),
        "neurocache_cpu_fp16": BenchVariant("neurocache_cpu_fp16", offload_mode="cpu_fp16", min_tensor_kb=args.min_tensor_kb),
        "neurocache_cpu_int8": BenchVariant("neurocache_cpu_int8", offload_mode="cpu_int8", min_tensor_kb=args.min_tensor_kb),
        "neurocache_checkpoint_fp16": BenchVariant(
            "neurocache_checkpoint_fp16",
            checkpoint_blocks=True,
            offload_mode="cpu_fp16",
            min_tensor_kb=args.min_tensor_kb,
            max_pending_d2h=args.max_pending_d2h,
        ),
        "neurocache_checkpoint_cpu": BenchVariant(
            "neurocache_checkpoint_cpu",
            checkpoint_blocks=True,
            offload_mode="cpu",
            min_tensor_kb=args.min_tensor_kb,
            max_pending_d2h=args.max_pending_d2h,
        ),
        "neurocache_sync_checkpoint_fp16": BenchVariant(
            "neurocache_sync_checkpoint_fp16",
            checkpoint_blocks=True,
            offload_mode="cpu_fp16",
            min_tensor_kb=args.min_tensor_kb,
            async_transfer=False,
        ),
        "neurocache_sync_checkpoint_cpu": BenchVariant(
            "neurocache_sync_checkpoint_cpu",
            checkpoint_blocks=True,
            offload_mode="cpu",
            min_tensor_kb=args.min_tensor_kb,
            async_transfer=False,
        ),
        "neurocache_sync_checkpoint_cpu_cpuadam": BenchVariant(
            "neurocache_sync_checkpoint_cpu_cpuadam",
            checkpoint_blocks=True,
            offload_mode="cpu",
            min_tensor_kb=args.min_tensor_kb,
            async_transfer=False,
            cpu_optimizer=True,
        ),
        "neurocache_sync_checkpoint_cpu_hybridadam": BenchVariant(
            "neurocache_sync_checkpoint_cpu_hybridadam",
            checkpoint_blocks=True,
            offload_mode="cpu",
            min_tensor_kb=args.min_tensor_kb,
            async_transfer=False,
            hybrid_optimizer=True,
        ),
        "neurocache_budget4_checkpoint_cpu_hybridadam": BenchVariant(
            "neurocache_budget4_checkpoint_cpu_hybridadam",
            checkpoint_blocks=True,
            offload_mode="cpu",
            min_tensor_kb=args.min_tensor_kb,
            async_transfer=False,
            hybrid_optimizer=True,
            max_offloads_per_step=4,
        ),
        "neurocache_budget4_checkpoint_cpu_fp16adam": BenchVariant(
            "neurocache_budget4_checkpoint_cpu_fp16adam",
            checkpoint_blocks=True,
            offload_mode="cpu",
            min_tensor_kb=args.min_tensor_kb,
            async_transfer=False,
            fp16_optimizer=True,
            max_offloads_per_step=4,
        ),
        "neurocache_budget4_checkpoint_cpu_bf16adam": BenchVariant(
            "neurocache_budget4_checkpoint_cpu_bf16adam",
            checkpoint_blocks=True,
            offload_mode="cpu",
            min_tensor_kb=args.min_tensor_kb,
            async_transfer=False,
            bf16_optimizer=True,
            max_offloads_per_step=4,
        ),
        "neurocache_budget5_checkpoint_cpu_bf16adam": BenchVariant(
            "neurocache_budget5_checkpoint_cpu_bf16adam",
            checkpoint_blocks=True,
            offload_mode="cpu",
            min_tensor_kb=args.min_tensor_kb,
            async_transfer=False,
            bf16_optimizer=True,
            max_offloads_per_step=5,
        ),
        "neurocache_budget8_checkpoint_cpu_hybridadam": BenchVariant(
            "neurocache_budget8_checkpoint_cpu_hybridadam",
            checkpoint_blocks=True,
            offload_mode="cpu",
            min_tensor_kb=args.min_tensor_kb,
            async_transfer=False,
            hybrid_optimizer=True,
            max_offloads_per_step=8,
        ),
        "neurocache_heuristic_checkpoint_fp16": BenchVariant(
            "neurocache_heuristic_checkpoint_fp16",
            checkpoint_blocks=True,
            offload_mode="cpu_fp16",
            min_tensor_kb=args.min_tensor_kb,
            offload_policy="heuristic",
            max_pending_d2h=args.max_pending_d2h,
        ),
        "neurocache_predictor_checkpoint_fp16": BenchVariant(
            "neurocache_predictor_checkpoint_fp16",
            checkpoint_blocks=True,
            offload_mode="cpu_fp16",
            min_tensor_kb=args.min_tensor_kb,
            offload_policy="predictor",
            predictor_path=args.predictor_path,
            max_pending_d2h=args.max_pending_d2h,
        ),
    }
    variants = [variant_map[name.strip()] for name in args.variants.split(",") if name.strip()]

    metadata = {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_capability": torch.cuda.get_device_capability(0),
        "nvidia_smi": get_nvidia_smi(),
        "system_ram_gb": psutil.virtual_memory().total / 1e9,
        "model_config": asdict(cfg),
        "notes": [
            "Synthetic random tokens are used to isolate systems performance.",
            "Peak CUDA memory is measured with torch.cuda.max_memory_allocated.",
            "NeuroCache variants use real autograd saved-tensor hooks, not simulated scores.",
        ],
    }

    print(json.dumps({"metadata": metadata}, indent=2))
    results = []
    for variant in variants:
        print(f"\n[benchmark] running {variant.name}...")
        result = run_variant(
            variant=variant,
            cfg=cfg,
            device=device,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
            seed=args.seed,
        )
        results.append(result)
        print(json.dumps(summarize([results[0], result])[-1], indent=2, default=str))
        if result.get("oom"):
            print("[benchmark] stopping after CUDA OOM to avoid a poisoned CUDA context")
            break

    write_outputs(results, metadata, Path(args.output_dir))
    print(f"\n[benchmark] wrote results to {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
