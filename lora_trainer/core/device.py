"""GPU 정보 확인 및 VRAM 에 따른 자동 메모리 옵션 결정."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import importlib.util
import torch

_BNB_OK: Optional[bool] = None


def bnb_quant_available() -> bool:
    """bitsandbytes 4bit/8bit GPU 양자화가 현재 환경에서 실제로 동작하는지 확인."""
    global _BNB_OK
    if _BNB_OK is not None:
        return _BNB_OK
    if not torch.cuda.is_available() or importlib.util.find_spec("bitsandbytes") is None:
        _BNB_OK = False
        return False
    try:
        import bitsandbytes as bnb

        _BNB_OK = hasattr(bnb.nn, "Linear4bit") and getattr(bnb.nn, "Params4bit", None) is not None
    except Exception:
        _BNB_OK = False
    return _BNB_OK


@dataclass
class GPUInfo:
    available: bool
    name: str = "CPU"
    total_gb: float = 0.0
    capability: tuple = (0, 0)

    @property
    def supports_bf16(self) -> bool:
        return self.available and self.capability >= (8, 0)

    def describe(self) -> str:
        if not self.available:
            return "CUDA GPU 없음 (CPU 모드)"
        return f"{self.name} · {self.total_gb:.1f} GB · SM {self.capability[0]}.{self.capability[1]}"


def gpu_info(index: int = 0) -> GPUInfo:
    if not torch.cuda.is_available():
        return GPUInfo(available=False)
    props = torch.cuda.get_device_properties(index)
    return GPUInfo(
        available=True,
        name=props.name,
        total_gb=props.total_memory / (1024 ** 3),
        capability=(props.major, props.minor),
    )


def vram_used_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def free_memory() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class MemoryPlan:
    transformer_quant: str
    text_encoder_quant: str
    block_swap: int
    note: str


def auto_memory_plan(arch: str, vram_gb: Optional[float] = None) -> MemoryPlan:
    """모델/VRAM 에 맞는 기본 양자화·블록스왑 조합.

    - Krea 2  : 12.9B DiT (bf16 약 26GB), Qwen3-VL-4B 텍스트 인코더 (bf16 약 9GB)
    - MOSS    : 5B DiT (bf16 약 10GB), Qwen3.5-9B 텍스트 인코더 (bf16 약 18GB)
    텍스트 인코더/VAE 는 사전 캐싱 단계에서만 GPU 에 올라가므로 DiT 와 동시에 쓰지 않는다.
    """
    if vram_gb is None:
        vram_gb = gpu_info().total_gb
    bnb_ok = bnb_quant_available()
    if arch == "krea2":
        if vram_gb >= 40:
            return MemoryPlan("none", "none", 0, "bf16 전체 로딩")
        if vram_gb >= 22:
            return MemoryPlan("fp8", "none", 0, "fp8 DiT (약 13GB)")
        if not bnb_ok:
            swap = 10 if vram_gb >= 14 else (16 if vram_gb >= 11 else 22)
            return MemoryPlan("fp8", "none", swap, f"FP8 DiT + 블록 스왑 {swap} (bitsandbytes 미사용 환경)")
        if vram_gb >= 11:
            return MemoryPlan("nf4", "none", 0, "NF4 4bit DiT (약 7GB) + bf16 텍스트 인코더")
        return MemoryPlan("nf4", "nf4", 0, "NF4 4bit DiT + NF4 텍스트 인코더 (저VRAM)")
    # moss_image
    if vram_gb >= 22:
        return MemoryPlan("none", "none", 0, "bf16 전체 로딩")
    if vram_gb >= 14:
        return MemoryPlan("fp8", "int8" if bnb_ok else "none", 0, "fp8 DiT (약 5GB) + 텍스트 인코더")
    if not bnb_ok:
        swap = 0 if vram_gb >= 9 else 12
        return MemoryPlan("fp8", "none", swap, f"FP8 DiT + 블록 스왑 {swap} (bitsandbytes 미사용 환경)")
    if vram_gb >= 9:
        return MemoryPlan("fp8", "nf4", 0, "fp8 DiT (약 5GB) + NF4 텍스트 인코더")
    return MemoryPlan("nf4", "nf4", 0, "NF4 DiT + NF4 텍스트 인코더 (저VRAM)")


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(name, torch.bfloat16)
