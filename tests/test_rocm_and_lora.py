from __future__ import annotations

import os
import tempfile

import torch
import torch.nn as nn
from safetensors.torch import save_file

from lora_trainer.core.device import auto_memory_plan, gpu_info
from lora_trainer.core.model_files import identify
from lora_trainer.core.recommend import DatasetStats, recommend
from lora_trainer.core.sysinfo import detect_system
from lora_trainer.gui.app_settings import AppSettings
from lora_trainer.network.lora import LoRANetwork, build_target_filter
from lora_trainer.quant.layers import FP8Linear, fp8_supported
from lora_trainer.quant.loader import LazyCheckpoint


def test_gpu_and_fp8_linear():
    info = gpu_info()
    assert info.available
    assert fp8_supported()
    device = torch.device("cuda:0")
    w = torch.randn(32, 64, device=device, dtype=torch.bfloat16)
    b = torch.randn(32, device=device, dtype=torch.bfloat16)
    layer = FP8Linear.from_weight(w, b, torch.bfloat16, device)
    x = torch.randn(4, 64, device=device, dtype=torch.bfloat16)
    out = layer(x)
    assert out.shape == (4, 32)
    assert out.dtype == torch.bfloat16


def test_auto_memory_plan_and_recommend_fits():
    spec = detect_system()
    assert spec.gpu_available
    for arch in ("moss_image", "krea2"):
        plan = auto_memory_plan(arch, spec.vram_gb)
        assert plan.transformer_quant in ("fp8", "nf4", "none")
        analysis = recommend(arch, spec, DatasetStats(images=20, captioned=20, min_short=1024, median_short=1024, weighted=20, folders=1))
        assert len(analysis.options) == 3
        assert any(o.fits for o in analysis.options)


def test_split_qkv_lora_merge_load():
    class DummyBlock(nn.Module):
        def __init__(self, dim: int = 32):
            super().__init__()
            self.proj = nn.ModuleDict({
                "qkv": nn.Linear(dim, 3 * dim, bias=False),
                "o": nn.Linear(dim, dim, bias=False),
            })

    class DummyModel(nn.Module):
        def __init__(self, dim: int = 32):
            super().__init__()
            self.layers = nn.ModuleList([DummyBlock(dim)])

    dim = 32
    rank = 12
    part_rank = 4
    model = DummyModel(dim)
    filt = build_target_filter({"attn": [r"layers\.\d+\.proj\.(qkv|o)"]}, "attn")
    net = LoRANetwork(model, filt, rank=rank, alpha=float(rank))

    sd = {}
    parts_delta = []
    for part in ("q", "k", "v"):
        base = f"diffusion_model.layers.0.proj.{part}"
        d = torch.randn(part_rank, dim) * 0.1
        u = torch.randn(dim, part_rank) * 0.1
        sd[f"{base}.lora_down.weight"] = d
        sd[f"{base}.lora_up.weight"] = u
        sd[f"{base}.alpha"] = torch.tensor(float(part_rank))
        parts_delta.append(u @ d)
    sd["diffusion_model.layers.0.proj.o.lora_down.weight"] = torch.randn(rank, dim) * 0.1
    sd["diffusion_model.layers.0.proj.o.lora_up.weight"] = torch.randn(dim, rank) * 0.1
    sd["diffusion_model.layers.0.proj.o.alpha"] = torch.tensor(float(rank))

    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "split_qkv.safetensors")
        save_file(sd, p)
        loaded, total = net.load_weights(p)
        assert total == 2
        assert loaded == 2
        qkv_lora = [l for l in net.loras if l.lora_name == "layers.0.proj.qkv"][0]
        expected_delta = torch.cat(parts_delta, dim=0)
        actual_delta = (qkv_lora.lora_up.weight @ qkv_lora.lora_down.weight) * qkv_lora.scale
        assert torch.allclose(actual_delta, expected_delta, atol=1e-5)


def test_comfy_models_and_convrot_dequant():
    settings = AppSettings()
    moss_dit = settings.model_paths.get("moss_image", {}).get("transformer", "")
    if moss_dit and os.path.isfile(moss_dit):
        assert identify(moss_dit) == "moss_dit"
        ckpt = LazyCheckpoint([moss_dit])
        w = ckpt.get("layers.0.proj.qkv.weight")
        assert w.dtype == torch.float32
        assert w.shape == (9216, 3072)
        assert torch.isfinite(w).all()
        ckpt.close()
