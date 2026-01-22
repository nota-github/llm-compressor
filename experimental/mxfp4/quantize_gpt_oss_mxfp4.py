#!/usr/bin/env python3
"""
Directly convert a GPT-OSS checkpoint (local path or HF repo id) into a
GPT-OSS-compatible MXFP4 MoE checkpoint (gate_up/down *_blocks/*_scales).

Key properties:
- Does NOT require an existing *-MXFP4 directory.
- Keeps FP4(E2M1) rounding + uint8 nibble packing identical to `compressed_tensors`.
- Changes ONLY the per-(out,row,group) scale rule (v2 rule):
    scale ~= absmax/6, then power-of-two, then a 1-step no-clipping fix.
- Copies all non-expert tensors from the original checkpoint as-is.
- Removes original fused expert weights (gate_up_proj, down_proj) from output
  to avoid vLLM accidentally loading the wrong tensors.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from compressed_tensors.compressors.quantized_compressors.fp4_quantized import (
    pack_fp4_to_uint8,
)
from compressed_tensors.quantization.quant_args import FP4_E2M1_DATA

# Regex helpers for weight mapping
_FUSED_GATE_UP_KEY = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj$")
_FUSED_DOWN_KEY = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.down_proj$")
_FUSED_GATE_UP_BIAS_KEY = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj_bias$")
_FUSED_DOWN_BIAS_KEY = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.down_proj_bias$")

def detect_checkpoint_layout(weight_map: dict[str, str]) -> str:
    if any(_FUSED_GATE_UP_KEY.match(k) for k in weight_map):
        return "fused_fp32"
    raise ValueError("Unsupported checkpoint layout: could not find expert tensors.")

def resolve_model_dir(orig: str) -> str:
    if os.path.isdir(orig):
        return orig
    raise ValueError(f"Input path '{orig}' is not a valid directory. Only local directories are supported.")

def load_weight_map(model_dir: str) -> dict[str, str]:
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]

def invert_weight_map(weight_map: dict[str, str]) -> dict[str, list[str]]:
    files: dict[str, list[str]] = {}
    for name, fn in weight_map.items():
        files.setdefault(fn, []).append(name)
    return {fn: sorted(names) for fn, names in sorted(files.items())}

def compute_mxfp4_scale_u8(absmax: torch.Tensor, fp4_max: float = 6.0) -> torch.Tensor:
    target = torch.maximum(absmax / fp4_max, torch.tensor(2.0**-126, device=absmax.device))
    exp = torch.round(torch.log2(target)).to(torch.int32)
    scale = torch.pow(2.0, exp.to(torch.float32))
    # no-clipping fix
    exp = exp + ((absmax / scale) > fp4_max).to(torch.int32)
    return torch.clamp(exp + 127, min=1, max=254).to(torch.uint8)

def quantize_fp4_pack(weight: torch.Tensor, group_size: int = 32, fp4_max: float = 6.0) -> tuple[torch.Tensor, torch.Tensor]:
    out, in_f = weight.shape
    groups = in_f // group_size
    absmax = weight.view(out, groups, group_size).abs().amax(dim=-1).to(torch.float32)
    scale_u8 = compute_mxfp4_scale_u8(absmax, fp4_max)
    scale = scale_u8.view(torch.float8_e8m0fnu).to(torch.float32)
    scaled = (weight.view(out, groups, group_size) / scale.view(out, groups, 1)).to(torch.bfloat16)
    q = FP4_E2M1_DATA.cast_to_fp4(scaled).reshape(out, in_f)
    return pack_fp4_to_uint8(q).contiguous(), scale_u8.contiguous()

def reshape_to_gpt_oss_blocks(packed: torch.Tensor) -> torch.Tensor:
    out, p_in = packed.shape
    return packed.reshape(out, p_in // 16, 16)

def interleave_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    out = torch.empty((gate.shape[0] * 2,) + gate.shape[1:], dtype=gate.dtype)
    out[0::2], out[1::2] = gate, up
    return out

@dataclass
class ShardWriter:
    out_dir: str
    shard_size_bytes: int
    shard_idx: int = 0
    cur: dict = field(default_factory=dict)
    cur_size: int = 0
    total_size: int = 0
    weight_map: dict = field(default_factory=dict)
    shard_files: list = field(default_factory=list)

    def add(self, name: str, tensor: torch.Tensor):
        tensor = tensor.cpu().contiguous()
        size = tensor.numel() * tensor.element_size()
        if self.cur and self.cur_size + size > self.shard_size_bytes:
            self.flush()
        self.cur[name], self.cur_size, self.total_size = tensor, self.cur_size + size, self.total_size + size
        self.weight_map[name] = f"shard-{self.shard_idx:05d}.safetensors"

    def flush(self):
        if not self.cur: return
        fn = f"shard-{self.shard_idx:05d}.safetensors"
        save_file(self.cur, os.path.join(self.out_dir, fn))
        self.shard_files.append(fn)
        self.cur, self.cur_size, self.shard_idx = {}, 0, self.shard_idx + 1

    def finalize(self) -> dict:
        self.flush()
        num = len(self.shard_files)
        res = {}
        for i, old in enumerate(self.shard_files):
            new = f"model-{i:05d}-of-{num:05d}.safetensors"
            os.replace(os.path.join(self.out_dir, old), os.path.join(self.out_dir, new))
            res[old] = new
        return {
            "metadata": {"total_size": self.total_size},
            "weight_map": {k: res[v] for k, v in self.weight_map.items()}
        }

def convert(orig_dir: str, out_dir: str, group_size: int = 32, fp4_max: float = 6.0, shard_size_gb: int = 5, layers: Optional[list[int]] = None):
    os.makedirs(out_dir, exist_ok=True)
    w_map = load_weight_map(orig_dir)
    
    detect_checkpoint_layout(w_map)

    # Copy metadata files
    for fn in os.listdir(orig_dir):
        if not fn.endswith(".safetensors") and fn not in {"model.safetensors.index.json", "config.json"}:
            shutil.copy2(os.path.join(orig_dir, fn), os.path.join(out_dir, fn))
    
    # Update config.json
    with open(os.path.join(orig_dir, "config.json")) as f:
        cfg = json.load(f)
    cfg["quantization_config"] = {
        "modules_to_not_convert": ["model.layers.*.self_attn", "model.layers.*.mlp.router", "model.embed_tokens", "lm_head"],
        "quant_method": "mxfp4"
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    writer = ShardWriter(out_dir, shard_size_gb * 1024**3)
    f_map = invert_weight_map(w_map)
    
    # 1) Copy non-expert tensors
    for fn, names in f_map.items():
        with safe_open(os.path.join(orig_dir, fn), framework="pt") as f:
            for n in names:
                if not any(r.match(n) for r in [_FUSED_GATE_UP_KEY, _FUSED_DOWN_KEY, _FUSED_GATE_UP_BIAS_KEY, _FUSED_DOWN_BIAS_KEY]):
                    writer.add(n, f.get_tensor(n))

    # 2) Process and Quantize Expert tensors
    layer_ids = sorted({int(_FUSED_GATE_UP_KEY.match(k).group(1)) for k in w_map if _FUSED_GATE_UP_KEY.match(k)})
    for li in (layer_ids if layers is None else layers):
        p = f"model.layers.{li}.mlp.experts"
        with safe_open(os.path.join(orig_dir, w_map[f"{p}.gate_up_proj"]), framework="pt") as f_gu, \
             safe_open(os.path.join(orig_dir, w_map[f"{p}.down_proj"]), framework="pt") as f_d, \
             safe_open(os.path.join(orig_dir, w_map[f"{p}.gate_up_proj_bias"]), framework="pt") as f_gub, \
             safe_open(os.path.join(orig_dir, w_map[f"{p}.down_proj_bias"]), framework="pt") as f_db:
            
            gu_sl, d_sl = f_gu.get_slice(f"{p}.gate_up_proj"), f_d.get_slice(f"{p}.down_proj")
            gub_sl, db_sl = f_gub.get_slice(f"{p}.gate_up_proj_bias"), f_db.get_slice(f"{p}.down_proj_bias")
            
            num_e = gu_sl.get_shape()[0]
            gu_b_l, gu_s_l, d_b_l, d_s_l = [], [], [], []
            writer.add(f"{p}.gate_up_proj_bias", gub_sl[:].to(torch.bfloat16))
            writer.add(f"{p}.down_proj_bias", db_sl[:].to(torch.bfloat16))

            for ei in range(num_e):
                Wi, Wd = gu_sl[ei].to(torch.float32), d_sl[ei].to(torch.float32)
                Wg, Wu = Wi[:, 0::2].t().contiguous(), Wi[:, 1::2].t().contiguous()
                Wdown = Wd.t().contiguous()

                g_p, g_s = quantize_fp4_pack(Wg, group_size, fp4_max)
                u_p, u_s = quantize_fp4_pack(Wu, group_size, fp4_max)
                d_p, d_s = quantize_fp4_pack(Wdown, group_size, fp4_max)

                gu_b_l.append(interleave_gate_up(reshape_to_gpt_oss_blocks(g_p), reshape_to_gpt_oss_blocks(u_p)))
                gu_s_l.append(interleave_gate_up(g_s, u_s))
                d_b_l.append(reshape_to_gpt_oss_blocks(d_p))
                d_s_l.append(d_s)

            writer.add(f"{p}.gate_up_proj_blocks", torch.stack(gu_b_l))
            writer.add(f"{p}.gate_up_proj_scales", torch.stack(gu_s_l))
            writer.add(f"{p}.down_proj_blocks", torch.stack(d_b_l))
            writer.add(f"{p}.down_proj_scales", torch.stack(d_s_l))

    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(writer.finalize(), f, indent=2)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--orig", required=True, help="Local directory or HF repo id")
    p.add_argument("--out", required=True, help="Output directory (*-gpt-oss)")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--fp4-max", type=float, default=6.0)
    p.add_argument("--shard-size-gb", type=int, default=5)
    p.add_argument("--layers", default=None, help="Comma-separated layer indices to process")
    
    args = p.parse_args()
    orig_dir = resolve_model_dir(args.orig)
    layers = [int(x) for x in args.layers.split(",") if x.strip()] if args.layers else None
    
    convert(orig_dir, args.out, args.group_size, args.fp4_max, args.shard_size_gb, layers)
    print(f"[OK] Wrote GPT-OSS MXFP4 model to: {args.out}")

if __name__ == "__main__":
    main()
