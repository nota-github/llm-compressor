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
from tqdm import tqdm
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
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    single_path = os.path.join(model_dir, "model.safetensors")
    
    # Case 1: Sharded model with index file
    if os.path.exists(index_path):
        with open(index_path) as f:
            return json.load(f)["weight_map"]
    
    # Case 2: Single safetensors file
    if os.path.exists(single_path):
        with safe_open(single_path, framework="pt") as f:
            return {key: "model.safetensors" for key in f.keys()}
    
    raise FileNotFoundError(
        f"Could not find 'model.safetensors.index.json' or 'model.safetensors' in {model_dir}"
    )

def invert_weight_map(weight_map: dict[str, str]) -> dict[str, list[str]]:
    files: dict[str, list[str]] = {}
    for name, fn in weight_map.items():
        files.setdefault(fn, []).append(name)
    return {fn: sorted(names) for fn, names in sorted(files.items())}

def compute_mxfp4_scale_u8(absmax: torch.Tensor, fp4_max: float = 6.0,
                           weight: torch.Tensor = None, optimize: bool = False):
    # 기준 exp 계산 (heuristic)
    target = torch.maximum(absmax / fp4_max, torch.tensor(2.0**-126, device=absmax.device))
    base_exp = torch.round(torch.log2(target)).to(torch.int32)
    scale = torch.pow(2.0, base_exp.to(torch.float32))
    # no-clipping fix
    base_exp = base_exp + ((absmax / scale) > fp4_max).to(torch.int32)
    
    if not optimize or weight is None:
        # 기존 heuristic 방식
        return torch.clamp(base_exp + 127, min=1, max=254).to(torch.uint8), None, None
    
    # Heuristic MSE 계산 (per-group)
    heuristic_scale = torch.pow(2.0, base_exp.float())
    heuristic_scaled = weight / heuristic_scale.unsqueeze(-1)
    heuristic_quantized = FP4_E2M1_DATA.cast_to_fp4(heuristic_scaled.to(torch.bfloat16))
    heuristic_dequantized = heuristic_quantized.float() * heuristic_scale.unsqueeze(-1)
    heuristic_mse_per_group = ((weight - heuristic_dequantized) ** 2).mean(dim=-1)
    heuristic_mse = heuristic_mse_per_group.mean()
    
    # Grid search: exp-5 ~ exp-1 (휴리스틱보다 작은 방향만)
    # best_mse를 heuristic MSE로 초기화하여 더 좋은 경우에만 업데이트
    best_exp = base_exp.clone()
    best_mse = heuristic_mse_per_group.clone()
    
    for delta in range(-5, 0):  # -5, -4, -3, -2, -1
        candidate_exp = base_exp + delta
        scale = torch.pow(2.0, candidate_exp.float())
        
        # FP4 양자화 및 복원
        scaled = weight / scale.unsqueeze(-1)
        quantized = FP4_E2M1_DATA.cast_to_fp4(scaled.to(torch.bfloat16))
        dequantized = quantized.float() * scale.unsqueeze(-1)
        
        # MSE 계산
        mse = ((weight - dequantized) ** 2).mean(dim=-1)
        
        # 더 좋은 경우 업데이트
        better = mse < best_mse
        best_exp = torch.where(better, candidate_exp, best_exp)
        best_mse = torch.where(better, mse, best_mse)
    
    # Optimized MSE 계산
    optimized_mse = best_mse.mean()
    
    return torch.clamp(best_exp + 127, min=1, max=254).to(torch.uint8), heuristic_mse.item(), optimized_mse.item()

def quantize_fp4_pack(weight: torch.Tensor, group_size: int = 32, fp4_max: float = 6.0,
                      optimize: bool = False):
    out, in_f = weight.shape
    groups = in_f // group_size
    weight_grouped = weight.view(out, groups, group_size)
    absmax = weight_grouped.abs().amax(dim=-1).to(torch.float32)
    scale_u8, heuristic_mse, optimized_mse = compute_mxfp4_scale_u8(
        absmax, fp4_max,
        weight=weight_grouped.to(torch.float32) if optimize else None,
        optimize=optimize
    )
    scale = scale_u8.view(torch.float8_e8m0fnu).to(torch.float32)
    scaled = (weight_grouped / scale.view(out, groups, 1)).to(torch.bfloat16)
    q = FP4_E2M1_DATA.cast_to_fp4(scaled).reshape(out, in_f)
    return pack_fp4_to_uint8(q).contiguous(), scale_u8.contiguous(), heuristic_mse, optimized_mse

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

def convert(orig_dir: str, out_dir: str, group_size: int = 32, fp4_max: float = 6.0, 
            shard_size_gb: int = 5, layers: Optional[list[int]] = None, optimize_scale: bool = False):
    # Auto-detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(out_dir, exist_ok=True)
    w_map = load_weight_map(orig_dir)
    
    detect_checkpoint_layout(w_map)

    # Copy metadata files
    for fn in os.listdir(orig_dir):
        if not fn.endswith(".safetensors") and fn not in {"model.safetensors.index.json", "config.json"} and not fn.startswith('global_step'):
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
    target_layers = layer_ids if layers is None else layers
    for li in tqdm(target_layers, desc="Processing layers"):
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

            layer_heuristic_mse, layer_optimized_mse = [], []
            for ei in range(num_e):
                Wi, Wd = gu_sl[ei].to(device=device, dtype=torch.float32), d_sl[ei].to(device=device, dtype=torch.float32)
                Wg, Wu = Wi[:, 0::2].t().contiguous(), Wi[:, 1::2].t().contiguous()
                Wdown = Wd.t().contiguous()

                g_p, g_s, g_h_mse, g_o_mse = quantize_fp4_pack(Wg, group_size, fp4_max, optimize_scale)
                u_p, u_s, u_h_mse, u_o_mse = quantize_fp4_pack(Wu, group_size, fp4_max, optimize_scale)
                d_p, d_s, d_h_mse, d_o_mse = quantize_fp4_pack(Wdown, group_size, fp4_max, optimize_scale)

                if optimize_scale:
                    layer_heuristic_mse.extend([g_h_mse, u_h_mse, d_h_mse])
                    layer_optimized_mse.extend([g_o_mse, u_o_mse, d_o_mse])

                gu_b_l.append(interleave_gate_up(reshape_to_gpt_oss_blocks(g_p), reshape_to_gpt_oss_blocks(u_p)))
                gu_s_l.append(interleave_gate_up(g_s, u_s))
                d_b_l.append(reshape_to_gpt_oss_blocks(d_p))
                d_s_l.append(d_s)

            if optimize_scale and layer_heuristic_mse:
                avg_h_mse = sum(layer_heuristic_mse) / len(layer_heuristic_mse)
                avg_o_mse = sum(layer_optimized_mse) / len(layer_optimized_mse)
                reduction = (1 - avg_o_mse / avg_h_mse) * 100 if avg_h_mse > 0 else 0
                tqdm.write(f"  Layer {li}: MSE {avg_h_mse:.6e} -> {avg_o_mse:.6e} (reduction: {reduction:.2f}%)")

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
    p.add_argument("--optimize-scale", action="store_true",
                   help="Enable grid search to find optimal scale (slower but more accurate)")
    
    args = p.parse_args()
    orig_dir = resolve_model_dir(args.orig)
    layers = [int(x) for x in args.layers.split(",") if x.strip()] if args.layers else None
    
    convert(orig_dir, args.out, args.group_size, args.fp4_max, args.shard_size_gb, layers, args.optimize_scale)
    print(f"[OK] Wrote GPT-OSS MXFP4 model to: {args.out}")

if __name__ == "__main__":
    main()
