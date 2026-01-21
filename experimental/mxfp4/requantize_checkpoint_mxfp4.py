#!/usr/bin/env python3
"""
Re-quantize GPT-OSS checkpoint MoE expert weights to MXFP4 with a different
scale rule, and output a compressed-tensors-style checkpoint directory.

Key idea:
  - Keep the FP4(E2M1) codebook + uint8 nibble packing identical to
    `compressed_tensors` (for compatibility)
  - Change ONLY the per-(row,group) scale rule:
        scale ≈ absmax / 6, then power-of-two (MXFP4) with minimal no-clipping fix.

This script overlays updated expert tensors onto an existing MXFP4 directory
(so non-expert tensors/config/tokenizer stay identical), by writing new
*.safetensors files and updating model.safetensors.index.json weight_map.

Example:
  python3 requantize_checkpoint_mxfp4.py \
    --orig /mnt/nas/.../checkpoint-705 \
    --base /home/beomseok.kwon/llm-compressor/experimental/mxfp4/checkpoint-705-MXFP4 \
    --out  /home/beomseok.kwon/llm-compressor/experimental/mxfp4/checkpoint-705-MXFP4-v2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from compressed_tensors.compressors.quantized_compressors.fp4_quantized import (
    pack_fp4_to_uint8,
)
from compressed_tensors.quantization.quant_args import FP4_E2M1_DATA


def _load_weight_map(model_dir: str) -> dict[str, str]:
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    with open(idx, "r") as f:
        return json.load(f)["weight_map"]


def _load_index_json(model_dir: str) -> dict:
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    with open(idx, "r") as f:
        return json.load(f)


def _save_index_json(model_dir: str, index_obj: dict) -> None:
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    with open(idx, "w") as f:
        json.dump(index_obj, f, indent=2)


def _iter_layers_from_weight_map(weight_map: dict[str, str]) -> list[int]:
    layers = set()
    prefix = "model.layers."
    for k in weight_map.keys():
        if not k.startswith(prefix):
            continue
        rest = k[len(prefix) :]
        # rest begins with "{layer}."
        layer_str = rest.split(".", 1)[0]
        if layer_str.isdigit():
            layers.add(int(layer_str))
    return sorted(layers)


def _float8_scale_from_u8(scale_u8: torch.Tensor) -> torch.Tensor:
    # u8 exponent-coded -> float8_e8m0fnu reinterpret -> float32 scale value
    return scale_u8.view(torch.float8_e8m0fnu).to(torch.float32)


def _compute_mxfp4_scale_u8_from_absmax(
    absmax: torch.Tensor,
    fp4_max: float = 6.0,
) -> torch.Tensor:
    """
    absmax: [...], float32 >= 0
    returns: uint8 exponent-coded (reinterpret as float8_e8m0 gives power-of-two scale)
    """
    # target ~= absmax / fp4_max
    target = absmax / fp4_max
    # smallest positive float8_e8m0 exponent value is 2**(-126) (exp byte 1)
    min_scale = torch.tensor(2.0 ** (-126), device=absmax.device, dtype=torch.float32)
    target = torch.maximum(target, min_scale)

    # nearest power-of-two via rounding log2
    exp = torch.round(torch.log2(target)).to(torch.int32)  # unbiased exponent
    scale = torch.pow(2.0, exp.to(torch.float32))

    # ensure no clipping: absmax/scale <= fp4_max (one-step fix is enough)
    ratio = absmax / scale
    exp = exp + (ratio > fp4_max).to(torch.int32)

    # clamp to float8_e8m0 exponent range [1, 254] after bias
    exp = torch.clamp(exp, min=-126, max=127)
    scale_u8 = torch.clamp(exp + 127, min=1, max=254).to(torch.uint8)
    return scale_u8


def _quantize_fp4_and_pack(
    weight: torch.Tensor,  # [out, in] float32
    scale_u8: torch.Tensor,  # [out, groups] uint8 (float8_e8m0 exponent-coded)
    group_size: int = 32,
) -> torch.Tensor:
    out, in_features = weight.shape
    assert in_features % group_size == 0
    groups = in_features // group_size
    assert tuple(scale_u8.shape) == (out, groups)

    scale = _float8_scale_from_u8(scale_u8)  # float32
    w3 = weight.view(out, groups, group_size)
    s3 = scale.view(out, groups, 1)
    scaled = (w3 / s3).to(torch.bfloat16)

    # FP4(E2M1) rounding (values in [-6, 6])
    q = FP4_E2M1_DATA.cast_to_fp4(scaled)
    q2d = q.reshape(out, in_features).contiguous()

    packed = pack_fp4_to_uint8(q2d)
    return packed.contiguous()


def _get_slice_dtype_shape(
    model_dir: str,
    weight_map: dict[str, str],
    key: str,
) -> Tuple[str, list[int]]:
    file_name = weight_map[key]
    path = os.path.join(model_dir, file_name)
    with safe_open(path, framework="pt") as f:
        sl = f.get_slice(key)
        return sl.get_dtype(), list(sl.get_shape())


def requantize(
    orig_dir: str,
    base_mxfp4_dir: str,
    out_dir: str,
    group_size: int = 32,
    fp4_max: float = 6.0,
):
    # 0) Copy base directory to out_dir
    out_path = Path(out_dir)
    if out_path.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    shutil.copytree(base_mxfp4_dir, out_dir)

    # 1) Read indices
    orig_weight_map = _load_weight_map(orig_dir)
    out_index = _load_index_json(out_dir)
    out_weight_map: dict[str, str] = out_index["weight_map"]

    layers = _iter_layers_from_weight_map(orig_weight_map)

    # Keep only layers that have MoE experts in the original checkpoint
    moe_layers = []
    for li in layers:
        if f"model.layers.{li}.mlp.experts.gate_up_proj" in orig_weight_map:
            moe_layers.append(li)

    if not moe_layers:
        raise RuntimeError("No GPT-OSS MoE layers found in original checkpoint index.")

    # Basic shape sanity from layer 0
    sample_li = moe_layers[0]
    gate_up_key = f"model.layers.{sample_li}.mlp.experts.gate_up_proj"
    down_key = f"model.layers.{sample_li}.mlp.experts.down_proj"
    _, gate_up_shape = _get_slice_dtype_shape(orig_dir, orig_weight_map, gate_up_key)
    _, down_shape = _get_slice_dtype_shape(orig_dir, orig_weight_map, down_key)
    # gate_up: [E, H, 2D]
    num_experts, hidden_size, twoD = gate_up_shape
    intermediate_size = twoD // 2
    if twoD % 2 != 0:
        raise ValueError(f"Unexpected gate_up_proj last dim (not even): {twoD}")
    if down_shape[0] != num_experts:
        raise ValueError("Unexpected down_proj expert dim mismatch.")

    if hidden_size % group_size != 0 or intermediate_size % group_size != 0:
        raise ValueError("hidden/intermediate must be divisible by group_size=32.")

    # 2) For each MoE layer, write an overlay safetensors with updated expert weights
    for li in moe_layers:
        gate_up_key = f"model.layers.{li}.mlp.experts.gate_up_proj"
        down_key = f"model.layers.{li}.mlp.experts.down_proj"
        gate_up_b_key = f"model.layers.{li}.mlp.experts.gate_up_proj_bias"
        down_b_key = f"model.layers.{li}.mlp.experts.down_proj_bias"

        gate_up_path = os.path.join(orig_dir, orig_weight_map[gate_up_key])
        down_path = os.path.join(orig_dir, orig_weight_map[down_key])
        gate_up_b_path = os.path.join(orig_dir, orig_weight_map[gate_up_b_key])
        down_b_path = os.path.join(orig_dir, orig_weight_map[down_b_key])

        with (
            safe_open(gate_up_path, framework="pt") as gate_up_file,
            safe_open(down_path, framework="pt") as down_file,
            safe_open(gate_up_b_path, framework="pt") as gate_up_b_file,
            safe_open(down_b_path, framework="pt") as down_b_file,
        ):
            gate_up_sl = gate_up_file.get_slice(gate_up_key)
            down_sl = down_file.get_slice(down_key)
            gate_up_b_sl = gate_up_b_file.get_slice(gate_up_b_key)
            down_b_sl = down_b_file.get_slice(down_b_key)

            layer_tensors: Dict[str, torch.Tensor] = {}

            for ei in range(num_experts):
                # Load original fused expert weights/biases (float32)
                # gate_up: [H, 2D], bias: [2D]
                Wi = gate_up_sl[ei].to(torch.float32)
                bi = gate_up_b_sl[ei].to(torch.float32)

                # down: [D, H], bias: [H]
                Wd = down_sl[ei].to(torch.float32)
                bd = down_b_sl[ei].to(torch.float32)

                # De-interleave and transpose to nn.Linear convention
                Wg = Wi[:, 0::2].contiguous().t().contiguous()  # [D, H]
                Wu = Wi[:, 1::2].contiguous().t().contiguous()  # [D, H]
                # down_proj.weight is [H, D]
                Wdown = Wd.contiguous().t().contiguous()  # [H, D]

                # Biases are not re-written (kept from base dir),
                # but we compute them here for potential debugging/sanity.
                _bg = bi[0::2].contiguous()
                _bu = bi[1::2].contiguous()
                _bd = bd.contiguous()
                assert _bg.numel() == intermediate_size and _bu.numel() == intermediate_size
                assert _bd.numel() == hidden_size

                # Quantize each projection
                for proj_name, W in (
                    ("gate_proj", Wg),
                    ("up_proj", Wu),
                    ("down_proj", Wdown),
                ):
                    # scale per (out,row) and group over input dim
                    out_dim, in_dim = W.shape
                    groups = in_dim // group_size
                    absmax = (
                        W.view(out_dim, groups, group_size)
                        .abs()
                        .amax(dim=-1)
                        .to(torch.float32)
                    )
                    scale_u8 = _compute_mxfp4_scale_u8_from_absmax(
                        absmax=absmax,
                        fp4_max=fp4_max,
                    )
                    packed = _quantize_fp4_and_pack(
                        weight=W,
                        scale_u8=scale_u8,
                        group_size=group_size,
                    )

                    base = (
                        f"model.layers.{li}.mlp.experts.experts.{ei}."
                        f"{proj_name}."
                    )
                    layer_tensors[base + "weight_packed"] = packed
                    layer_tensors[base + "weight_scale"] = scale_u8.contiguous()

        # Write overlay file for this layer (name chosen to sort after model-*.safetensors)
        overlay_name = f"zz-experts-layer-{li:02d}.safetensors"
        overlay_path = os.path.join(out_dir, overlay_name)
        save_file(layer_tensors, overlay_path)

        # Update index mapping for overridden keys
        for k in layer_tensors.keys():
            out_weight_map[k] = overlay_name

        # Free memory
        del layer_tensors

    # 3) Update index file in out_dir
    out_index["weight_map"] = out_weight_map
    _save_index_json(out_dir, out_index)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--orig", required=True, help="Original checkpoint directory")
    p.add_argument(
        "--base",
        required=True,
        help="Existing MXFP4 (compressed-tensors) directory to overlay",
    )
    p.add_argument("--out", required=True, help="Output directory (new MXFP4-v2)")
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--fp4-max", type=float, default=6.0)
    return p.parse_args()


def main():
    args = parse_args()
    requantize(
        orig_dir=args.orig,
        base_mxfp4_dir=args.base,
        out_dir=args.out,
        group_size=args.group_size,
        fp4_max=args.fp4_max,
    )
    print(f"[OK] Wrote: {args.out}")


if __name__ == "__main__":
    main()

