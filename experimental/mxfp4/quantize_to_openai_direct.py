#!/usr/bin/env python3
"""
Directly convert a GPT-OSS checkpoint (local path or HF repo id) into an
OpenAI/vLLM-compatible MXFP4 MoE checkpoint (gate_up/down *_blocks/*_scales).

Key properties:
- Does NOT require an existing *-MXFP4 directory.
- Keeps FP4(E2M1) rounding + uint8 nibble packing identical to `compressed_tensors`.
- Changes ONLY the per-(out,row,group) scale rule (v2 rule):
    scale ~= absmax/6, then power-of-two, then a 1-step no-clipping fix.
- Copies all non-expert tensors from the original checkpoint as-is.
- Removes original fused expert weights (gate_up_proj, down_proj) from output
  to avoid vLLM accidentally loading the wrong tensors.

Example (local):
  python3 quantize_to_openai_direct.py \
    --orig /mnt/nas/.../checkpoint-705 \
    --out  /home/.../checkpoint-705-MXFP4-v3-openai

Example (HF repo id):
  HF_TOKEN=... python3 quantize_to_openai_direct.py \
    --orig openai/gpt-oss-20b \
    --out  ./gpt-oss-20b-MXFP4-v3-openai
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from compressed_tensors.compressors.quantized_compressors.fp4_quantized import (
    pack_fp4_to_uint8,
)
from compressed_tensors.quantization.quant_args import FP4_E2M1_DATA


_FUSED_GATE_UP_KEY = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj$")
_FUSED_DOWN_KEY = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.down_proj$")
_FUSED_GATE_UP_BIAS_KEY = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj_bias$"
)
_FUSED_DOWN_BIAS_KEY = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.down_proj_bias$")

_OAI_GATE_UP_BLOCKS_KEY = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj_blocks$"
)
_OAI_DOWN_BLOCKS_KEY = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.down_proj_blocks$"
)


def detect_checkpoint_layout(weight_map: dict[str, str]) -> str:
    """
    Determine how MoE expert weights are represented in the source checkpoint.

    Returns:
      - "fused_fp32": has fused float weights (gate_up_proj / down_proj tensors)
      - "openai_mxfp4": already has *_blocks/*_scales tensors
    """
    has_oai = any(_OAI_GATE_UP_BLOCKS_KEY.match(k) for k in weight_map.keys())
    has_fused = any(_FUSED_GATE_UP_KEY.match(k) for k in weight_map.keys())
    if has_oai:
        return "openai_mxfp4"
    if has_fused:
        return "fused_fp32"
    raise ValueError(
        "Unsupported checkpoint layout: could not find either "
        "`model.layers.*.mlp.experts.gate_up_proj` (fused fp32) or "
        "`model.layers.*.mlp.experts.gate_up_proj_blocks` (openai mxfp4)."
    )


def resolve_model_dir(
    orig: str,
    cache_dir: Optional[str] = None,
) -> str:
    """Return a local directory containing HF-style files."""
    if os.path.isdir(orig):
        return orig

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "orig is not a local directory and huggingface_hub is unavailable. "
            "Install huggingface_hub or provide a local path."
        ) from e

    allow_patterns = [
        "*.json",
        "*.safetensors",
        "*.jinja",
        "tokenizer.*",
        "special_tokens_map.json",
        "chat_template.jinja",
        "generation_config.json",
        "zero_to_fp32.py",
        "*.py",
    ]

    return snapshot_download(
        repo_id=orig,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
        local_files_only=False,
    )


def load_index(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "model.safetensors.index.json"), "r") as f:
        return json.load(f)


def load_weight_map(model_dir: str) -> dict[str, str]:
    return load_index(model_dir)["weight_map"]


def invert_weight_map(weight_map: dict[str, str]) -> dict[str, list[str]]:
    files: dict[str, list[str]] = {}
    for name, fn in weight_map.items():
        files.setdefault(fn, []).append(name)
    # stable order
    for fn in files:
        files[fn].sort()
    return dict(sorted(files.items()))


def float8_scale_from_u8(scale_u8: torch.Tensor) -> torch.Tensor:
    # uint8 exponent-coded == reinterpret float8_e8m0fnu == 2**(byte-127)
    return scale_u8.view(torch.float8_e8m0fnu).to(torch.float32)


def compute_mxfp4_scale_u8_from_absmax(
    absmax: torch.Tensor,
    fp4_max: float = 6.0,
) -> torch.Tensor:
    """
    absmax: [...], float32 >= 0
    returns: uint8 exponent-coded scale. (reinterpret as float8_e8m0 -> power-of-two)
    """
    target = absmax / fp4_max
    min_scale = torch.tensor(2.0 ** (-126), device=absmax.device, dtype=torch.float32)
    target = torch.maximum(target, min_scale)

    exp = torch.round(torch.log2(target)).to(torch.int32)
    scale = torch.pow(2.0, exp.to(torch.float32))

    # no-clipping fix: if absmax/scale > fp4_max then increase scale by 2x
    exp = exp + ((absmax / scale) > fp4_max).to(torch.int32)

    exp = torch.clamp(exp, min=-126, max=127)
    return torch.clamp(exp + 127, min=1, max=254).to(torch.uint8)


def quantize_fp4_pack(
    weight: torch.Tensor,  # float32 [out, in]
    *,
    group_size: int = 32,
    fp4_max: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, in_features = weight.shape
    assert in_features % group_size == 0
    groups = in_features // group_size

    absmax = (
        weight.view(out, groups, group_size).abs().amax(dim=-1).to(torch.float32)
    )  # [out, groups]
    scale_u8 = compute_mxfp4_scale_u8_from_absmax(absmax, fp4_max=fp4_max)
    scale = float8_scale_from_u8(scale_u8)  # float32

    scaled = (weight.view(out, groups, group_size) / scale.view(out, groups, 1)).to(
        torch.bfloat16
    )
    q = FP4_E2M1_DATA.cast_to_fp4(scaled).reshape(out, in_features).contiguous()
    packed = pack_fp4_to_uint8(q).contiguous()  # [out, in/2] uint8
    return packed, scale_u8.contiguous()


def reshape_packed_to_openai_blocks(packed: torch.Tensor) -> torch.Tensor:
    # packed: [out, in/2], group_size=32 => 16 bytes per group
    out, packed_in = packed.shape
    bytes_per_group = 16
    assert packed_in % bytes_per_group == 0
    groups = packed_in // bytes_per_group
    return packed.reshape(out, groups, bytes_per_group)


def interleave_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    out = torch.empty((gate.shape[0] * 2,) + gate.shape[1:], dtype=gate.dtype)
    out[0::2] = gate
    out[1::2] = up
    return out


@dataclass
class ShardWriter:
    out_dir: str
    shard_size_bytes: int

    shard_idx: int = 0
    cur: dict[str, torch.Tensor] = None  # type: ignore
    cur_size: int = 0
    total_size: int = 0
    weight_map: dict[str, str] = None  # type: ignore
    shard_files: list[str] = None  # type: ignore

    def __post_init__(self):
        os.makedirs(self.out_dir, exist_ok=True)
        self.cur = {}
        self.weight_map = {}
        self.shard_files = []

    def _tmp_name(self, idx: int) -> str:
        return f"shard-{idx:05d}.safetensors"

    def add(self, name: str, tensor: torch.Tensor):
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()

        size = tensor.numel() * tensor.element_size()
        if self.cur and self.cur_size + size > self.shard_size_bytes:
            self.flush()

        tmp = self._tmp_name(self.shard_idx)
        self.cur[name] = tensor
        self.cur_size += size
        self.total_size += size
        self.weight_map[name] = tmp

    def flush(self):
        if not self.cur:
            return
        tmp = self._tmp_name(self.shard_idx)
        save_file(self.cur, os.path.join(self.out_dir, tmp))
        self.shard_files.append(tmp)
        self.cur = {}
        self.cur_size = 0
        self.shard_idx += 1

    def finalize(self) -> dict:
        self.flush()

        total_shards = len(self.shard_files)
        rename_map: dict[str, str] = {}
        for i, tmp in enumerate(self.shard_files):
            final = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
            os.replace(os.path.join(self.out_dir, tmp), os.path.join(self.out_dir, final))
            rename_map[tmp] = final

        # update weight_map values
        self.weight_map = {k: rename_map[v] for k, v in self.weight_map.items()}

        return {
            "metadata": {"total_size": self.total_size},
            "weight_map": self.weight_map,
        }


def copy_non_safetensors_files(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    skip = {"model.safetensors.index.json", "config.json"}
    for fn in os.listdir(src_dir):
        if fn in skip:
            continue
        if fn.endswith(".safetensors"):
            continue
        src = os.path.join(src_dir, fn)
        dst = os.path.join(dst_dir, fn)
        if os.path.isfile(src):
            shutil.copy2(src, dst)


def write_config(src_dir: str, dst_dir: str):
    with open(os.path.join(src_dir, "config.json"), "r") as f:
        cfg = json.load(f)
    cfg["quantization_config"] = {
        "modules_to_not_convert": [
            "model.layers.*.self_attn",
            "model.layers.*.mlp.router",
            "model.embed_tokens",
            "lm_head",
        ],
        "quant_method": "mxfp4",
    }
    with open(os.path.join(dst_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def convert(
    orig_dir: str,
    out_dir: str,
    *,
    group_size: int = 32,
    fp4_max: float = 6.0,
    shard_size_gb: int = 5,
    layers: Optional[list[int]] = None,
):
    os.makedirs(out_dir, exist_ok=True)

    weight_map = load_weight_map(orig_dir)
    file_map = invert_weight_map(weight_map)
    layout = detect_checkpoint_layout(weight_map)

    # If the source is already OpenAI MXFP4 (e.g. openai/gpt-oss-20b),
    # just copy it through to the output directory to avoid re-sharding
    # and to avoid accidentally dropping required tensors like *_bias.
    if layout == "openai_mxfp4":
        if os.listdir(out_dir):
            raise FileExistsError(
                f"Output directory must be empty for pass-through copy: {out_dir}"
            )
        shutil.rmtree(out_dir)
        shutil.copytree(orig_dir, out_dir)
        # ensure config has the expected quantization_config (idempotent)
        write_config(out_dir, out_dir)
        print("[INFO] Source checkpoint is already OpenAI/vLLM MXFP4; copied as-is.")
        return

    # fused_fp32: we will generate OpenAI blocks/scales and exclude fused tensors
    copy_non_safetensors_files(orig_dir, out_dir)
    write_config(orig_dir, out_dir)

    shard_writer = ShardWriter(out_dir=out_dir, shard_size_bytes=shard_size_gb * 1024**3)

    # 1) Copy non-expert tensors as-is (streamed)
    for fn, names in file_map.items():
        path = os.path.join(orig_dir, fn)
        with safe_open(path, framework="pt") as f:
            for name in names:
                # skip fused expert weights/bias; we will generate openai equivalents
                if (
                    _FUSED_GATE_UP_KEY.match(name)
                    or _FUSED_DOWN_KEY.match(name)
                    or _FUSED_GATE_UP_BIAS_KEY.match(name)
                    or _FUSED_DOWN_BIAS_KEY.match(name)
                ):
                    continue
                t = f.get_tensor(name)
                shard_writer.add(name, t)

    # 2) Generate quantized openai MoE tensors from original fused experts
    # Determine layers from weight_map (fused_fp32 layout)
    layer_ids = set()
    for name in weight_map.keys():
        m = _FUSED_GATE_UP_KEY.match(name)
        if m:
            layer_ids.add(int(m.group(1)))
    layer_list = sorted(layer_ids) if layers is None else layers

    for li in layer_list:
        gate_up_key = f"model.layers.{li}.mlp.experts.gate_up_proj"
        down_key = f"model.layers.{li}.mlp.experts.down_proj"
        gate_up_b_key = f"model.layers.{li}.mlp.experts.gate_up_proj_bias"
        down_b_key = f"model.layers.{li}.mlp.experts.down_proj_bias"

        # locate files
        gate_up_path = os.path.join(orig_dir, weight_map[gate_up_key])
        down_path = os.path.join(orig_dir, weight_map[down_key])
        gate_up_b_path = os.path.join(orig_dir, weight_map[gate_up_b_key])
        down_b_path = os.path.join(orig_dir, weight_map[down_b_key])

        with (
            safe_open(gate_up_path, framework="pt") as gate_up_file,
            safe_open(down_path, framework="pt") as down_file,
            safe_open(gate_up_b_path, framework="pt") as gate_up_b_file,
            safe_open(down_b_path, framework="pt") as down_b_file,
        ):
            gate_up_sl = gate_up_file.get_slice(gate_up_key)  # [E,H,2D]
            down_sl = down_file.get_slice(down_key)  # [E,D,H]
            gate_up_b_sl = gate_up_b_file.get_slice(gate_up_b_key)  # [E,2D]
            down_b_sl = down_b_file.get_slice(down_b_key)  # [E,H]

            num_experts = gate_up_sl.get_shape()[0]

            gate_up_blocks_list: list[torch.Tensor] = []
            gate_up_scales_list: list[torch.Tensor] = []
            down_blocks_list: list[torch.Tensor] = []
            down_scales_list: list[torch.Tensor] = []

            # biases: keep original interleaved order, but cast to bf16
            gate_up_bias = gate_up_b_sl[:].to(torch.bfloat16).contiguous()  # [E,2D]
            down_bias = down_b_sl[:].to(torch.bfloat16).contiguous()  # [E,H]

            for ei in range(num_experts):
                Wi = gate_up_sl[ei].to(torch.float32)  # [H,2D]
                Wd = down_sl[ei].to(torch.float32)  # [D,H]

                # de-interleave fused gate_up, then transpose to [out, in]
                Wg = Wi[:, 0::2].contiguous().t().contiguous()  # [D,H]
                Wu = Wi[:, 1::2].contiguous().t().contiguous()  # [D,H]

                # down_proj: [H,D]
                Wdown = Wd.t().contiguous()

                gate_packed, gate_scale_u8 = quantize_fp4_pack(
                    Wg, group_size=group_size, fp4_max=fp4_max
                )
                up_packed, up_scale_u8 = quantize_fp4_pack(
                    Wu, group_size=group_size, fp4_max=fp4_max
                )
                down_packed, down_scale_u8 = quantize_fp4_pack(
                    Wdown, group_size=group_size, fp4_max=fp4_max
                )

                gate_blocks = reshape_packed_to_openai_blocks(gate_packed)
                up_blocks = reshape_packed_to_openai_blocks(up_packed)
                down_blocks = reshape_packed_to_openai_blocks(down_packed)

                gate_up_blocks = interleave_gate_up(gate_blocks, up_blocks)  # [2D,G,16]
                gate_up_scales = interleave_gate_up(gate_scale_u8, up_scale_u8)  # [2D,G]

                gate_up_blocks_list.append(gate_up_blocks)
                gate_up_scales_list.append(gate_up_scales)
                down_blocks_list.append(down_blocks)
                down_scales_list.append(down_scale_u8)

            prefix = f"model.layers.{li}.mlp.experts"
            shard_writer.add(
                f"{prefix}.gate_up_proj_blocks",
                torch.stack(gate_up_blocks_list, dim=0),
            )
            shard_writer.add(
                f"{prefix}.gate_up_proj_scales",
                torch.stack(gate_up_scales_list, dim=0),
            )
            shard_writer.add(f"{prefix}.gate_up_proj_bias", gate_up_bias)
            shard_writer.add(
                f"{prefix}.down_proj_blocks",
                torch.stack(down_blocks_list, dim=0),
            )
            shard_writer.add(
                f"{prefix}.down_proj_scales",
                torch.stack(down_scales_list, dim=0),
            )
            shard_writer.add(f"{prefix}.down_proj_bias", down_bias)

    # 3) finalize shards + write index
    index_obj = shard_writer.finalize()
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index_obj, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--orig", required=True, help="Local directory or HF repo id")
    p.add_argument("--out", required=True, help="Output directory (*-openai)")
    p.add_argument("--cache-dir", default=None, help="HF cache dir (optional)")
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--fp4-max", type=float, default=6.0)
    p.add_argument("--shard-size-gb", type=int, default=5)
    p.add_argument(
        "--layers",
        default=None,
        help="Comma-separated layer indices to process (debug/smoke). Default: all.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    orig_dir = resolve_model_dir(args.orig, cache_dir=args.cache_dir)
    layers = (
        [int(x) for x in args.layers.split(",") if x.strip() != ""]
        if args.layers is not None
        else None
    )
    convert(
        orig_dir=orig_dir,
        out_dir=args.out,
        group_size=args.group_size,
        fp4_max=args.fp4_max,
        shard_size_gb=args.shard_size_gb,
        layers=layers,
    )
    print(f"[OK] Wrote OpenAI MXFP4 model to: {args.out}")


if __name__ == "__main__":
    main()

