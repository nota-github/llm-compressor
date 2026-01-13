#!/usr/bin/env python3
"""
Convert llm-compressor MXFP4 quantized model to OpenAI MXFP4 format.

This enables compatibility with llama.cpp for GGUF conversion.

Usage:
    python convert_to_openai_format.py <input_dir> <output_dir>
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


def load_tensors(input_dir: str) -> Dict[str, torch.Tensor]:
    """Load all tensors from safetensors files."""
    tensors = {}
    shard_files = sorted(Path(input_dir).glob("*.safetensors"))
    
    for shard_path in tqdm(shard_files, desc="Loading shards"):
        with safe_open(str(shard_path), framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    
    return tensors


def find_expert_layers(tensors: Dict[str, torch.Tensor]) -> Dict[int, Dict[int, Dict[str, torch.Tensor]]]:
    """
    Organize expert layers by layer index and expert index.
    
    Returns:
        {layer_idx: {expert_idx: {proj_type: tensor}}}
    """
    # Pattern: model.layers.{layer}.mlp.experts.experts.{expert}.{proj}.{param}
    pattern = re.compile(
        r"model\.layers\.(\d+)\.mlp\.experts\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight_packed|weight_scale|bias)"
    )
    
    organized = defaultdict(lambda: defaultdict(dict))
    
    for name, tensor in tensors.items():
        match = pattern.match(name)
        if match:
            layer_idx = int(match.group(1))
            expert_idx = int(match.group(2))
            proj_type = match.group(3)  # gate_proj, up_proj, down_proj
            param_type = match.group(4)  # weight_packed, weight_scale, bias
            
            key = f"{proj_type}.{param_type}"
            organized[layer_idx][expert_idx][key] = tensor
    
    return organized


def reshape_packed_to_openai(packed: torch.Tensor) -> torch.Tensor:
    """
    Reshape llm-compressor packed format to OpenAI format.
    
    llm-compressor: [out_features, in_features/2]
    OpenAI: [out_features, num_groups, 16]
    
    group_size=32, 4bit -> 16 bytes per group
    """
    out_features, packed_in = packed.shape
    bytes_per_group = 16  # group_size=32, 4bit packing
    num_groups = packed_in // bytes_per_group
    
    return packed.reshape(out_features, num_groups, bytes_per_group)


def interleave_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Interleave gate_proj and up_proj to create gate_up_proj.
    
    OpenAI GPT-OSS uses interleaved format: [gate[0], up[0], gate[1], up[1], ...]
    """
    # gate, up: [out_features, ...]
    # result: [out_features * 2, ...]
    out_features = gate.shape[0]
    result_shape = (out_features * 2,) + gate.shape[1:]
    
    result = torch.empty(result_shape, dtype=gate.dtype, device=gate.device)
    result[0::2] = gate  # Even indices: gate
    result[1::2] = up    # Odd indices: up
    
    return result


def convert_expert_layers(
    organized: Dict[int, Dict[int, Dict[str, torch.Tensor]]]
) -> Dict[str, torch.Tensor]:
    """
    Convert organized expert layers to OpenAI format.
    """
    converted = {}
    
    for layer_idx in tqdm(sorted(organized.keys()), desc="Converting layers"):
        experts = organized[layer_idx]
        num_experts = len(experts)
        
        # Collect all experts for this layer
        gate_up_blocks_list = []
        gate_up_scales_list = []
        gate_up_bias_list = []
        down_blocks_list = []
        down_scales_list = []
        down_bias_list = []
        
        for expert_idx in range(num_experts):
            expert = experts[expert_idx]
            
            # Get gate and up projections
            gate_packed = expert["gate_proj.weight_packed"]
            gate_scale = expert["gate_proj.weight_scale"]
            gate_bias = expert["gate_proj.bias"]
            
            up_packed = expert["up_proj.weight_packed"]
            up_scale = expert["up_proj.weight_scale"]
            up_bias = expert["up_proj.bias"]
            
            down_packed = expert["down_proj.weight_packed"]
            down_scale = expert["down_proj.weight_scale"]
            down_bias = expert["down_proj.bias"]
            
            # Reshape packed tensors to OpenAI format [out, groups, 16]
            gate_blocks = reshape_packed_to_openai(gate_packed)
            up_blocks = reshape_packed_to_openai(up_packed)
            down_blocks = reshape_packed_to_openai(down_packed)
            
            # Interleave gate and up
            gate_up_blocks = interleave_gate_up(gate_blocks, up_blocks)
            gate_up_scales = interleave_gate_up(gate_scale, up_scale)
            gate_up_bias = interleave_gate_up(gate_bias.unsqueeze(-1), up_bias.unsqueeze(-1)).squeeze(-1)
            
            gate_up_blocks_list.append(gate_up_blocks)
            gate_up_scales_list.append(gate_up_scales)
            gate_up_bias_list.append(gate_up_bias)
            down_blocks_list.append(down_blocks)
            down_scales_list.append(down_scale)
            down_bias_list.append(down_bias)
        
        # Stack all experts: [num_experts, ...]
        gate_up_blocks = torch.stack(gate_up_blocks_list, dim=0)
        gate_up_scales = torch.stack(gate_up_scales_list, dim=0)
        gate_up_bias = torch.stack(gate_up_bias_list, dim=0)
        down_blocks = torch.stack(down_blocks_list, dim=0)
        down_scales = torch.stack(down_scales_list, dim=0)
        down_bias = torch.stack(down_bias_list, dim=0)
        
        # Save with OpenAI naming convention
        prefix = f"model.layers.{layer_idx}.mlp.experts"
        converted[f"{prefix}.gate_up_proj_blocks"] = gate_up_blocks
        converted[f"{prefix}.gate_up_proj_scales"] = gate_up_scales
        converted[f"{prefix}.gate_up_proj_bias"] = gate_up_bias
        converted[f"{prefix}.down_proj_blocks"] = down_blocks
        converted[f"{prefix}.down_proj_scales"] = down_scales
        converted[f"{prefix}.down_proj_bias"] = down_bias
    
    return converted


def get_non_expert_tensors(tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Get tensors that are not expert weights (attention, embeddings, etc.)."""
    non_expert = {}
    expert_pattern = re.compile(r"model\.layers\.\d+\.mlp\.experts\.experts\.")
    
    for name, tensor in tensors.items():
        if not expert_pattern.match(name):
            non_expert[name] = tensor
    
    return non_expert


def convert_config(input_dir: str, output_dir: str, num_experts: int):
    """Convert quantization config to OpenAI format."""
    config_path = os.path.join(input_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    
    # Replace quantization_config with OpenAI format
    config["quantization_config"] = {
        "modules_to_not_convert": [
            "model.layers.*.self_attn",
            "model.layers.*.mlp.router",
            "model.embed_tokens",
            "lm_head"
        ],
        "quant_method": "mxfp4"
    }
    
    output_config_path = os.path.join(output_dir, "config.json")
    with open(output_config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved config to {output_config_path}")


def copy_other_files(input_dir: str, output_dir: str):
    """Copy non-safetensors files (tokenizer, etc.)."""
    import shutil
    
    # Files to skip (already generated or will be generated)
    skip_files = {
        "config.json",
        "model.safetensors.index.json",
    }
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".safetensors") or filename in skip_files:
            continue
        
        src = os.path.join(input_dir, filename)
        dst = os.path.join(output_dir, filename)
        
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"Copied {filename}")


def save_converted_model(
    converted_experts: Dict[str, torch.Tensor],
    non_expert_tensors: Dict[str, torch.Tensor],
    output_dir: str
):
    """Save converted model to safetensors format."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Merge all tensors
    all_tensors = {**non_expert_tensors, **converted_experts}
    
    # Save as single file or sharded
    total_size = sum(t.numel() * t.element_size() for t in all_tensors.values())
    
    if total_size > 5 * 1024 * 1024 * 1024:  # > 5GB, shard
        # Simple sharding by size
        shard_size = 5 * 1024 * 1024 * 1024
        shards = []  # List of (shard_dict, tensor_names)
        current_shard = {}
        current_names = []
        current_size = 0
        
        for name, tensor in tqdm(all_tensors.items(), desc="Organizing shards"):
            tensor_size = tensor.numel() * tensor.element_size()
            
            if current_size + tensor_size > shard_size and current_shard:
                # Save current shard info
                shards.append((current_shard, current_names))
                current_shard = {}
                current_names = []
                current_size = 0
            
            current_shard[name] = tensor
            current_names.append(name)
            current_size += tensor_size
        
        # Don't forget last shard
        if current_shard:
            shards.append((current_shard, current_names))
        
        # Now save all shards with correct names
        total_shards = len(shards)
        index = {"metadata": {"total_size": total_size}, "weight_map": {}}
        
        for shard_idx, (shard_tensors, tensor_names) in enumerate(shards):
            # HuggingFace uses 0-based indexing, total count is last index (total - 1)
            shard_name = f"model-{shard_idx:05d}-of-{total_shards - 1:05d}.safetensors"
            save_file(shard_tensors, os.path.join(output_dir, shard_name))
            print(f"  Saved {shard_name} ({len(shard_tensors)} tensors)")
            
            for name in tensor_names:
                index["weight_map"][name] = shard_name
        
        # Save index
        with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)
    else:
        # Save as single file
        save_file(all_tensors, os.path.join(output_dir, "model.safetensors"))
    
    print(f"Saved model to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert llm-compressor MXFP4 model to OpenAI format"
    )
    parser.add_argument("input_dir", help="Input model directory")
    parser.add_argument("output_dir", help="Output model directory")
    args = parser.parse_args()
    
    print(f"Converting {args.input_dir} -> {args.output_dir}")
    
    # Load tensors
    print("\n[1/5] Loading tensors...")
    tensors = load_tensors(args.input_dir)
    
    # Organize expert layers
    print("\n[2/5] Organizing expert layers...")
    organized = find_expert_layers(tensors)
    
    if not organized:
        print("Warning: No expert layers found! This may not be a MoE model.")
        print("Copying model as-is...")
        os.makedirs(args.output_dir, exist_ok=True)
        save_file(tensors, os.path.join(args.output_dir, "model.safetensors"))
        convert_config(args.input_dir, args.output_dir, 0)
        copy_other_files(args.input_dir, args.output_dir)
        return
    
    num_layers = len(organized)
    num_experts = len(organized[min(organized.keys())])
    print(f"Found {num_layers} layers with {num_experts} experts each")
    
    # Convert expert layers
    print("\n[3/5] Converting expert layers to OpenAI format...")
    converted_experts = convert_expert_layers(organized)
    
    # Get non-expert tensors
    print("\n[4/5] Processing non-expert tensors...")
    non_expert_tensors = get_non_expert_tensors(tensors)
    print(f"Found {len(non_expert_tensors)} non-expert tensors")
    
    # Save converted model
    print("\n[5/5] Saving converted model...")
    save_converted_model(converted_experts, non_expert_tensors, args.output_dir)
    
    # Convert config
    convert_config(args.input_dir, args.output_dir, num_experts)
    
    # Copy other files
    copy_other_files(args.input_dir, args.output_dir)
    
    print("\nConversion complete!")
    print(f"Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

