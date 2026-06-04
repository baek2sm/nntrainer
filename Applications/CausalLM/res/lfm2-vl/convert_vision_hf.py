#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

"""Convert the HF safetensors vision tower to nntrainer raw binary.

The HF safetensors path does not include the learnable positional embedding, so
this converter cannot write that tensor.
"""

import argparse
import os

import torch
from safetensors import safe_open


PREFIX = "model.vision_tower.vision_model."
NUM_LAYERS = 12


def layer_keys(index):
    base = f"encoder.layers.{index}."
    return [
        (base + "layer_norm1.weight", False),
        (base + "layer_norm1.bias", False),
        (base + "self_attn.q_proj.weight", True),
        (base + "self_attn.q_proj.bias", False),
        (base + "self_attn.k_proj.weight", True),
        (base + "self_attn.k_proj.bias", False),
        (base + "self_attn.v_proj.weight", True),
        (base + "self_attn.v_proj.bias", False),
        (base + "self_attn.out_proj.weight", True),
        (base + "self_attn.out_proj.bias", False),
        (base + "layer_norm2.weight", False),
        (base + "layer_norm2.bias", False),
        (base + "mlp.fc1.weight", True),
        (base + "mlp.fc1.bias", False),
        (base + "mlp.fc2.weight", True),
        (base + "mlp.fc2.bias", False),
    ]


def write_tensor(handle, out_file, key, transpose=False):
    hf_key = PREFIX + key
    tensor = handle.get_tensor(hf_key).to(torch.float32)
    if transpose:
        tensor = tensor.t().contiguous()
    else:
        tensor = tensor.contiguous()

    print(f"{hf_key}: {list(tensor.shape)}")
    out_file.write(tensor.numpy().tobytes())
    return tensor.numel() * tensor.element_size()


def main():
    parser = argparse.ArgumentParser(
        description="Convert vision tower weights to nntrainer raw binary."
    )
    parser.add_argument("--hf-model", required=True, help="Path to model.safetensors")
    parser.add_argument("--out-dir", default=".", help="Output directory")
    parser.add_argument(
        "--out-name", default="lfm2_vl_450m_vision.bin", help="Output filename"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.out_name)

    total_bytes = 0
    with safe_open(args.hf_model, framework="pt", device="cpu") as handle:
        with open(out_path, "wb") as out_file:
            total_bytes += write_tensor(
                handle, out_file, "embeddings.patch_embedding.weight"
            )
            total_bytes += write_tensor(handle, out_file, "embeddings.patch_embedding.bias")

            for index in range(NUM_LAYERS):
                for key, transpose in layer_keys(index):
                    total_bytes += write_tensor(handle, out_file, key, transpose)

            total_bytes += write_tensor(handle, out_file, "post_layernorm.weight")
            total_bytes += write_tensor(handle, out_file, "post_layernorm.bias")

    print(f"total bytes: {total_bytes}")


if __name__ == "__main__":
    main()
