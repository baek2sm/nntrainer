#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
##
# @package convert_lm
# @brief Convert LFM2-VL language model weights to nntrainer raw binary format.

import argparse
import os

from safetensors import safe_open
import torch


PREFIX = "model.language_model."
NUM_LAYERS = 16


def layer_keys(index):
    base = f"layers.{index}."
    return [
        (base + "operator_norm.weight", False),
        (base + "self_attn.q_proj.weight", True),
        (base + "self_attn.q_layernorm.weight", False),
        (base + "self_attn.k_proj.weight", True),
        (base + "self_attn.k_layernorm.weight", False),
        (base + "self_attn.v_proj.weight", True),
        (base + "self_attn.out_proj.weight", True),
        (base + "ffn_norm.weight", False),
        (base + "feed_forward.w1.weight", True),
        (base + "feed_forward.w3.weight", True),
        (base + "feed_forward.w2.weight", True),
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
        description="Convert language model weights to nntrainer raw binary."
    )
    parser.add_argument("--hf-model", required=True, help="Path to model.safetensors")
    parser.add_argument("--out-dir", default=".", help="Output directory")
    parser.add_argument("--out-name", default="lfm2_vl_450m_lm.bin", help="Output filename")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.out_name)

    total_bytes = 0
    with safe_open(args.hf_model, framework="pt", device="cpu") as handle:
        with open(out_path, "wb") as out_file:
            for index in range(NUM_LAYERS):
                for key, transpose in layer_keys(index):
                    total_bytes += write_tensor(handle, out_file, key, transpose)

            total_bytes += write_tensor(handle, out_file, "embedding_norm.weight")
            total_bytes += write_tensor(handle, out_file, "embed_tokens.weight", True)

    print(f"total bytes: {total_bytes}")


if __name__ == "__main__":
    main()
