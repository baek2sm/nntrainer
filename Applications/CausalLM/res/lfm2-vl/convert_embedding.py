#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

import argparse
import os

import torch
from safetensors import safe_open


KEY = "model.language_model.embed_tokens.weight"
EXPECTED_BYTES = 268435456


def main():
    parser = argparse.ArgumentParser(
        description="Convert token embedding weights to nntrainer raw binary."
    )
    parser.add_argument("--hf-model", required=True, help="Path to model.safetensors")
    parser.add_argument("--out-dir", default=".", help="Output directory")
    parser.add_argument(
        "--out-name", default="lfm2_vl_450m_embedding.bin", help="Output filename"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.out_name)

    with safe_open(args.hf_model, framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(KEY).to(torch.float32).contiguous()

    total_bytes = tensor.numel() * tensor.element_size()
    print(f"{KEY}: {list(tensor.shape)}")
    print(f"bytes: {total_bytes}")
    assert total_bytes == EXPECTED_BYTES, (
        f"unexpected embedding size: {total_bytes} != {EXPECTED_BYTES}"
    )

    with open(out_path, "wb") as out_file:
        out_file.write(tensor.numpy().tobytes())


if __name__ == "__main__":
    main()
