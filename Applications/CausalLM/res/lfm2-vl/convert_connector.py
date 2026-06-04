#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

import argparse
import os

import torch
from safetensors import safe_open


EXPECTED_BYTES = 41981952

TENSORS = [
    "model.multi_modal_projector.layer_norm.weight",
    "model.multi_modal_projector.layer_norm.bias",
    "model.multi_modal_projector.linear_1.weight",
    "model.multi_modal_projector.linear_1.bias",
    "model.multi_modal_projector.linear_2.weight",
    "model.multi_modal_projector.linear_2.bias",
]


def write_tensor(handle, out_file, key):
    tensor = handle.get_tensor(key).to(torch.float32).contiguous()
    print(f"{key}: {list(tensor.shape)}")
    out_file.write(tensor.numpy().tobytes())
    return tensor.numel() * tensor.element_size()


def main():
    parser = argparse.ArgumentParser(
        description="Convert multimodal projector weights to nntrainer raw binary."
    )
    parser.add_argument("--hf-model", required=True, help="Path to model.safetensors")
    parser.add_argument("--out-dir", default=".", help="Output directory")
    parser.add_argument(
        "--out-name", default="lfm2_vl_450m_connector.bin", help="Output filename"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.out_name)

    total_bytes = 0
    with safe_open(args.hf_model, framework="pt", device="cpu") as handle:
        with open(out_path, "wb") as out_file:
            for key in TENSORS:
                total_bytes += write_tensor(handle, out_file, key)

    print(f"total bytes: {total_bytes}")
    assert total_bytes == EXPECTED_BYTES, (
        f"unexpected connector size: {total_bytes} != {EXPECTED_BYTES}"
    )


if __name__ == "__main__":
    main()
