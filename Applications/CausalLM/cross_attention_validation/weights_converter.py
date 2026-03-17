#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert PyTorch cross-attention validation weights for nntrainer."""

import argparse
from pathlib import Path

import numpy as np
import torch


WEIGHT_KEYS = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "out_proj.weight",
)


def resolve_key(state_dict, key: str) -> str:
    if key in state_dict:
        return key

    prefixed_key = f"model.{key}"
    if prefixed_key in state_dict:
        return prefixed_key

    raise KeyError(f"missing weight in state_dict: {key}")


def save_weights_for_nntrainer(
    state_dict,
    output_path: Path,
    data_type: str,
) -> None:
    with open(output_path, "wb") as out_file:
        for key in WEIGHT_KEYS:
            weight_key = resolve_key(state_dict, key)
            torch_weight = state_dict[weight_key]
            nntr_weight = torch_weight.detach().cpu().numpy().transpose(1, 0)
            np.asarray(nntr_weight, dtype=data_type).tofile(out_file)
            print(
                f"saved {weight_key}: "
                f"torch={tuple(torch_weight.shape)} -> nntr={nntr_weight.shape}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_checkpoint",
        type=Path,
        default=Path("torch_weights.pt"),
        help="PyTorch checkpoint path.",
    )
    parser.add_argument(
        "--output_weight",
        type=Path,
        default=Path("nntr_weights.bin"),
        help="Output nntrainer weight path.",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="float32",
        choices=("float32", "float16"),
    )
    return parser.parse_args()


def load_state_dict(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("unsupported checkpoint format")


def main() -> int:
    args = parse_args()
    state_dict = load_state_dict(args.input_checkpoint)

    save_weights_for_nntrainer(state_dict, args.output_weight, args.data_type)

    print(f"saved nntrainer weights: {args.output_weight}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
