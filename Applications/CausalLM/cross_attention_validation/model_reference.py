#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a PyTorch cross-attention reference and export validation artifacts."""

import argparse
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


BATCH_SIZE = 1
QUERY_LEN = 2
KEY_LEN = 3
INPUT_DIM = 4
NUM_HEADS = 2
HEAD_DIM = 2
HIDDEN_DIM = NUM_HEADS * HEAD_DIM
OUTPUT_DIM = 4


class CrossAttentionValidationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(INPUT_DIM, HIDDEN_DIM, bias=False)
        self.k_proj = torch.nn.Linear(INPUT_DIM, HIDDEN_DIM, bias=False)
        self.v_proj = torch.nn.Linear(INPUT_DIM, HIDDEN_DIM, bias=False)
        self.out_proj = torch.nn.Linear(HIDDEN_DIM, OUTPUT_DIM, bias=False)

        self._init_deterministic_weights()

    def _fill_weight(self, layer: torch.nn.Linear, start: float) -> None:
        count = layer.weight.numel()
        values = torch.linspace(start, start + (count - 1) * 0.01, count)
        layer.weight.data.copy_(values.view_as(layer.weight))

    def _init_deterministic_weights(self) -> None:
        self._fill_weight(self.q_proj, -0.25)
        self._fill_weight(self.k_proj, -0.05)
        self._fill_weight(self.v_proj, 0.15)
        self._fill_weight(self.out_proj, 0.35)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(BATCH_SIZE, -1, NUM_HEADS, HEAD_DIM).transpose(1, 2)

    def forward(self, decoder_input: torch.Tensor,
                encoder_input: torch.Tensor) -> torch.Tensor:
        q = self._reshape_heads(self.q_proj(decoder_input))
        k = self._reshape_heads(self.k_proj(encoder_input))
        v = self._reshape_heads(self.v_proj(encoder_input))

        score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(HEAD_DIM)
        weight = torch.softmax(score, dim=-1)
        context = torch.matmul(weight, v)

        merged = context.transpose(1, 2).contiguous().view(
            BATCH_SIZE, QUERY_LEN, HIDDEN_DIM
        )
        return self.out_proj(merged)


def save_float_tensor(path: Path, tensor: torch.Tensor) -> None:
    np.asarray(tensor.detach().cpu(), dtype=np.float32).tofile(path)


def create_inputs() -> Tuple[torch.Tensor, torch.Tensor]:
    decoder = torch.linspace(
        -0.4,
        0.4,
        BATCH_SIZE * QUERY_LEN * INPUT_DIM,
        dtype=torch.float32,
    ).view(BATCH_SIZE, QUERY_LEN, INPUT_DIM)

    encoder = torch.linspace(
        0.1,
        1.3,
        BATCH_SIZE * KEY_LEN * INPUT_DIM,
        dtype=torch.float32,
    ).view(BATCH_SIZE, KEY_LEN, INPUT_DIM)

    return decoder, encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("."),
        help="Directory for exported binary and checkpoint files.",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="torch_weights.pt",
    )
    parser.add_argument(
        "--decoder_input_name",
        type=str,
        default="decoder_input.bin",
    )
    parser.add_argument(
        "--encoder_input_name",
        type=str,
        default="encoder_input.bin",
    )
    parser.add_argument(
        "--expected_output_name",
        type=str,
        default="expected_output.bin",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model = CrossAttentionValidationModel().eval()
    decoder_input, encoder_input = create_inputs()

    with torch.no_grad():
        expected_output = model(decoder_input, encoder_input)

    checkpoint_path = output_dir / args.checkpoint_name
    decoder_path = output_dir / args.decoder_input_name
    encoder_path = output_dir / args.encoder_input_name
    expected_path = output_dir / args.expected_output_name

    torch.save(model.state_dict(), checkpoint_path)
    save_float_tensor(decoder_path, decoder_input)
    save_float_tensor(encoder_path, encoder_input)
    save_float_tensor(expected_path, expected_output)

    print(f"saved checkpoint    : {checkpoint_path}")
    print(f"saved decoder input : {decoder_path}")
    print(f"saved encoder input : {encoder_path}")
    print(f"saved expected out  : {expected_path}")

    flat_output = expected_output.reshape(-1).tolist()
    print("expected output values:")
    print("[" + ", ".join(f"{v:.8f}" for v in flat_output) + "]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
