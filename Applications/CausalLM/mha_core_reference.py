#!/usr/bin/env python3
"""PyTorch reference for the fixed mha_core self/cross attention cases."""

import math
import struct
import sys

try:
    import torch
except ImportError as exc:
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


NUM_HEADS = 2
HEAD_DIM = 2
HIDDEN_SIZE = NUM_HEADS * HEAD_DIM
TOLERANCE = 1.0e-4

CASES = (
    {
        "name": "self_attention",
        "is_cross_attention": False,
        "is_causal": True,
        "query_len": 2,
        "key_len": 2,
        "query": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "key": [0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 0.8, 0.7],
        "value": [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
        "expected": [
            0.89990234,
            1.0,
            1.09960938,
            1.20019531,
            1.13977249,
            1.24016302,
            1.36347616,
            1.46341848,
        ],
    },
    {
        "name": "cross_attention",
        "is_cross_attention": True,
        "is_causal": False,
        "query_len": 2,
        "key_len": 3,
        "query": [0.15, 0.05, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75],
        "key": [0.12, 0.22, 0.32, 0.42, 0.52, 0.62, 0.72, 0.82,
                0.92, 1.02, 1.12, 1.22],
        "value": [1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75,
                  1.85, 1.95, 2.05, 2.15],
        "expected": [
            1.46507692,
            1.56507695,
            1.69503891,
            1.79503894,
            1.52443612,
            1.62443614,
            1.75292563,
            1.85292566,
        ],
    },
)


def format_tensor(tensor):
    values = tensor.detach().cpu().reshape(-1).tolist()
    return "[" + ", ".join(f"{value:.8f}" for value in values) + "]"


def reshape_heads(tensor):
    seq_len = tensor.shape[0]
    return tensor.reshape(1, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)


def apply_rope(tensor):
    seq_len = tensor.shape[0]
    tensor = tensor.clone().reshape(seq_len, NUM_HEADS, HEAD_DIM)
    positions = torch.arange(seq_len, dtype=tensor.dtype)
    cos = torch.cos(positions).reshape(seq_len, 1, 1)
    sin = torch.sin(positions).reshape(seq_len, 1, 1)

    real = tensor[..., 0:1]
    imag = tensor[..., 1:2]
    rotated = torch.cat((real * cos - imag * sin,
                         real * sin + imag * cos), dim=-1)
    return rotated.reshape(seq_len, HIDDEN_SIZE)


def quantize_cache_tensor(tensor):
    return tensor.to(torch.float16).to(torch.float32)


def run_self_attention(query, key, value):
    query = reshape_heads(apply_rope(query))
    key = reshape_heads(quantize_cache_tensor(apply_rope(key)))
    value = reshape_heads(quantize_cache_tensor(value))

    score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(HEAD_DIM)
    mask = torch.triu(
        torch.ones(query.shape[-2], key.shape[-2], dtype=torch.bool),
        diagonal=1,
    )
    score = score.masked_fill(mask, torch.finfo(score.dtype).min)
    weight = torch.softmax(score, dim=-1)
    return torch.matmul(weight, value).transpose(1, 2).reshape(-1)


def run_cross_attention(query, key, value):
    query = reshape_heads(query)
    key = reshape_heads(key)
    value = reshape_heads(value)

    score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(HEAD_DIM)
    weight = torch.softmax(score, dim=-1)
    return torch.matmul(weight, value).transpose(1, 2).reshape(-1)


def format_values(values):
    return "[" + ", ".join(f"{value:.8f}" for value in values) + "]"


def reshape_heads_fallback(rows):
    heads = [[] for _ in range(NUM_HEADS)]
    for row in rows:
        for head in range(NUM_HEADS):
            start = head * HEAD_DIM
            heads[head].append(row[start : start + HEAD_DIM])
    return heads


def apply_rope_fallback(rows):
    rotated = []
    for pos, row in enumerate(rows):
        row_out = []
        cos = math.cos(pos)
        sin = math.sin(pos)
        for head in range(NUM_HEADS):
            start = head * HEAD_DIM
            real = row[start]
            imag = row[start + 1]
            row_out.extend(
                (real * cos - imag * sin, real * sin + imag * cos)
            )
        rotated.append(row_out)
    return rotated


def quantize_cache_tensor_fallback(rows):
    quantized = []
    for row in rows:
        quantized.append(
            [struct.unpack("e", struct.pack("e", value))[0] for value in row]
        )
    return quantized


def softmax_fallback(values):
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def matmul_score_fallback(query_rows, key_rows):
    rows = []
    for query_row in query_rows:
        row = []
        for key_row in key_rows:
            dot = sum(a * b for a, b in zip(query_row, key_row))
            row.append(dot / math.sqrt(HEAD_DIM))
        rows.append(row)
    return rows


def attention_fallback(query_rows, key_rows, value_rows, is_causal):
    query_heads = reshape_heads_fallback(query_rows)
    key_heads = reshape_heads_fallback(key_rows)
    value_heads = reshape_heads_fallback(value_rows)
    head_outputs = []

    for head in range(NUM_HEADS):
        score = matmul_score_fallback(query_heads[head], key_heads[head])
        if is_causal:
            for query_idx, row in enumerate(score):
                for key_idx in range(query_idx + 1, len(row)):
                    row[key_idx] = -float("inf")

        weights = [softmax_fallback(row) for row in score]
        outputs = []
        for row_weights in weights:
            out_row = []
            for dim in range(HEAD_DIM):
                out_row.append(
                    sum(
                        row_weights[key_idx] * value_heads[head][key_idx][dim]
                        for key_idx in range(len(value_heads[head]))
                    )
                )
            outputs.append(out_row)
        head_outputs.append(outputs)

    merged = []
    for query_idx in range(len(query_rows)):
        for head in range(NUM_HEADS):
            merged.extend(head_outputs[head][query_idx])
    return merged


def run_case_fallback(case):
    query = [
        case["query"][idx : idx + HIDDEN_SIZE]
        for idx in range(0, len(case["query"]), HIDDEN_SIZE)
    ]
    key = [
        case["key"][idx : idx + HIDDEN_SIZE]
        for idx in range(0, len(case["key"]), HIDDEN_SIZE)
    ]
    value = [
        case["value"][idx : idx + HIDDEN_SIZE]
        for idx in range(0, len(case["value"]), HIDDEN_SIZE)
    ]

    if case["is_cross_attention"]:
        return attention_fallback(query, key, value, False)

    return attention_fallback(
        apply_rope_fallback(query),
        quantize_cache_tensor_fallback(apply_rope_fallback(key)),
        quantize_cache_tensor_fallback(value),
        True,
    )


def verify_case(case):
    if torch is None:
        actual = run_case_fallback(case)
        diff = max(
            abs(actual_value - expected_value)
            for actual_value, expected_value in zip(actual, case["expected"])
        )

        print(f"[{case['name']}]")
        print(f"  expected : {format_values(case['expected'])}")
        print(f"  actual   : {format_values(actual)}")
        print(f"  max_diff : {diff:.8f}")

        if diff > TOLERANCE:
            print("  validation failed", file=sys.stderr)
            return False

        print("  validation passed")
        return True

    query = torch.tensor(case["query"], dtype=torch.float32).reshape(
        case["query_len"], HIDDEN_SIZE
    )
    key = torch.tensor(case["key"], dtype=torch.float32).reshape(
        case["key_len"], HIDDEN_SIZE
    )
    value = torch.tensor(case["value"], dtype=torch.float32).reshape(
        case["key_len"], HIDDEN_SIZE
    )
    expected = torch.tensor(case["expected"], dtype=torch.float32)

    if case["is_cross_attention"]:
        actual = run_cross_attention(query, key, value)
    else:
        actual = run_self_attention(query, key, value)

    diff = torch.max(torch.abs(actual - expected)).item()

    print(f"[{case['name']}]")
    print(f"  expected : {format_tensor(expected)}")
    print(f"  actual   : {format_tensor(actual)}")
    print(f"  max_diff : {diff:.8f}")

    if diff > TOLERANCE:
        print("  validation failed", file=sys.stderr)
        return False

    print("  validation passed")
    return True


def main():
    if torch is None:
        print(
            "PyTorch is not installed; running the built-in numeric fallback "
            f"instead ({TORCH_IMPORT_ERROR})."
        )
    else:
        torch.set_printoptions(precision=8, sci_mode=False)

    success = True

    for case in CASES:
        success &= verify_case(case)

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
