#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 SeungBaek Hong <sb92.hong@samsung.com>
#
# @file   verify_cross_attention.py
# @brief  PyTorch reference implementation for cross-attention verification.
#         Generates weights in nntrainer binary format and prints reference
#         outputs for comparison with the nntrainer C++ test app.
#
# Usage:
#   python3 verify_cross_attention.py [--output_dir ./]
#
# This script:
#   1. Creates Q/K/V/O FC layers with deterministic weights (no bias)
#   2. Computes cross-attention: softmax(Q @ K^T / sqrt(head_dim)) @ V
#   3. Saves weights in nntrainer binary format (cross_attn_test_weights.bin)
#   4. Saves reference input/output tensors (cross_attn_test_io.bin)
#   5. Prints all intermediate values for manual verification
#
# Note: The C++ test uses model_tensor_type=FP32-FP32.

import argparse
import numpy as np
import struct
import os

# ============================================================
# Model Configuration (must match C++ test app)
# ============================================================
BATCH_SIZE = 1
D_MODEL = 64           # input/output feature dimension
NUM_HEADS_Q = 4        # number of query heads
NUM_HEADS_KV = 2       # number of key/value heads (GQA)
HEAD_DIM = 16          # head dimension (D_MODEL / NUM_HEADS_Q)
Q_SEQ_LEN = 3          # query sequence length (decoder side)
KV_SEQ_LEN = 5         # key/value sequence length (encoder side)
D_KV = NUM_HEADS_KV * HEAD_DIM  # = 32
D_Q = NUM_HEADS_Q * HEAD_DIM    # = 64

SEED = 42

assert D_MODEL == NUM_HEADS_Q * HEAD_DIM, \
    f"D_MODEL ({D_MODEL}) must equal NUM_HEADS_Q * HEAD_DIM ({NUM_HEADS_Q * HEAD_DIM})"
assert HEAD_DIM == D_KV // NUM_HEADS_KV, \
    f"head_dim mismatch: Q has {HEAD_DIM}, KV has {D_KV // NUM_HEADS_KV}"


def save_nntrainer_weight(f, weight_np, transpose=False):
    """Save a weight tensor in nntrainer binary format (row-major float32).
    
    Args:
        f: File handle to write to
        weight_np: Numpy array to save
        transpose: If True, transpose before saving (required for FC weights)
    """
    
    if transpose:        
        np.array(weight_np.T, dtype=np.float32).tofile(f)
    else:
        np.array(weight_np, dtype=np.float32).tofile(f)


# ============================================================
# Cross-Attention Reference Implementation
# ============================================================
def cross_attention_reference(Q, K, V, num_heads_q, num_heads_kv, head_dim):
    """
    Compute scaled dot-product cross-attention with GQA support.

    Q: (B, q_len, num_heads_q * head_dim)
    K: (B, kv_len, num_heads_kv * head_dim)
    V: (B, kv_len, num_heads_kv * head_dim)
    Returns: output (B, q_len, num_heads_q * head_dim), attn_weights
    """
    B, q_len, _ = Q.shape
    _, kv_len, _ = K.shape
    gqa_size = num_heads_q // num_heads_kv

    # Reshape to per-head: (B, seq, heads, dim) -> (B, heads, seq, dim)
    Q_h = Q.reshape(B, q_len, num_heads_q, head_dim).transpose(0, 2, 1, 3)
    K_h = K.reshape(B, kv_len, num_heads_kv, head_dim).transpose(0, 2, 1, 3)
    V_h = V.reshape(B, kv_len, num_heads_kv, head_dim).transpose(0, 2, 1, 3)

    # GQA: repeat KV heads
    K_exp = np.repeat(K_h, gqa_size, axis=1)
    V_exp = np.repeat(V_h, gqa_size, axis=1)

    # Scaled dot-product attention
    scale = 1.0 / np.sqrt(float(head_dim))
    scores = np.matmul(Q_h, K_exp.transpose(0, 1, 3, 2)) * scale

    # Softmax (numerically stable)
    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    attn_w = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)

    # Weighted sum of values
    out = np.matmul(attn_w, V_exp)
    out = out.transpose(0, 2, 1, 3).reshape(B, q_len, num_heads_q * head_dim)

    return out, attn_w


def main():
    parser = argparse.ArgumentParser(description="Cross-attention verification")
    parser.add_argument("--output_dir", type=str, default="./",
                        help="Directory to save output files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(SEED)

    # ============================================================
    # Generate deterministic inputs
    # ============================================================
    query_input = np.random.randn(BATCH_SIZE, Q_SEQ_LEN, D_MODEL).astype(np.float32) * 0.1
    kv_input = np.random.randn(BATCH_SIZE, KV_SEQ_LEN, D_MODEL).astype(np.float32) * 0.1

    # ============================================================
    # Generate FC weights (nntrainer format: [in, out], no bias)
    # ============================================================
    W_q = np.random.randn(D_MODEL, D_Q).astype(np.float32) * 0.1    # (64, 64)
    W_k = np.random.randn(D_MODEL, D_KV).astype(np.float32) * 0.1   # (64, 32)
    W_v = np.random.randn(D_MODEL, D_KV).astype(np.float32) * 0.1   # (64, 32)
    W_o = np.random.randn(D_Q, D_MODEL).astype(np.float32) * 0.1    # (64, 64)

    print("Weight_Q:", W_q.reshape(-1)[:3])
    print("Weight_K:", W_k.reshape(-1)[:3])
    print("Weight_V:", W_v.reshape(-1)[:3])
    print("Weight_O:", W_o.reshape(-1)[:3])

    # ============================================================
    # FP32 Forward Pass
    # ============================================================
    Q = np.matmul(query_input, W_q)    
    K = np.matmul(kv_input, W_k)
    V = np.matmul(kv_input, W_v)
    attn_output, attn_weights = cross_attention_reference(
        Q, K, V, NUM_HEADS_Q, NUM_HEADS_KV, HEAD_DIM
    )
    final_output = np.matmul(attn_output, W_o)

    # ============================================================
    # Print Results
    # ============================================================
    print("=" * 60)
    print("Cross-Attention Verification - PyTorch Reference")
    print("=" * 60)
    print(f"\nConfig:")
    print(f"  batch_size    = {BATCH_SIZE}")
    print(f"  d_model       = {D_MODEL}")
    print(f"  num_heads_Q   = {NUM_HEADS_Q}")
    print(f"  num_heads_KV  = {NUM_HEADS_KV}")
    print(f"  head_dim      = {HEAD_DIM}")
    print(f"  q_seq_len     = {Q_SEQ_LEN}")
    print(f"  kv_seq_len    = {KV_SEQ_LEN}")
    print(f"  GQA group     = {NUM_HEADS_Q // NUM_HEADS_KV}")

    print(f"\n--- Query Input (first 8 cols) ---")
    print(query_input.reshape(-1)[:3])
    print(f"\n--- KV Input (first 8 cols) ---")
    print(kv_input.reshape(-1)[:3])

    print(f"\n--- Q projection (first 8 cols) ---")
    print(Q.reshape(-1)[:3])
    print(f"\n--- K projection (first 8 cols) ---")
    print(K.reshape(-1)[:3])
    print(f"\n--- V projection (first 8 cols) ---")
    print(V.reshape(-1)[:3])

    print(f"\n--- Attention Weights per head ---")
    for h in range(NUM_HEADS_Q):
        print(f"  Head {h}: {attn_weights[0, h].reshape(-1)[:3]}")

    print(f"\n--- Attention Output (first 8 cols) ---")
    print(attn_output[0, :, :8].reshape(-1)[:3])

    print(f"\n{'=' * 60}")
    print("FINAL OUTPUT (FP32):")
    print(f"{'=' * 60}")
    for row in range(Q_SEQ_LEN):
        print(f"  Row {row}: {final_output[0, row].reshape(-1)[:3]}")

    # ============================================================
    # Save weights for nntrainer
    # ============================================================
    # nntrainer model->load() reads weights in layer add order.
    # FC layer: weight only (disable_bias=true)
    # Layer order: q_proj, k_proj, v_proj, mha_core(no weights), o_proj
    weight_path = os.path.join(args.output_dir, "cross_attn_test_weights.bin")
    with open(weight_path, "wb") as f:
        # nntrainer model->load() reads layers in topological order.
        # Topological sort of this model: kv_input -> v_proj, k_proj -> query_input -> q_proj -> o_proj
        # So the weight file order is: v_proj, k_proj, q_proj, o_proj.
        # FC weights are stored as (in_features, out_features) matching nntrainer's
        # NCHW weight layout [1,1,in_features,out_features].
        # The computation is output = input @ weight, same as numpy matmul convention.
        save_nntrainer_weight(f, W_v, transpose=False)  # v_proj weight: (D_MODEL, D_KV)
        save_nntrainer_weight(f, W_k, transpose=False)  # k_proj weight: (D_MODEL, D_KV)
        save_nntrainer_weight(f, W_q, transpose=False)  # q_proj weight: (D_MODEL, D_Q)
        save_nntrainer_weight(f, W_o, transpose=False)  # o_proj weight: (D_Q, D_MODEL)
    print(f"\nWeights saved to: {weight_path}")

    # ============================================================
    # Save reference I/O
    # ============================================================
    io_path = os.path.join(args.output_dir, "cross_attn_test_io.bin")
    with open(io_path, "wb") as f:
        # Header
        for v in [BATCH_SIZE, Q_SEQ_LEN, KV_SEQ_LEN, D_MODEL, D_Q, D_KV,
                  NUM_HEADS_Q, NUM_HEADS_KV, HEAD_DIM]:
            f.write(struct.pack('<I', v))
        # Inputs
        save_nntrainer_weight(f, query_input)
        save_nntrainer_weight(f, kv_input)
        # Reference output
        save_nntrainer_weight(f, final_output)
    print(f"I/O tensors saved to: {io_path}")
    print(f"\nDone. Compare with nntrainer C++ test output.")


if __name__ == "__main__":
    main()
