#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 SeungBaek Hong <sb92.hong@samsung.com>
#
# @file   verify_against_pytorch_mha.py
# @brief  Verify that the manual cross-attention implementation in
#         verify_cross_attention.py produces the same results as
#         PyTorch's nn.MultiheadAttention (cross-attention mode).
#
# Usage:
#   python3 verify_against_pytorch_mha.py
#
# This script uses the same seed/config as verify_cross_attention.py,
# builds equivalent PyTorch nn.Linear + nn.MultiheadAttention layers
# with the same weights, and compares outputs element-by-element.

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("ERROR: PyTorch is required. Install with: pip install torch")
    exit(1)

# ============================================================
# Configuration (identical to verify_cross_attention.py)
# ============================================================
BATCH_SIZE = 1
D_MODEL = 64
NUM_HEADS_Q = 4
NUM_HEADS_KV = 2
HEAD_DIM = 16
Q_SEQ_LEN = 3
KV_SEQ_LEN = 5
D_KV = NUM_HEADS_KV * HEAD_DIM  # 32
D_Q = NUM_HEADS_Q * HEAD_DIM    # 64
SEED = 42


# ============================================================
# Manual cross-attention (same as verify_cross_attention.py)
# ============================================================
def manual_cross_attention(Q, K, V, num_heads_q, num_heads_kv, head_dim):
    """Numpy-based cross-attention with GQA."""
    B, q_len, _ = Q.shape
    _, kv_len, _ = K.shape
    gqa_size = num_heads_q // num_heads_kv

    Q_h = Q.reshape(B, q_len, num_heads_q, head_dim).transpose(0, 2, 1, 3)
    K_h = K.reshape(B, kv_len, num_heads_kv, head_dim).transpose(0, 2, 1, 3)
    V_h = V.reshape(B, kv_len, num_heads_kv, head_dim).transpose(0, 2, 1, 3)

    K_exp = np.repeat(K_h, gqa_size, axis=1)
    V_exp = np.repeat(V_h, gqa_size, axis=1)

    scale = 1.0 / np.sqrt(float(head_dim))
    scores = np.matmul(Q_h, K_exp.transpose(0, 1, 3, 2)) * scale

    scores_max = np.max(scores, axis=-1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    attn_w = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)

    out = np.matmul(attn_w, V_exp)
    out = out.transpose(0, 2, 1, 3).reshape(B, q_len, num_heads_q * head_dim)
    return out, attn_w


# ============================================================
# PyTorch nn.MultiheadAttention cross-attention (MHA mode)
# ============================================================
def pytorch_mha_cross_attention(query_input, kv_input, W_q, W_k, W_v, W_o):
    """
    Run cross-attention through PyTorch's nn.MultiheadAttention.
    Note: nn.MultiheadAttention does NOT support GQA (num_heads_kv != num_heads_q).
    So this test uses num_heads_q == num_heads_kv == NUM_HEADS_Q with
    separate K/V projections sized to match.
    """
    # For standard MHA (no GQA), we need K/V dim == Q dim
    # nn.MultiheadAttention expects: query (seq, batch, embed_dim)
    embed_dim = D_Q  # = NUM_HEADS_Q * HEAD_DIM = 64

    mha = nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=NUM_HEADS_Q,
        dropout=0.0,
        bias=False,
        kdim=embed_dim,  # K input projected dimension
        vdim=embed_dim,  # V input projected dimension
        batch_first=False,
    )

    # When kdim == vdim == embed_dim, nn.MHA uses a single in_proj_weight
    # of shape (3*embed_dim, embed_dim) = stacked [W_q, W_k, W_v].
    # Weight shapes (PyTorch format [out, in]):
    #   in_proj_weight: (3*embed_dim, embed_dim) = (192, 64)
    #   out_proj.weight: (embed_dim, embed_dim) = (64, 64)

    with torch.no_grad():
        # nntrainer W is [in, out], PyTorch weight is [out, in]
        in_proj = torch.cat([
            torch.from_numpy(W_q.T),  # (64, 64)
            torch.from_numpy(W_k.T),  # (64, 64)
            torch.from_numpy(W_v.T),  # (64, 64)
        ], dim=0)  # (192, 64)
        mha.in_proj_weight.copy_(in_proj)
        mha.out_proj.weight.copy_(torch.from_numpy(W_o.T))

    # Convert to torch tensors: nn.MHA expects (seq_len, batch, embed_dim)
    q_t = torch.from_numpy(query_input).permute(1, 0, 2)  # (q_len, B, D)
    kv_t = torch.from_numpy(kv_input).permute(1, 0, 2)    # (kv_len, B, D)

    with torch.no_grad():
        # Cross-attention: query from decoder, key/value from encoder
        output, attn_weights = mha(q_t, kv_t, kv_t, need_weights=True)

    # Output: (q_len, B, embed_dim) -> (B, q_len, embed_dim)
    output = output.permute(1, 0, 2).numpy()
    attn_weights = attn_weights.numpy()  # (B, q_len, kv_len)

    return output, attn_weights


def pytorch_manual_cross_attention(query_input, kv_input, W_q, W_k, W_v, W_o):
    """
    Pure PyTorch implementation of cross-attention using F.scaled_dot_product_attention.
    Supports GQA (num_heads_kv != num_heads_q).
    """
    q_in = torch.from_numpy(query_input)   # (B, q_len, D_MODEL)
    kv_in = torch.from_numpy(kv_input)     # (B, kv_len, D_MODEL)
    wq = torch.from_numpy(W_q)  # (D_MODEL, D_Q)
    wk = torch.from_numpy(W_k)  # (D_MODEL, D_KV)
    wv = torch.from_numpy(W_v)  # (D_MODEL, D_KV)
    wo = torch.from_numpy(W_o)  # (D_Q, D_MODEL)

    with torch.no_grad():
        Q = q_in @ wq     # (B, q_len, D_Q)
        K = kv_in @ wk    # (B, kv_len, D_KV)
        V = kv_in @ wv    # (B, kv_len, D_KV)

        B = Q.shape[0]
        gqa_size = NUM_HEADS_Q // NUM_HEADS_KV

        # Reshape: (B, seq, heads, dim) -> (B, heads, seq, dim)
        Q_h = Q.reshape(B, Q_SEQ_LEN, NUM_HEADS_Q, HEAD_DIM).transpose(1, 2)
        K_h = K.reshape(B, KV_SEQ_LEN, NUM_HEADS_KV, HEAD_DIM).transpose(1, 2)
        V_h = V.reshape(B, KV_SEQ_LEN, NUM_HEADS_KV, HEAD_DIM).transpose(1, 2)

        # Expand KV for GQA
        K_exp = K_h.repeat_interleave(gqa_size, dim=1)  # (B, NUM_HEADS_Q, kv_len, HEAD_DIM)
        V_exp = V_h.repeat_interleave(gqa_size, dim=1)

        # F.scaled_dot_product_attention (PyTorch >= 2.0)
        attn_out = F.scaled_dot_product_attention(
            Q_h, K_exp, V_exp,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        # (B, NUM_HEADS_Q, q_len, HEAD_DIM) -> (B, q_len, D_Q)
        attn_out = attn_out.transpose(1, 2).reshape(B, Q_SEQ_LEN, D_Q)

        final_out = attn_out @ wo  # (B, q_len, D_MODEL)

    return final_out.numpy(), attn_out.numpy()


def main():
    np.random.seed(SEED)

    # Generate same inputs/weights as verify_cross_attention.py
    query_input = np.random.randn(BATCH_SIZE, Q_SEQ_LEN, D_MODEL).astype(np.float32) * 0.1
    kv_input = np.random.randn(BATCH_SIZE, KV_SEQ_LEN, D_MODEL).astype(np.float32) * 0.1
    W_q = np.random.randn(D_MODEL, D_Q).astype(np.float32) * 0.1
    W_k = np.random.randn(D_MODEL, D_KV).astype(np.float32) * 0.1
    W_v = np.random.randn(D_MODEL, D_KV).astype(np.float32) * 0.1
    W_o = np.random.randn(D_Q, D_MODEL).astype(np.float32) * 0.1

    # ============================================================
    # 1. Manual numpy cross-attention (from verify_cross_attention.py)
    # ============================================================
    Q = np.matmul(query_input, W_q)
    K = np.matmul(kv_input, W_k)
    V = np.matmul(kv_input, W_v)
    manual_out, manual_attn_w = manual_cross_attention(
        Q, K, V, NUM_HEADS_Q, NUM_HEADS_KV, HEAD_DIM
    )
    manual_final = np.matmul(manual_out, W_o)

    print("=" * 70)
    print("Cross-Attention: Manual numpy vs PyTorch Verification")
    print("=" * 70)
    print(f"\nConfig: B={BATCH_SIZE}, q={Q_SEQ_LEN}, kv={KV_SEQ_LEN}, "
          f"d={D_MODEL}, heads_Q={NUM_HEADS_Q}, heads_KV={NUM_HEADS_KV}, "
          f"head_dim={HEAD_DIM}")

    # ============================================================
    # 2. PyTorch F.scaled_dot_product_attention (supports GQA)
    # ============================================================
    print(f"\n{'─' * 70}")
    print("Test 1: Manual numpy  vs  PyTorch F.scaled_dot_product_attention")
    print(f"{'─' * 70}")

    pt_sdpa_final, pt_sdpa_attn = pytorch_manual_cross_attention(
        query_input, kv_input, W_q, W_k, W_v, W_o
    )

    attn_diff = np.max(np.abs(manual_out - pt_sdpa_attn))
    final_diff = np.max(np.abs(manual_final - pt_sdpa_final))

    print(f"\n  Attention output max diff : {attn_diff:.2e}")
    print(f"  Final output max diff     : {final_diff:.2e}")
    if final_diff < 1e-5:
        print("  => PASS")
    else:
        print("  => FAIL (diff too large)")

    print(f"\n  Manual final[0, 0, :8]  : {manual_final[0, 0, :8]}")
    print(f"  PyTorch final[0, 0, :8] : {pt_sdpa_final[0, 0, :8]}")

    # ============================================================
    # 3. PyTorch nn.MultiheadAttention (no GQA, standard MHA)
    # ============================================================
    # nn.MHA doesn't support GQA. To compare, we create a non-GQA version
    # where num_heads_kv == num_heads_q and K/V projections output D_Q.
    print(f"\n{'─' * 70}")
    print("Test 2: Manual numpy (no GQA)  vs  nn.MultiheadAttention")
    print(f"{'─' * 70}")

    # Re-generate K/V weights with output dim = D_Q (no GQA)
    np.random.seed(SEED + 100)
    W_k_full = np.random.randn(D_MODEL, D_Q).astype(np.float32) * 0.1  # (64, 64)
    W_v_full = np.random.randn(D_MODEL, D_Q).astype(np.float32) * 0.1  # (64, 64)

    Q2 = np.matmul(query_input, W_q)
    K2 = np.matmul(kv_input, W_k_full)
    V2 = np.matmul(kv_input, W_v_full)

    # Manual (no GQA: num_heads_kv == num_heads_q)
    manual_out2, _ = manual_cross_attention(
        Q2, K2, V2, NUM_HEADS_Q, NUM_HEADS_Q, HEAD_DIM
    )
    manual_final2 = np.matmul(manual_out2, W_o)

    # nn.MultiheadAttention
    pt_mha_final, pt_mha_attn_w = pytorch_mha_cross_attention(
        query_input, kv_input, W_q, W_k_full, W_v_full, W_o
    )

    attn_diff2 = np.max(np.abs(manual_out2 - pt_mha_final + np.matmul(manual_out2, W_o) - pt_mha_final))
    final_diff2 = np.max(np.abs(manual_final2 - pt_mha_final))

    print(f"\n  Final output max diff     : {final_diff2:.2e}")
    if final_diff2 < 1e-5:
        print("  => PASS")
    else:
        print("  => FAIL (diff too large)")

    print(f"\n  Manual final[0, 0, :8]     : {manual_final2[0, 0, :8]}")
    print(f"  nn.MHA final[0, 0, :8]     : {pt_mha_final[0, 0, :8]}")

    # ============================================================
    # 4. Summary
    # ============================================================
    print(f"\n{'=' * 70}")
    print("Summary:")
    print(f"{'=' * 70}")
    print(f"  Test 1 (SDPA, GQA)      : max_diff = {final_diff:.2e}  "
          f"{'PASS' if final_diff < 1e-5 else 'FAIL'}")
    print(f"  Test 2 (nn.MHA, no GQA) : max_diff = {final_diff2:.2e}  "
          f"{'PASS' if final_diff2 < 1e-5 else 'FAIL'}")
    all_pass = final_diff < 1e-5 and final_diff2 < 1e-5
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
