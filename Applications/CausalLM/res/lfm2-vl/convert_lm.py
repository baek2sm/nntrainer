#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
##
# @package convert_lm
# @brief Convert LFM2-VL language model weights to nntrainer raw binary format.
#
# Supports the hybrid LFM2 architecture with interleaved conv and full_attention
# layers. Weight order matches transformer.cpp / lfm2_causallm.cpp graph
# execution order as determined by swiglu input ordering in createMlp():
#
# In createMlp() the swiglu layer is called as swiglu({gate, up}), so the
# graph topological sort processes ffn_gate before ffn_up (gate is swiglu's
# first listed dependency). Weight loading follows graph execution order, so
# ffn_gate (w1 = gate_proj) must appear first in the binary, followed by
# ffn_up (w3 = up_proj). The swiglu computation is silu(input[0])*input[1]
# = silu(gate)*up, which is the correct LFM2/SwiGLU formula.
#
# Attention block (createTransformerDecoderBlock + Lfm2Transformer::createAttention):
#   operator_norm, wq, q_norm, wk, k_norm, wv, attention_out, ffn_norm,
#   ffn_gate (w1), ffn_up (w3), ffn_down (w2)
#
# Conv block (createConvBlock):
#   conv_norm (operator_norm), conv_in_proj, causal_conv1d, conv_out_proj,
#   ffn_norm, ffn_gate (w1), ffn_up (w3), ffn_down (w2)
#
# causal_conv1d weight layout:
#   HF stores conv.conv.weight as [CONV_DIM, 1, KERNEL_SIZE] = [W, 1, K].
#   PyTorch causal conv convention: kernel_pos=0 is x_{t-2} (oldest),
#   kernel_pos=K-1 is x_t (current).
#
#   NNTrainer causal_conv1d_layer expects [1, 1, K, W] with:
#     w0 (offset 0*W):   applied to x_t        (current)
#     w1 (offset 1*W):   applied to x_{t-1}
#     w2 (offset 2*W):   applied to x_{t-2}    (oldest)
#
#   So NNTrainer w0 = HF kernel_pos K-1, w1 = HF kernel_pos K-2, w2 = HF kernel_pos 0.
#   Conversion: flip the K dimension, then permute [W,1,K] -> [1,K_reversed,W].
#   Python: w.flip(2).permute(1, 2, 0).contiguous()

import argparse
import os

from safetensors import safe_open
import torch


PREFIX = "model.language_model."
NUM_LAYERS = 16

# Layer types from LFM2-VL-450M config: conv/conv/attn pattern
LAYER_TYPES = [
    "conv", "conv", "full_attention", "conv", "conv",
    "full_attention", "conv", "conv", "full_attention", "conv",
    "full_attention", "conv", "full_attention", "conv",
    "full_attention", "conv",
]


def attn_layer_keys(index):
    """Weight order for a full_attention block (Lfm2Transformer::createAttention
    followed by Transformer::createMlp via createTransformerDecoderBlock).

    Each entry: (hf_key_suffix, transpose, permute, flip_dims)
    """
    base = f"layers.{index}."
    return [
        (base + "operator_norm.weight", False, None, None),
        (base + "self_attn.q_proj.weight", True, None, None),
        (base + "self_attn.q_layernorm.weight", False, None, None),
        (base + "self_attn.k_proj.weight", True, None, None),
        (base + "self_attn.k_layernorm.weight", False, None, None),
        (base + "self_attn.v_proj.weight", True, None, None),
        (base + "self_attn.out_proj.weight", True, None, None),
        (base + "ffn_norm.weight", False, None, None),
        (base + "feed_forward.w1.weight", True, None, None),   # ffn_gate first (gate_proj)
        (base + "feed_forward.w3.weight", True, None, None),   # ffn_up second (up_proj)
        (base + "feed_forward.w2.weight", True, None, None),
    ]


def conv_layer_keys(index):
    """Weight order for a conv block (Lfm2Transformer::createConvBlock).

    Graph construction order:
      conv_norm         <- rms_norm (uses operator_norm weight)
      conv_in_proj      <- FC [DIM -> 3*CONV_DIM], transposed
      causal_conv1d     <- kernel, flip K then permute [W,1,K]->[1,K_reversed,W]
      conv_out_proj     <- FC [CONV_DIM -> DIM], transposed
      ffn_norm          <- rms_norm
      ffn_gate (w1)     <- FC [DIM -> ff], transposed
      ffn_up (w3)       <- FC [DIM -> ff], transposed
      ffn_down (w2)     <- FC [ff -> DIM], transposed

    causal_conv1d weight: HF [W,1,K] -> flip(K) -> permute(1,2,0) -> [1,K_rev,W]
    This puts w0=HF_k2 (x_t weight), w1=HF_k1 (x_{t-1}), w2=HF_k0 (x_{t-2})
    at offsets 0, W, 2W respectively, matching nntrainer's k3 kernel convention.
    """
    base = f"layers.{index}."
    return [
        (base + "operator_norm.weight", False, None, None),           # conv_norm
        (base + "conv.in_proj.weight", True, None, None),             # conv_in_proj
        (base + "conv.conv.weight", False, (1, 2, 0), [2]),           # causal_conv1d
        (base + "conv.out_proj.weight", True, None, None),            # conv_out_proj
        (base + "ffn_norm.weight", False, None, None),
        (base + "feed_forward.w1.weight", True, None, None),          # ffn_gate first (gate_proj)
        (base + "feed_forward.w3.weight", True, None, None),          # ffn_up second (up_proj)
        (base + "feed_forward.w2.weight", True, None, None),
    ]


def write_tensor(handle, out_file, key, transpose=False, permute=None,
                 flip_dims=None):
    hf_key = PREFIX + key
    tensor = handle.get_tensor(hf_key).to(torch.float32)
    if flip_dims is not None:
        tensor = tensor.flip(flip_dims)
    if permute is not None:
        tensor = tensor.permute(*permute).contiguous()
    elif transpose:
        tensor = tensor.t().contiguous()
    else:
        tensor = tensor.contiguous()

    print(f"{hf_key}: {list(tensor.shape)}")
    out_file.write(tensor.numpy().tobytes())
    return tensor.numel() * tensor.element_size()


def main():
    parser = argparse.ArgumentParser(
        description="Convert LFM2-VL language model weights to nntrainer raw binary."
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
                ltype = LAYER_TYPES[index]
                if ltype == "full_attention":
                    keys = attn_layer_keys(index)
                else:
                    keys = conv_layer_keys(index)
                for key, transpose, permute, flip_dims in keys:
                    total_bytes += write_tensor(handle, out_file, key,
                                                transpose, permute, flip_dims)

            total_bytes += write_tensor(handle, out_file, "embedding_norm.weight")
            total_bytes += write_tensor(handle, out_file, "embed_tokens.weight", True)

    print(f"total bytes: {total_bytes}")


if __name__ == "__main__":
    main()
