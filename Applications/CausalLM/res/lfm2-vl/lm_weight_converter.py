## SPDX-License-Identifier: Apache-2.0
## Copyright (C) 2026 SeungBaek Hong <sb92.hong@samsung.com>
##
## @file   lm_weight_converter.py
## @brief  Weight conversion for LFM2.5-VL-450M language_model to nntrainer binary.
##
## Reads from HuggingFace LiquidAI/LFM2.5-VL-450M (Lfm2VlForConditionalGeneration)
## and converts the language_model weights to the nntrainer binary format
## expected by Lfm2CausalLM.
##
## The layer_types list must match the HF config's
## text_config.layer_types (or model.config.text_config.layer_types).
## This script reads them from the HF model config automatically.
##
## Usage:
##   python lm_weight_converter.py <model_path> <output_prefix> [float32|float16]
##
##   Produces:
##     <output_prefix>model.bin  -- transformer weights
##     <output_prefix>embed.bin  -- embedding table
##
## The weight order matches Lfm2CausalLM::constructModel (mirrors
## res/lfm2-350m-2/convert_weights.py but reads from language_model.*
## and uses HF config for layer_types instead of hardcoding).

import sys
import numpy as np


def save_weight(t, dtype, f, transform=None):
    """Save tensor t to binary file f with optional transform."""
    import torch
    arr = t.detach().cpu().float().numpy()
    if transform == "transpose" and arr.ndim == 2:
        arr = arr.T
    elif transform == "conv_causal" and arr.ndim == 3:
        # [F, 1, K] -> [K, F] with kernel reversal
        arr = arr.squeeze(1)   # [F, K]
        arr = arr[:, ::-1]     # flip kernel axis
        arr = arr.T            # [K, F]
        arr = np.ascontiguousarray(arr)
    arr.astype(dtype).tofile(f)


def make_weight_map(layer_types):
    """Build the ordered weight map matching Lfm2CausalLM::constructModel."""
    weight_map = []
    for i, lt in enumerate(layer_types):
        if lt in ("full_attention", "attention"):
            weight_map.extend([
                (f"layers.{i}.operator_norm.weight", "none"),
                (f"layers.{i}.self_attn.q_proj.weight", "transpose"),
                (f"layers.{i}.self_attn.q_layernorm.weight", "none"),
                (f"layers.{i}.self_attn.k_proj.weight", "transpose"),
                (f"layers.{i}.self_attn.k_layernorm.weight", "none"),
                (f"layers.{i}.self_attn.v_proj.weight", "transpose"),
                (f"layers.{i}.self_attn.out_proj.weight", "transpose"),
                (f"layers.{i}.ffn_norm.weight", "none"),
                (f"layers.{i}.feed_forward.w1.weight", "transpose"),
                (f"layers.{i}.feed_forward.w3.weight", "transpose"),
                (f"layers.{i}.feed_forward.w2.weight", "transpose"),
            ])
        elif lt == "conv":
            weight_map.extend([
                (f"layers.{i}.operator_norm.weight", "none"),
                (f"layers.{i}.conv.in_proj.weight", "transpose"),
                (f"layers.{i}.conv.conv.weight", "conv_causal"),
                (f"layers.{i}.conv.out_proj.weight", "transpose"),
                (f"layers.{i}.ffn_norm.weight", "none"),
                (f"layers.{i}.feed_forward.w1.weight", "transpose"),
                (f"layers.{i}.feed_forward.w3.weight", "transpose"),
                (f"layers.{i}.feed_forward.w2.weight", "transpose"),
            ])
        else:
            raise ValueError(f"Unsupported layer type at index {i}: {lt}")
    weight_map.append(("embedding_norm.weight", "none"))
    weight_map.append(("embed_tokens.weight", "transpose"))
    return weight_map


def convert(model_path, output_prefix, dtype="float32"):
    """Convert LFM2.5-VL-450M language_model weights to nntrainer binary."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    np_dtype = np.float32 if dtype == "float32" else np.float16

    print(f"Loading config from {model_path} ...")
    cfg = AutoConfig.from_pretrained(model_path)

    # The language model config is nested under text_config for VL models
    text_cfg = getattr(cfg, "text_config", cfg)
    if hasattr(text_cfg, "layer_types"):
        layer_types = list(text_cfg.layer_types)
    elif hasattr(text_cfg, "ssm_layer_types"):
        layer_types = list(text_cfg.ssm_layer_types)
    else:
        raise AttributeError(
            "Cannot find layer_types in config. "
            "Expected text_config.layer_types for Lfm2VlForConditionalGeneration."
        )
    print(f"Layer types ({len(layer_types)}): {layer_types}")

    print(f"Loading model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32
    )

    # Extract language_model sub-module
    lm = getattr(model, "language_model", model)
    lm_model = getattr(lm, "model", lm)
    sd = lm_model.state_dict()

    weight_map = make_weight_map(layer_types)

    model_out = output_prefix + "model.bin"
    print(f"Writing transformer weights to: {model_out}")
    with open(model_out, "wb") as f:
        for hf_key, transform in weight_map:
            if hf_key not in sd:
                raise KeyError(f"Missing key: {hf_key}")
            print(f"  {hf_key} ({tuple(sd[hf_key].shape)}) transform={transform}")
            save_weight(sd[hf_key], np_dtype, f,
                        transform if transform != "none" else None)

    embed_out = output_prefix + "embed.bin"
    print(f"Writing embedding to: {embed_out}")
    with open(embed_out, "wb") as f:
        key = "embed_tokens.weight"
        if key not in sd:
            raise KeyError(f"Missing key: {key}")
        save_weight(sd[key], np_dtype, f, transform=None)

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python lm_weight_converter.py <model_path> <output_prefix> [float32|float16]")
        sys.exit(1)
    dtype = sys.argv[3] if len(sys.argv) > 3 else "float32"
    convert(sys.argv[1], sys.argv[2], dtype)
