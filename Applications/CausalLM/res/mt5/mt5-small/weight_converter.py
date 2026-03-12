# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file weight_converter.py
# @brief Weight conversion script for mt5 model (decoder path)
# @author nntrainer contributors

import argparse
import numpy as np

from transformers import AutoConfig, AutoModelForSeq2SeqLM


def save_mt5_for_nntrainer(params, config, dtype, file):
    """Convert and save mt5 decoder weights in nntrainer order."""
    n_layers = getattr(config, "num_decoder_layers", config.num_layers)
    tie_word_embeddings = getattr(config, "tie_word_embeddings", True)

    def save_weight(weight):
        np.array(weight, dtype=dtype).tofile(file)

    def find_key(candidates):
        for key in candidates:
            if key in params:
                return key
        raise KeyError(f"Unable to find required weight key among: {candidates}")

    def save_projection(prefix, proj_name):
        save_weight(params[f"{prefix}{proj_name}.weight"].permute(1, 0))

    embed_key = find_key(["decoder.embed_tokens.weight", "shared.weight"])
    save_weight(params[embed_key])

    for layer_idx in range(n_layers):
        layer_prefix = f"decoder.block.{layer_idx}."
        self_attn_prefix = f"{layer_prefix}layer.0."
        save_weight(params[f"{self_attn_prefix}layer_norm.weight"])

        # Transformer::createAttention order: V -> K -> Q -> O
        save_projection(f"{self_attn_prefix}SelfAttention.", "v")
        save_projection(f"{self_attn_prefix}SelfAttention.", "k")
        save_projection(f"{self_attn_prefix}SelfAttention.", "q")
        save_projection(f"{self_attn_prefix}SelfAttention.", "o")

        ffn_norm_key = find_key(
            [
                f"{layer_prefix}layer.2.layer_norm.weight",
                f"{layer_prefix}layer.1.layer_norm.weight",
            ]
        )
        save_weight(params[ffn_norm_key])

        dense_prefix = (
            f"{layer_prefix}layer.2.DenseReluDense."
            if f"{layer_prefix}layer.2.DenseReluDense.wo.weight" in params
            else f"{layer_prefix}layer.1.DenseReluDense."
        )

        gate_key = find_key(
            [f"{dense_prefix}wi_0.weight", f"{dense_prefix}wi.weight"]
        )
        up_key = find_key(
            [f"{dense_prefix}wi_1.weight", f"{dense_prefix}wi.weight"]
        )
        down_key = find_key([f"{dense_prefix}wo.weight"])

        # MT5Transformer::createMlp order: gate -> up -> down
        save_weight(params[gate_key].permute(1, 0))
        save_weight(params[up_key].permute(1, 0))
        save_weight(params[down_key].permute(1, 0))

    save_weight(params["decoder.final_layer_norm.weight"])

    if not tie_word_embeddings:
        lm_head_key = find_key(["lm_head.weight", "shared.weight"])
        save_weight(params[lm_head_key].permute(1, 0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./mt5-small")
    parser.add_argument(
        "--output_name", type=str, default="./nntr_mt5_small_fp32.bin"
    )
    parser.add_argument("--data_type", type=str, default="float32")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_path, torch_dtype="float", trust_remote_code=True
    )
    model.eval()

    with open(args.output_name, "wb") as f_model:
        save_mt5_for_nntrainer(
            model.state_dict(), config, args.data_type, f_model
        )
