## SPDX-License-Identifier: Apache-2.0
## Copyright (C) 2026 SeungBaek Hong <sb92.hong@samsung.com>
##
## @file   projector_weight_converter.py
## @brief  Weight conversion for LFM2.5-VL multi_modal_projector (vision -> LLM embedding).
##
## Converts the multi_modal_projector weights from a HuggingFace
## LiquidAI/LFM2.5-VL-450M (Lfm2VlForConditionalGeneration) checkpoint to
## the nntrainer binary format expected by VjepaProjector.
##
## The projector Sequential (multi_modal_projector) in LFM2.5-VL-450M:
##   0: LayerNorm(3072)
##   1: Linear(3072 -> 3072)
##   2: GELU
##   3: Linear(3072 -> 1536)
##   4: GELU
##   5: LayerNorm(1536)
##   6: Linear(1536 -> 1024)
##   7: LayerNorm(1024)
##
## VjepaProjector::constructModel creates layers in this same order:
##   merger_ln1  (LayerNorm 3072)
##   merger_fc1  (FC 3072->3072)
##   merger_gelu1 (no weights)
##   merger_fc2  (FC 3072->1536)
##   merger_gelu2 (no weights)
##   merger_ln2  (LayerNorm 1536)
##   merger_fc3  (FC 1536->1024)
##   merger_ln3  (LayerNorm 1024)
##
## HF key mapping (LiquidAI/LFM2.5-VL-450M):
##   multi_modal_projector.0.weight / .bias   -> merger_ln1
##   multi_modal_projector.1.weight / .bias   -> merger_fc1
##   multi_modal_projector.3.weight / .bias   -> merger_fc2
##   multi_modal_projector.5.weight / .bias   -> merger_ln2
##   multi_modal_projector.6.weight / .bias   -> merger_fc3
##   multi_modal_projector.7.weight / .bias   -> merger_ln3

import argparse
import numpy as np

try:
    from safetensors.torch import load_file
except ImportError:
    load_file = None


def save_weight(weight, dtype, file, transpose=False):
    """Save a tensor to nntrainer format (optionally transposing OI -> IO)."""
    if isinstance(weight, np.ndarray):
        array = weight
    else:
        array = weight.detach().cpu().float().numpy()
    if transpose and array.ndim >= 2:
        array = array.T
    array.astype(dtype).tofile(file)


def convert(model_path, output_path, dtype="float32"):
    """Convert LFM2.5-VL multi_modal_projector weights to nntrainer binary."""
    np_dtype = np.float32 if dtype == "float32" else np.float16

    # Load from safetensors or pytorch checkpoint
    if model_path.endswith(".safetensors"):
        if load_file is None:
            raise ImportError("safetensors package required: pip install safetensors")
        sd = load_file(model_path)
    else:
        try:
            import torch
            sd = torch.load(model_path, map_location="cpu")
            if "state_dict" in sd:
                sd = sd["state_dict"]
        except ImportError:
            raise ImportError("torch package required: pip install torch")

    # Filter to multi_modal_projector keys
    proj_keys = {k: v for k, v in sd.items() if "multi_modal_projector" in k}
    if not proj_keys:
        print("Available keys (first 30):")
        for i, k in enumerate(sorted(sd.keys())):
            if i >= 30:
                break
            print(f"  {k}")
        raise RuntimeError("No multi_modal_projector keys found.")

    print("Found multi_modal_projector keys:")
    for k in sorted(proj_keys.keys()):
        print(f"  {k} {tuple(proj_keys[k].shape)}")

    def get(idx, suffix):
        """Retrieve multi_modal_projector.{idx}.{suffix}."""
        key = f"multi_modal_projector.{idx}.{suffix}"
        if key not in sd:
            raise KeyError(f"Missing key: {key}")
        return sd[key]

    print(f"\nWriting projector weights to: {output_path}")
    with open(output_path, "wb") as f:
        # merger_ln1: LayerNorm(3072) -- index 0
        save_weight(get(0, "weight"), np_dtype, f)
        save_weight(get(0, "bias"),   np_dtype, f)
        # merger_fc1: Linear(3072->3072) -- index 1
        save_weight(get(1, "weight"), np_dtype, f, transpose=True)
        save_weight(get(1, "bias"),   np_dtype, f)
        # merger_fc2: Linear(3072->1536) -- index 3
        save_weight(get(3, "weight"), np_dtype, f, transpose=True)
        save_weight(get(3, "bias"),   np_dtype, f)
        # merger_ln2: LayerNorm(1536) -- index 5
        save_weight(get(5, "weight"), np_dtype, f)
        save_weight(get(5, "bias"),   np_dtype, f)
        # merger_fc3: Linear(1536->1024) -- index 6
        save_weight(get(6, "weight"), np_dtype, f, transpose=True)
        save_weight(get(6, "bias"),   np_dtype, f)
        # merger_ln3: LayerNorm(1024) -- index 7
        save_weight(get(7, "weight"), np_dtype, f)
        save_weight(get(7, "bias"),   np_dtype, f)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert LFM2.5-VL multi_modal_projector weights to nntrainer binary."
    )
    parser.add_argument("model_path", help="Path to .safetensors or pytorch checkpoint")
    parser.add_argument("output_path", help="Output .bin path")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    args = parser.parse_args()
    convert(args.model_path, args.output_path, args.dtype)
