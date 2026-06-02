## SPDX-License-Identifier: Apache-2.0
## Copyright (C) 2026 SeungBaek Hong <sb92.hong@samsung.com>
##
## @file   projector_weight_converter.py
## @brief  Weight conversion for LFM2.5-VL multi_modal_projector (vision -> LLM embedding).
##
## Converts the multi_modal_projector weights from a HuggingFace
## LiquidAI/LFM2.5-VL-450M (Lfm2VlForConditionalGeneration) checkpoint to
## the nntrainer binary format expected by Lfm2VlProjector.
##
## The projector (multi_modal_projector) in LFM2.5-VL-450M:
##   Lfm2VlMultiModalProjector(
##     pixel_unshuffle(factor=2),   [B,16,16,768] -> [B,64,3072]
##     linear_1: Linear(3072 -> 2048),
##     act: GELUActivation(),
##     linear_2: Linear(2048 -> 1024)
##   )
##
## Lfm2VlProjector::constructModel creates layers in this order:
##   proj_fc1  (FC 3072->2048, with bias)
##   proj_gelu (activation, no weights)
##   proj_fc2  (FC 2048->1024, with bias)
##
## HF key mapping (LiquidAI/LFM2.5-VL-450M):
##   model.multi_modal_projector.linear_1.weight -> proj_fc1 weight (transposed)
##   model.multi_modal_projector.linear_1.bias   -> proj_fc1 bias
##   model.multi_modal_projector.linear_2.weight -> proj_fc2 weight (transposed)
##   model.multi_modal_projector.linear_2.bias   -> proj_fc2 bias

import argparse
import numpy as np


def _tensor_to_numpy(tensor):
    return tensor.detach().cpu().float().numpy()


def convert_from_model(model, outfile):
    """Convert using an already-loaded Lfm2VlForConditionalGeneration model object."""
    projector = model.model.multi_modal_projector
    weights = [
        _tensor_to_numpy(projector.linear_1.weight).T,
        _tensor_to_numpy(projector.linear_1.bias),
        _tensor_to_numpy(projector.linear_2.weight).T,
        _tensor_to_numpy(projector.linear_2.bias),
    ]
    with open(outfile, "wb") as f:
        for weight in weights:
            np.asarray(weight, dtype=np.float32).tofile(f)
    print(f"Wrote {outfile} ({sum(w.size for w in weights)} parameters)")


def convert(model_id_or_path, output_path):
    """Load LFM2.5-VL from HF hub or local path and convert projector weights."""
    try:
        import torch
        from transformers import Lfm2VlForConditionalGeneration
    except ImportError:
        raise ImportError("torch and transformers required: pip install torch transformers")

    print(f"Loading model: {model_id_or_path}")
    model = Lfm2VlForConditionalGeneration.from_pretrained(
        model_id_or_path, torch_dtype=torch.float32)
    model.eval()

    print("Projector keys:")
    sd = model.state_dict()
    for k, v in sorted(sd.items()):
        if "multi_modal_projector" in k:
            print(f"  {k}: {tuple(v.shape)}")

    print(f"\nWriting projector weights to: {output_path}")
    convert_from_model(model, output_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert LFM2.5-VL multi_modal_projector weights to nntrainer binary."
    )
    parser.add_argument("model_id_or_path",
                        help="HuggingFace model ID or local path to LFM2.5-VL model")
    parser.add_argument("output_path", help="Output .bin path")
    args = parser.parse_args()
    convert(args.model_id_or_path, args.output_path)
