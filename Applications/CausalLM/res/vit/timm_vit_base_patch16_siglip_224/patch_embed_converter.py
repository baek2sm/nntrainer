#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 SimpleConv2D Pattern Application
#
# @file patch_embed_converter.py
# @brief Weight converter for timm ViT patch_embed using SimpleConv2D pattern
#        Directly saves weights in nntrainer format following
#        Applications/LLaMA/PyTorch/weights_converter.py pattern
#
# Usage:
#     source /home/seungbaek/miniconda3/etc/profile.d/conda.sh
#     conda activate 26_2
#     cd /home/seungbaek/projects/nntrainer/Applications/CausalLM/res/vit/timm_vit_base_patch16_siglip_224
#     python patch_embed_converter.py

import argparse
import numpy as np
import safetensors.torch

def save_weight(weight, dtype, file):
    """
    @brief Save weight tensor directly to binary file

    This follows the SimpleConv2D pattern which is based on
    Applications/LLaMA/PyTorch/weights_converter.py

    For Conv2D layers:
    - PyTorch shape: [out_channels, in_channels, H, W]
    - NNTrainer expects: [out_channels, in_channels, H, W]
    - NO permute needed for Conv2D (unlike Linear layers!)

    Args:
        weight: PyTorch tensor
        dtype: numpy dtype (e.g., np.float32)
        file: open file object
    """
    # Convert to cpu first in case tensor is on GPU
    np.array(weight.cpu(), dtype=dtype).tofile(file)

def convert_patch_embed_only(model_path, output_path, dtype):
    """
    @brief Convert timm ViT patch_embed weights to nntrainer format

    Following SimpleConv2D/Applications/LLaMA pattern:
    1. Load state_dict from safetensors
    2. Access parameters by name
    3. Save in correct order using save_weight()
    """

    print(f"Loading model from: {model_path}")
    state_dict = safetensors.torch.load_file(model_path)

    print(f"\nConverting patch_embed weights to: {output_path}")
    print(f"Using dtype: {dtype}")
    print(f"Pattern: SimpleConv2D (no permute for Conv2D)")

    with open(output_path, 'wb') as f:
        # 1. Patch embedding (Conv2D) - weight first
        print("\n  Processing patch_embed.proj.weight...")
        patch_weight = state_dict['patch_embed.proj.weight']
        print(f"    Shape: {patch_weight.shape}")
        print(f"    Expected: [out_channels=768, in_channels=3, H=16, W=16]")

        # Save directly - NO permute for Conv2D!
        save_weight(patch_weight, dtype, f)
        print(f"    Saved weight (no permute needed - Conv2D format matches)")

        # 2. Save bias
        if 'patch_embed.proj.bias' in state_dict:
            print("\n  Processing patch_embed.proj.bias...")
            patch_bias = state_dict['patch_embed.proj.bias']
            print(f"    Shape: {patch_bias.shape}")

            save_weight(patch_bias, dtype, f)
            print(f"    Saved bias")
        else:
            print("\n  No bias found in model")

    # Verify file size
    import os
    file_size = os.path.getsize(output_path)
    expected_size = (768 * 3 * 16 * 16 + 768) * 4  # weight + bias, float32

    print(f"\n[Verification]")
    print(f"  File size: {file_size} bytes")
    print(f"  Expected: {expected_size} bytes")

    if file_size == expected_size:
        print(f"  ✓ File size matches!")
    else:
        print(f"  ✗ WARNING: File size mismatch!")

    print(f"\n[Complete]")
    print(f"  Weights saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Patch Embed Weight Converter')
    parser.add_argument("--input", type=str,
                       default="model.safetensors",
                       help="Input model path (safetensors)")
    parser.add_argument("--output", type=str,
                       default="nntr_vit_patch_embed_fp32.bin",
                       help="Output nntrainer weight file")
    parser.add_argument("--dtype", type=str, default="float32",
                       help="Data type (float32 or float16)")

    args = parser.parse_args()

    dtype_map = {
        'float32': np.float32,
        'float16': np.float16
    }
    dtype = dtype_map.get(args.dtype, np.float32)

    convert_patch_embed_only(args.input, args.output, dtype)
