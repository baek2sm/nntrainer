#!/usr/bin/env python3
"""
Weight converter for timm ViT patch_embed only to nntrainer format.
Converts safetensors format to nntrainer binary format (patch_embed only).

Usage:
    source /home/seungbaek/miniconda3/etc/profile.d/conda.sh
    conda activate 26_2
    cd /home/seungbaek/projects/nntrainer/Applications/CausalLM/res/vit/timm_vit_base_patch16_siglip_224
    python patch_embed_converter.py
"""

import argparse
import numpy as np
import safetensors.torch

def save_weight(weight, dtype, file, transpose=False):
    """Save weight tensor to nntrainer format.

    Args:
        weight: PyTorch tensor or numpy array
        dtype: numpy dtype (e.g., np.float32)
        file: open file object
        transpose: whether to transpose (PyTorch uses OI, nntrainer uses IO)
    """
    if isinstance(weight, np.ndarray):
        array = weight
    else:
        array = weight.detach().cpu().numpy()

    if transpose:
        # PyTorch: [out_dim, in_dim, ...] -> nntrainer: [in_dim, out_dim, ...]
        if array.ndim >= 2:
            array = array.T

    array.astype(dtype).tofile(file)

def convert_patch_embed_only(model_path, output_path, dtype=np.float32):
    """Convert timm ViT patch_embed weights to nntrainer format (patch_embed only)."""

    print(f"Loading model from: {model_path}")
    state_dict = safetensors.torch.load_file(model_path)

    print(f"Converting patch_embed weights to: {output_path}")

    with open(output_path, 'wb') as f:
        # 1. Patch embedding (Conv2D)
        # PyTorch: [out_channels, in_channels, H, W] = [768, 3, 16, 16]
        # nntrainer: also [out_channels, in_channels, H, W] = [768, 3, 16, 16] (no transpose needed)
        print("  Processing patch embedding (conv2d)...")
        patch_weight = state_dict['patch_embed.proj.weight']

        # PyTorch and nntrainer use same format: [out_channels, in_channels, H, W]
        array = patch_weight.detach().cpu().numpy()
        array.astype(dtype).tofile(f)
        print(f"    Weight shape: {array.shape} (same as PyTorch, no transpose)")

        # Save bias
        if 'patch_embed.proj.bias' in state_dict:
            patch_bias = state_dict['patch_embed.proj.bias'].detach().cpu().numpy().astype(dtype)
            patch_bias.tofile(f)
            print(f"    Bias shape: {patch_bias.shape}")
        else:
            print("    No bias found in PyTorch model")

    print(f"\nConversion complete!")
    print(f"  Patch embed weights saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
