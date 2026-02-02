## SPDX-License-Identifier: Apache-2.0
## Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
##
## @file weight_converter.py
## @brief weight conversion script for vit model (timm version)
## @author SeungBaek Hong <sb92.hong@samsung.com>

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

def convert_timm_vit_to_nntrainer(model_path, output_path, dtype=np.float32):
    """Convert timm ViT weights to nntrainer format."""

    print(f"Loading model from: {model_path}")
    state_dict = safetensors.torch.load_file(model_path)

    print(f"Converting weights to: {output_path}")

    with open(output_path, 'wb') as f:
        # 1. Patch embedding (Conv2D)
        # PyTorch: [out_channels, in_channels, H, W] = [768, 3, 16, 16]
        print("  Processing patch embedding...")
        patch_weight = state_dict['patch_embed.proj.weight']
        save_weight(patch_weight, dtype, f, transpose=False)  # Conv2D: no transpose

        if 'patch_embed.proj.bias' in state_dict:
            save_weight(state_dict['patch_embed.proj.bias'], dtype, f)

        # 2. Position embedding (NO class token for SIGLIP)
        # Shape: [1, 196, 768] -> use as-is
        print("  Processing position embedding...")
        save_weight(state_dict['pos_embed'], dtype, f, transpose=False)

        # 3. Transformer blocks
        num_layers = 12  # vit_base
        print(f"  Processing {num_layers} transformer blocks...")
        for i in range(num_layers):
            layer_prefix = f'blocks.{i}.'

            # LayerNorm 1
            save_weight(state_dict[layer_prefix + 'norm1.weight'], dtype, f)
            save_weight(state_dict[layer_prefix + 'norm1.bias'], dtype, f)

            # Attention QKV (fused)
            # PyTorch: [3*768, 768] -> nntrainer: [768, 3*768]
            save_weight(state_dict[layer_prefix + 'attn.qkv.weight'], dtype, f, transpose=True)
            save_weight(state_dict[layer_prefix + 'attn.qkv.bias'], dtype, f)

            # Attention Output
            save_weight(state_dict[layer_prefix + 'attn.proj.weight'], dtype, f, transpose=True)
            save_weight(state_dict[layer_prefix + 'attn.proj.bias'], dtype, f)

            # LayerNorm 2
            save_weight(state_dict[layer_prefix + 'norm2.weight'], dtype, f)
            save_weight(state_dict[layer_prefix + 'norm2.bias'], dtype, f)

            # MLP FC1
            save_weight(state_dict[layer_prefix + 'mlp.fc1.weight'], dtype, f, transpose=True)
            save_weight(state_dict[layer_prefix + 'mlp.fc1.bias'], dtype, f)

            # MLP FC2
            save_weight(state_dict[layer_prefix + 'mlp.fc2.weight'], dtype, f, transpose=True)
            save_weight(state_dict[layer_prefix + 'mlp.fc2.bias'], dtype, f)

            print(f"    Layer {i+1}/{num_layers} done")

        # 4. Final normalization
        print("  Processing final normalization...")
        save_weight(state_dict['norm.weight'], dtype, f)
        save_weight(state_dict['norm.bias'], dtype, f)

        # 5. Attention Pool (for "global_pool": "map")
        if 'attn_pool.latent' in state_dict:
            print("  Processing attention pool...")
            # Learnable latent query
            save_weight(state_dict['attn_pool.latent'], dtype, f, transpose=False)

            # Query projection
            save_weight(state_dict['attn_pool.q.weight'], dtype, f, transpose=True)
            save_weight(state_dict['attn_pool.q.bias'], dtype, f)

            # Key-Value projection (fused)
            save_weight(state_dict['attn_pool.kv.weight'], dtype, f, transpose=True)
            save_weight(state_dict['attn_pool.kv.bias'], dtype, f)

            # Attention output projection
            save_weight(state_dict['attn_pool.proj.weight'], dtype, f, transpose=True)
            save_weight(state_dict['attn_pool.proj.bias'], dtype, f)

            # LayerNorm
            save_weight(state_dict['attn_pool.norm.weight'], dtype, f)
            save_weight(state_dict['attn_pool.norm.bias'], dtype, f)

            # MLP
            save_weight(state_dict['attn_pool.mlp.fc1.weight'], dtype, f, transpose=True)
            save_weight(state_dict['attn_pool.mlp.fc1.bias'], dtype, f)
            save_weight(state_dict['attn_pool.mlp.fc2.weight'], dtype, f, transpose=True)
            save_weight(state_dict['attn_pool.mlp.fc2.bias'], dtype, f)

        # 6. Head (none for embedding-only model)
        print("  Skipping head (num_classes=0)")

    print(f"\nConversion complete!")
    print(f"  Total weights saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                       default="model.safetensors",
                       help="Input model path (safetensors)")
    parser.add_argument("--output", type=str,
                       default="nntr_vit_base_patch16_siglip_224_fp32.bin",
                       help="Output nntrainer weight file")
    parser.add_argument("--dtype", type=str, default="float32",
                       help="Data type (float32 or float16)")

    args = parser.parse_args()

    dtype_map = {
        'float32': np.float32,
        'float16': np.float16
    }
    dtype = dtype_map.get(args.dtype, np.float32)

    convert_timm_vit_to_nntrainer(args.input, args.output, dtype)
