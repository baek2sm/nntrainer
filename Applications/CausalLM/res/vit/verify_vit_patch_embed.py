#!/usr/bin/env python3
"""Verify ViT patch_embed implementation matches nntrainer.

This script loads the exact weights from nntr_vit_patch_embed_fp32.bin
and runs the same patch_embed operation as in nntrainer to verify
the implementation is correct.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms

def load_and_preprocess_image(image_path, img_size=224):
    """Load and preprocess image for ViT."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    print(f"Image loaded: {image_path}")
    print(f"Image shape: {image_tensor.shape}")  # Should be [1, 3, 224, 224]
    
    # Print first 3 and last 3 values
    flat_img = image_tensor.flatten()
    print(f"First 3 values: {flat_img[:3].tolist()}")
    print(f"Last 3 values: {flat_img[-3:].tolist()}")

    return image_tensor

def load_nntrainer_weights(weight_path):
    """Load weights from nntrainer binary file."""
    weights = np.fromfile(weight_path, dtype=np.float32)
    conv_weights = weights[:589824]
    conv_bias = weights[589824:]
    
    # Reshape conv weights to [768, 3, 16, 16]
    conv_weights_reshaped = conv_weights.reshape(768, 3, 16, 16)
    
    print(f"Conv weights shape: {conv_weights_reshaped.shape}")
    print(f"Conv bias shape: {conv_bias.shape}")
    print(f"First 10 weight values: {conv_weights[:10]}")
    print(f"First 10 bias values: {conv_bias[:10]}")
    
    return conv_weights_reshaped, conv_bias

def create_vit_patch_embed(conv_weights, conv_bias):
    """Create ViT patch_embed layer with loaded weights."""
    # Create conv2d layer (patch embedding)
    patch_size = 16
    embed_dim = 768
    in_chans = 3
    
    conv = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
                     stride=patch_size, bias=True)
    
    # Load weights
    with torch.no_grad():
        # PyTorch expects [out_channels, in_channels, H, W]
        conv.weight.copy_(torch.from_numpy(conv_weights))
        conv.bias.copy_(torch.from_numpy(conv_bias))
    
    conv.eval()
    return conv

def test_vit_patch_embed():
    """Test ViT patch_embed with nntrainer weights."""
    print("="*60)
    print("Testing ViT Patch Embed with nntrainer weights")
    print("="*60)
    
    # Load weights from nntrainer
    weight_path = "/home/seungbaek/projects/nntrainer/Applications/CausalLM/res/vit/timm_vit_base_patch16_siglip_224/nntr_vit_patch_embed_fp32.bin"
    conv_weights, conv_bias = load_nntrainer_weights(weight_path)
    
    # Load and preprocess image
    image_path = "/home/seungbaek/Downloads/test.png"
    img_tensor = load_and_preprocess_image(image_path, img_size=224)
    
    # Create patch_embed layer
    conv = create_vit_patch_embed(conv_weights, conv_bias)
    
    # Run patch embed
    with torch.no_grad():
        patch_embed_output = conv(img_tensor)
    
    print(f"\nPatch embed output shape: {patch_embed_output.shape}")  # Should be [1, 768, 14, 14]
    print(f"Patch embed output dtype: {patch_embed_output.dtype}")
    
    # Display some statistics for conv2d output
    print(f"\nPatch embed (conv2d) output statistics:")
    print(f"  Mean: {patch_embed_output.mean().item():.6f}")
    print(f"  Std: {patch_embed_output.std().item():.6f}")
    print(f"  Min: {patch_embed_output.min().item():.6f}")
    print(f"  Max: {patch_embed_output.max().item():.6f}")
    
    # Display first few values of first channel
    print(f"\nFirst 10 values of conv2d output[0, 0, 0, 0:10]:")
    print(f"  {patch_embed_output[0, 0, 0, :10].tolist()}")
    
    # # Flatten patch output: [1, 768, 14, 14] -> [1, 196, 768]
    # B, C, H, W = patch_embed_output.shape
    # patch_flattened = patch_embed_output.view(B, C, -1).transpose(1, 2)  # [1, 196, 768]
    # print(f"\nFlattened output shape: {patch_flattened.shape}")  # Should be [1, 196, 768]
    
    # # Statistics for final output
    # print(f"\nFinal output statistics:")
    # print(f"  Mean: {patch_flattened.mean().item():.6f}")
    # print(f"  Std: {patch_flattened.std().item():.6f}")
    # print(f"  Min: {patch_flattened.min().item():.6f}")
    # print(f"  Max: {patch_flattened.max().item():.6f}")
    
    # # Display first few values for verification
    # print(f"\nFirst 10 values of final_output[0, 0, :10] (first patch):")
    # print(f"  {patch_flattened[0, 0, :10].tolist()}")
    
    # print(f"\nFirst 10 values of final_output[0, -1, :10] (last patch):")
    # print(f"  {patch_flattened[0, -1, :10].tolist()}")
    
    # # Save outputs for comparison with nntrainer
    # output_dir = "/home/seungbaek/projects/nntrainer/Applications/CausalLM/res/vit/timm_vit_base_patch16_siglip_224"
    
    # # Save conv2d output for intermediate verification
    # conv2d_output_numpy = patch_embed_output.cpu().numpy().astype(np.float32)
    # conv2d_file = f"{output_dir}/conv2d_output_pytorch.npy"
    # np.save(conv2d_file, conv2d_output_numpy)
    # print(f"\nConv2D output saved to: {conv2d_file}")
    # print(f"  Shape: {conv2d_output_numpy.shape}")
    
    # # Save final output
    # final_output_numpy = patch_flattened.cpu().numpy().astype(np.float32)
    # output_file = f"{output_dir}/final_output_pytorch.npy"
    # np.save(output_file, final_output_numpy)
    # print(f"Final output saved to: {output_file}")
    # print(f"  Shape: {final_output_numpy.shape}")
    
    # # Compare with reference outputs
    # print("\n" + "="*60)
    # print("Comparing with reference outputs")
    # print("="*60)
    
    # # Load reference outputs
    # ref_conv2d = np.load(f"{output_dir}/conv2d_output_ref.npy")
    # ref_final = np.load(f"{output_dir}/final_output_ref.npy")
    
    # print(f"Reference conv2d shape: {ref_conv2d.shape}")
    # print(f"PyTorch conv2d shape: {conv2d_output_numpy.shape}")
    
    # print(f"Reference final shape: {ref_final.shape}")
    # print(f"PyTorch final shape: {final_output_numpy.shape}")
    
    # # Calculate differences
    # conv2d_diff = np.abs(ref_conv2d - conv2d_output_numpy)
    # final_diff = np.abs(ref_final - final_output_numpy)
    
    # print(f"\nConv2d difference:")
    # print(f"  Max difference: {np.max(conv2d_diff):.8f}")
    # print(f"  Mean difference: {np.mean(conv2d_diff):.8f}")
    
    # print(f"\nFinal output difference:")
    # print(f"  Max difference: {np.max(final_diff):.8f}")
    # print(f"  Mean difference: {np.mean(final_diff):.8f}")
    
    # # Check if they match (within floating point precision)
    # conv2d_match = np.allclose(ref_conv2d, conv2d_output_numpy, rtol=1e-5, atol=1e-6)
    # final_match = np.allclose(ref_final, final_output_numpy, rtol=1e-5, atol=1e-6)
    
    # print(f"\nConv2d outputs match: {conv2d_match}")
    # print(f"Final outputs match: {final_match}")
    
    # if conv2d_match and final_match:
    #     print("\nSUCCESS: PyTorch implementation matches reference!")
    # else:
    #     print("\nWARNING: Outputs don't match exactly. Check implementation.")

if __name__ == "__main__":
    test_vit_patch_embed()
