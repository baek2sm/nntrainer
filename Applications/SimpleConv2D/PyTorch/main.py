#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 SimpleConv2D Example
#
# @file main.py
# @brief Create simple Conv2D model, run inference, and save weights for NNTrainer
#       This script demonstrates the weight conversion workflow following the
#       pattern used in Applications/LLaMA/PyTorch/weights_converter.py

import os
import torch
import numpy as np


def save_conv2d_for_nntrainer(params, dtype, file):
    """
    @brief convert and save Conv2D weights as nntrainer format
    
    This function follows the same pattern as Applications/LLaMA/PyTorch/weights_converter.py:
    - Uses model.state_dict() to access parameters
    - Saves parameters in correct order
    - No permute needed for Conv2D (shape matches NNTrainer)
    
    Param order for Conv2D:
    1. conv1.weight: shape (out_channels, in_channels, kernel_h, kernel_w)
    2. conv1.bias: shape (out_channels,)
    """
    
    def save_weight(weight):
        """
        @brief Save weight tensor to binary file
        
        Note: For Conv2D layers, no permute is needed because PyTorch shape
        (out_channels, in_channels, h, w) matches NNTrainer's expected format.
        """
        # Convert to cpu first in case tensor is on GPU
        np.array(weight.cpu(), dtype=dtype).tofile(file)
    
    # Save conv layer weights - weight first
    print(f"  Saving conv1.weight: shape={params['conv1.weight'].shape}")
    save_weight(params['conv1.weight'])
    
    # Save conv layer biases
    print(f"  Saving conv1.bias: shape={params['conv1.bias'].shape}")
    save_weight(params['conv1.bias'])


if __name__ == "__main__":
    print("=" * 70)
    print("SimpleConv2D PyTorch -> NNTrainer Weight Conversion Demo")
    print("=" * 70)
    
    # Model configuration
    data_dtype = "float32"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Step 1: Create model
    print("\n[Step 1] Creating Conv2D model (matching timm_vit: 3→768, kernel=16x16, stride=16)...")
    # Note: padding='same' is only supported for stride=1 in PyTorch
    # With stride=16, kernel=16, input=224, output will be 14x14
    class SimpleConv2D(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # timm_vit: 3 in, 768 out, 16x16 kernel, stride=16
            self.conv1 = torch.nn.Conv2d(3, 768, kernel_size=16, stride=16, padding=0, bias=True)
        
        def forward(self, x):
            return self.conv1(x)
    
    model = SimpleConv2D().to(device)
    model.eval()
    print(f"Model: {model}")
    print(f"Device: {device}")
    
    # Get model parameters as state_dict
    params = model.state_dict()
    print(f"\nParameter keys: {list(params.keys())}")
    
    # Step 2: Create all-ones input
    # Use timm_vit input size: 224x224x3
    input_shape = (1, 3, 224, 224)
    input_data = torch.ones(*input_shape).to(device)
    print(f"\n[Step 2] Input shape: {input_shape}, all values = 1.0")
    
    # Step 3: Run inference
    print("\n[Step 3] Running inference...")
    with torch.no_grad():
        output = model(input_data)
    print(f"Output shape: {output.shape}")
    
    # Step 4: Print key values for comparison
    print("\n[Step 4] Key values for comparison:")
    print(f"  Input sample [0,0,0,0:5]: {input_data[0,0,0,0:5].cpu().tolist()}")
    print(f"  Output sample [0,0,0,0:5]: {output[0,0,0,0:5].cpu().tolist()}")
    
    # Step 5: Print weight matrix info
    print("\n[Step 5] Weight matrices:")
    print(f"  conv1.weight shape: {params['conv1.weight'].shape}")
    print(f"  conv1.weight sample [0,0,0,:]: {params['conv1.weight'][0,0,0,:].cpu().tolist()[:5]}")
    print(f"  conv1.bias shape: {params['conv1.bias'].shape}")
    print(f"  conv1.bias: {params['conv1.bias'].cpu().tolist()[:5]}")
    
    # Step 6: Save weights in NNTrainer format
    print("\n[Step 6] Saving weights in NNTrainer format...")
    # Save to build directory (absolute path)
    output_file = "/home/seungbaek/projects/nntrainer/build/Applications/SimpleConv2D/PyTorch/conv2d_weights.bin"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"[Info] Output file: {output_file}")
    
    with open(output_file, "wb") as f:
        save_conv2d_for_nntrainer(params, data_dtype, f)
    
    # Verify saved file size
    # Weight shape: (out_channels, in_channels, kernel_h, kernel_w) = (768, 3, 16, 16)
    # Bias shape: (out_channels,) = (768,)
    # expected_size = (768 * 3 * 16 * 16 + 768) * 4  # weight + bias, each float32 (4 bytes)
    # actual_size = os.path.getsize(output_file)
    # print(f"  Expected file size: {expected_size} bytes")
    # print(f"  Actual file size: {actual_size} bytes")
    
    # if actual_size != expected_size:
    #     print(f"  WARNING: File size mismatch!")
    
    print(f"\n[Complete] Weights saved to: {output_file}")
    # print("\n" + "=" * 70)
    # print("Copy the following values for NNTrainer comparison:")
    # print("=" * 70)
    # print(f"pytorch_input[0:5] = {input_data.flatten()[:5].cpu().tolist()}")
    # print(f"pytorch_output[0:5] = {output.flatten()[:5].cpu().tolist()}")
    # print(f"conv1_weight[0,0,0,:] = {params['conv1.weight'][0,0,0,:].cpu().tolist()}")
    # print(f"conv1_bias = {params['conv1.bias'].cpu().tolist()}")
    # print("=" * 70)
