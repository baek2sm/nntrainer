#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch

def main():
    pytorch_path = "/home/seungbaek/Downloads/srcb_model/face_feature/face_feature.pt"
    nntr_decoded_path = "Applications/quick_ai/models/ResNetMona/face_feature/dump_decoded.bin"

    if not os.path.exists(pytorch_path):
        print(f"PyTorch model not found: {pytorch_path}")
        return
    if not os.path.exists(nntr_decoded_path):
        print(f"NNTrainer dumped decoded file not found: {nntr_decoded_path}")
        return

    # 1. Load PyTorch model and run on identical random input
    print("Loading PyTorch face feature model...")
    py_model = torch.jit.load(pytorch_path)
    py_model.eval()

    # Generate random input matching input.raw with seed 42
    np.random.seed(42)
    input_data = np.random.rand(1, 3, 112, 112).astype(np.float32)
    
    # Preprocess matching face_feature.py: (x - 127.5) / 128.0
    # But wait! Did NNTrainer run on raw input.raw [0,1], or was it preprocessed?
    # Since input.raw was written directly from np.random.rand [0,1], NNTrainer received [0,1]!
    # And face_feature.py preprocessed image [0,255] as: (x - 127.5) / 128.0.
    # To compare the models directly on the exact same input tensor, we must bypass
    # PyTorch's normalization and pass the identical [0,1] input directly!
    py_input = input_data

    # use_mona scalar input
    use_mona_tensor = torch.tensor(0.0, dtype=torch.float32)

    with torch.no_grad():
        py_tensor = torch.from_numpy(py_input)
        py_output = py_model(py_tensor, use_mona_tensor)
        py_feat = py_output[0].cpu().numpy()  # 256 elements

    # Normalise PyTorch features
    py_feat /= np.linalg.norm(py_feat)

    # 2. Load NNTrainer decoded output
    nntr_feat = np.fromfile(nntr_decoded_path, dtype=np.float32)

    # 3. Compare statistics
    max_diff = np.abs(py_feat - nntr_feat).max()
    mean_diff = np.abs(py_feat - nntr_feat).mean()

    print("=" * 80)
    print("      IR-50 + Mona FACE FEATURE MODEL NUMERICAL PARITY COMPARISON")
    print("=" * 80)
    print(f"Max Absolute Difference: {max_diff:.8e}")
    print(f"Mean Absolute Difference: {mean_diff:.8e}")
    print("-" * 80)
    print(f"{'Dim':^8} | {'PyTorch Value':^32} | {'NNTrainer Value':^32}")
    print("-" * 80)
    for i in range(10):
        print(f"{i:^8d} | {py_feat[i]:^32.8f} | {nntr_feat[i]:^32.8f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
