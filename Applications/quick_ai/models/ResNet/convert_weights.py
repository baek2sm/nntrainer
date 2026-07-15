#!/usr/bin/env python3
"""
@file convert_weights.py
@brief Convert IR-50 + Mona face_feature JIT (.pt) weights into a single
       nntrainer-compatible safetensors file.
"""

import os
import argparse
import numpy as np
import torch
from safetensors.numpy import save_file

class Converter:
    def __init__(self, sd):
        self.sd = sd
        self.T = {}

    def _np(self, name):
        val = self.sd[name]
        return val.cpu().numpy().astype(np.float32)

    def conv(self, pt, nn, dw=False):
        w = self._np(f"{pt}.weight")
        key = "dw" if dw else "conv"
        self.T[f"{nn}:filter"] = w

    def conv_bias(self, pt, nn, dw=False):
        w = self._np(f"{pt}.weight")
        b = self._np(f"{pt}.bias")
        key = "dw" if dw else "conv"
        self.T[f"{nn}:filter"] = w
        self.T[f"{nn}:bias"] = b.reshape(1, w.shape[0], 1, 1)

    def bn(self, pt, nn, axis=1):
        gamma = self._np(f"{pt}.weight")
        beta = self._np(f"{pt}.bias")
        mean = self._np(f"{pt}.running_mean")
        var = self._np(f"{pt}.running_var")
        
        c = gamma.shape[0]
        shape = [1, 1, 1, 1]
        shape[axis] = c
        self.T[f"{nn}:gamma"] = gamma.reshape(shape)
        self.T[f"{nn}:beta"] = beta.reshape(shape)
        self.T[f"{nn}:moving_mean"] = mean.reshape(shape)
        self.T[f"{nn}:moving_variance"] = var.reshape(shape)

    def prelu(self, pt, nn):
        alpha = self._np(f"{pt}.weight")
        self.T[f"{nn}:prelu::alpha"] = alpha.reshape(1, alpha.shape[0], 1, 1)

    def convert(self, out_path):
        # 1) Input Layer
        self.conv("input_layer.0", "input_layer/conv")
        self.bn("input_layer.1", "input_layer/bn")
        self.prelu("input_layer.2", "input_layer/prelu")

        # 2) Body Blocks (24 blocks)
        # s0: [0, 1, 2], s1: [3, 4, 5, 6], s2: [7 to 20], s3: [21, 22, 23]
        stages_channels = [64]*3 + [128]*4 + [256]*14 + [512]*3
        for i in range(24):
            pt_b = f"body.{i}"
            nn_b = f"body/{i}"
            ch = stages_channels[i]

            # res_layer
            self.bn(f"{pt_b}.res_layer.0", f"{nn_b}/bn1")
            self.conv(f"{pt_b}.res_layer.1", f"{nn_b}/conv1")
            self.prelu(f"{pt_b}.res_layer.2", f"{nn_b}/prelu")
            self.conv(f"{pt_b}.res_layer.3", f"{nn_b}/conv2")
            self.bn(f"{pt_b}.res_layer.4", f"{nn_b}/bn2")

            # shortcut_layer (blocks 3, 7, 21)
            if i in (3, 7, 21):
                self.conv(f"{pt_b}.shortcut_layer.0", f"{nn_b}/shortcut_conv")
                self.bn(f"{pt_b}.shortcut_layer.1", f"{nn_b}/shortcut_bn")

            # my_module (Mona)
            self.bn(f"{pt_b}.my_module.norm", f"{nn_b}/mona_norm")
            
            # gamma depthwise 1x1 conv
            gamma = self._np(f"{pt_b}.my_module.gamma")
            self.T[f"{nn_b}/mona_mul_gamma:filter"] = gamma.reshape(ch, 1, 1, 1)
            
            # gammax depthwise 1x1 conv
            gammax = self._np(f"{pt_b}.my_module.gammax")
            self.T[f"{nn_b}/mona_mul_gammax:filter"] = gammax.reshape(ch, 1, 1, 1)

            # project1 (with bias)
            self.conv_bias(f"{pt_b}.my_module.project1", f"{nn_b}/mona_proj1")
            self.prelu(f"{pt_b}.my_module.nonlinear", f"{nn_b}/mona_prelu")

            # adapter_conv (MonaOp)
            self.conv_bias(f"{pt_b}.my_module.adapter_conv.conv1", f"{nn_b}/mona_op_conv1", dw=True)
            self.conv_bias(f"{pt_b}.my_module.adapter_conv.conv2", f"{nn_b}/mona_op_conv2", dw=True)
            self.conv_bias(f"{pt_b}.my_module.adapter_conv.conv3", f"{nn_b}/mona_op_conv3", dw=True)

            # scale averaging by 1/3 (all 32 weights initialized to 1/3.0)
            scale_avg = np.ones((32, 1, 1, 1), dtype=np.float32) * (1.0 / 3.0)
            self.T[f"{nn_b}/mona_op_scale:filter"] = scale_avg

            # projector (with bias)
            self.conv_bias(f"{pt_b}.my_module.adapter_conv.projector", f"{nn_b}/mona_op_proj")

            # project2 (with bias)
            self.conv_bias(f"{pt_b}.my_module.project2", f"{nn_b}/mona_proj2")

        # 3) Output Layer
        self.bn("output_layer.0", "output_layer/bn2d")
        
        # fully connected weight and bias
        fc_w = self._np("output_layer.3.weight") # [256, 25088]
        fc_b = self._np("output_layer.3.bias")   # [256]
        # Transpose PyTorch [out, in] to NNTrainer [1, 1, in, out]
        self.T["output_layer/fc:weight"] = fc_w.T.reshape(1, 1, fc_w.shape[1], fc_w.shape[0])
        self.T["output_layer/fc:bias"] = fc_b.reshape(1, 1, 1, fc_b.shape[0])

        self.bn("output_layer.4", "output_layer/bn1d", axis=3)

        # Save to safetensors
        save_file(self.T, out_path)
        return len(self.T)

def main():
    ap = argparse.ArgumentParser(description="Convert FaceFeature IR-50+Mona model to Safetensors")
    ap.add_argument("--weights", required=True, help="path to face_feature.pt")
    ap.add_argument("--out", required=True, help="output safetensors path")
    args = ap.parse_args()

    print(f"Loading TorchScript model: {args.weights}")
    m = torch.jit.load(args.weights)
    sd = m.state_dict()

    p = Converter(sd)
    n = p.convert(args.out)
    print(f"Successfully wrote {n} tensors to: {args.out}")

if __name__ == "__main__":
    main()
