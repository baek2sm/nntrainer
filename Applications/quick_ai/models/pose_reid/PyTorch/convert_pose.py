# SPDX-License-Identifier: Apache-2.0
import os
import json
import struct
import argparse
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))

class Pack:
    """Accumulates {name: float32 ndarray} and writes one safetensors file."""
    def __init__(self):
        self.T = {}

    def add(self, name, array):
        self.T[name] = np.ascontiguousarray(array, dtype=np.float32)

    def write(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        names = list(self.T.keys())
        header, blob, offset = {}, bytearray(), 0
        for n in names:
            arr = self.T[n]
            b = arr.tobytes()
            header[n] = {
                "dtype": "F32",
                "shape": list(arr.shape),
                "data_offsets": [offset, offset + len(b)]
            }
            blob.extend(b)
            offset += len(b)
        hjson = json.dumps(header).encode("utf-8")
        hjson = hjson + b" " * ((8 - (len(hjson) % 8)) % 8)
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(hjson)))
            f.write(hjson)
            f.write(blob)
        return len(names)

def main(weights, out):
    print(f"Loading PyTorch checkpoint from: {weights}")
    state = torch.load(weights, map_location="cpu", weights_only=False)
    if "model" in state:
        sd = state["model"]
    elif "model_state_dict" in state:
        sd = state["model_state_dict"]
    else:
        sd = state

    p = Pack()

    # Helper to fuse Conv2d and BatchNorm2d
    def get_fused(conv_prefix, bn_prefix):
        w_conv = sd[f"{conv_prefix}.weight"]
        b_conv = sd[f"{conv_prefix}.bias"] if f"{conv_prefix}.bias" in sd else torch.zeros(w_conv.shape[0])
        
        gamma = sd[f"{bn_prefix}.weight"]
        beta = sd[f"{bn_prefix}.bias"]
        mean = sd[f"{bn_prefix}.running_mean"]
        var = sd[f"{bn_prefix}.running_var"]
        eps = 1e-5

        scale = gamma / torch.sqrt(var + eps)
        w_fused = w_conv * scale.reshape(-1, 1, 1, 1)
        b_fused = (b_conv - mean) * scale + beta
        return w_fused.numpy(), b_fused.numpy()

    # Helper to map standard ConvSiLU (with BN folding)
    def map_conv_silu(pt_prefix, cpp_prefix):
        w, b = get_fused(f"{pt_prefix}.conv", f"{pt_prefix}.bn")
        p.add(f"{cpp_prefix}/conv:filter", w)
        p.add(f"{cpp_prefix}/conv:bias", b.reshape(1, -1, 1, 1))

    # Helper to map ConvOnly (no BN folding)
    def map_conv_only(pt_prefix, cpp_prefix):
        w = sd[f"{pt_prefix}.weight"].numpy()
        b = sd[f"{pt_prefix}.bias"].numpy() if f"{pt_prefix}.bias" in sd else np.zeros(w.shape[0], dtype=np.float32)
        p.add(f"{cpp_prefix}/conv:filter", w)
        p.add(f"{cpp_prefix}/conv:bias", b.reshape(1, -1, 1, 1))

    # Helper to map standard Linear layer to 1x1 Conv
    def map_linear_as_conv(pt_prefix, cpp_prefix):
        w = sd[f"{pt_prefix}.weight"].numpy() # [out, in]
        b = sd[f"{pt_prefix}.bias"].numpy() if f"{pt_prefix}.bias" in sd else np.zeros(w.shape[0], dtype=np.float32)
        p.add(f"{cpp_prefix}/conv:filter", w.reshape(w.shape[0], w.shape[1], 1, 1))
        p.add(f"{cpp_prefix}/conv:bias", b.reshape(1, -1, 1, 1))

    # Helper to map ELAN block
    def map_elan(pt_prefix, cpp_prefix, n_blocks):
        map_conv_silu(f"{pt_prefix}.elan.conv1", f"{cpp_prefix}/elan/conv1")
        map_conv_silu(f"{pt_prefix}.elan.conv2", f"{cpp_prefix}/elan/conv2")
        for i in range(n_blocks):
            map_conv_silu(f"{pt_prefix}.elan.conv_blocks.{i}", f"{cpp_prefix}/elan/conv_blocks/{i}")
        map_conv_silu(f"{pt_prefix}.elan.last_conv", f"{cpp_prefix}/elan/last_conv")

    # Helper to map SPPCSPC block
    def map_sppcspc(pt_prefix, cpp_prefix):
        map_conv_silu(f"{pt_prefix}.cv1", f"{cpp_prefix}/spp/cv1")
        map_conv_silu(f"{pt_prefix}.cv2", f"{cpp_prefix}/spp/cv2")
        map_conv_silu(f"{pt_prefix}.cv3", f"{cpp_prefix}/spp/cv3")
        map_conv_silu(f"{pt_prefix}.cv4", f"{cpp_prefix}/spp/cv4")

    # Helper to map downsample block
    def map_downsample(pt_prefix, cpp_prefix):
        map_conv_silu(f"{pt_prefix}.base.conv", f"{cpp_prefix}/base/conv")

    # Helper to map upsample block
    def map_upsample(pt_prefix, cpp_prefix):
        map_conv_silu(f"{pt_prefix}.base.conv1", f"{cpp_prefix}/base/conv1")
        map_conv_silu(f"{pt_prefix}.base.conv2", f"{cpp_prefix}/base/conv2")

    # Helper to map full FeatureFPN Neck
    def map_fpn(pt_prefix, cpp_prefix):
        map_sppcspc(f"{pt_prefix}.spp", f"{cpp_prefix}/spp")
        map_upsample(f"{pt_prefix}.feature_up.0", f"{cpp_prefix}/feature_up/0")
        map_elan(f"{pt_prefix}.feature_up.0", f"{cpp_prefix}/feature_up/0", 2)
        map_upsample(f"{pt_prefix}.feature_up.1", f"{cpp_prefix}/feature_up/1")
        map_elan(f"{pt_prefix}.feature_up.1", f"{cpp_prefix}/feature_up/1", 2)
        
        map_downsample(f"{pt_prefix}.feature_down.0", f"{cpp_prefix}/feature_down/0")
        map_elan(f"{pt_prefix}.feature_down.0", f"{cpp_prefix}/feature_down/0", 2)
        map_downsample(f"{pt_prefix}.feature_down.1", f"{cpp_prefix}/feature_down/1")
        map_elan(f"{pt_prefix}.feature_down.1", f"{cpp_prefix}/feature_down/1", 2)

        map_conv_silu(f"{pt_prefix}.ends.0", f"{cpp_prefix}/ends/0")

    # === Map Backbone Blocks ===
    map_conv_silu("model.0.backbone.blocks.0.0", "backbone/blocks/0/0")
    map_conv_silu("model.0.backbone.blocks.1.base", "backbone/blocks/1/base")
    map_elan("model.0.backbone.blocks.1", "backbone/blocks/1", 2)
    map_elan("model.0.backbone.blocks.2", "backbone/blocks/2", 2)
    map_elan("model.0.backbone.blocks.3", "backbone/blocks/3", 2)
    map_elan("model.0.backbone.blocks.4", "backbone/blocks/4", 2)

    # === Map Neck FPNs ===
    map_fpn("model.0.features", "model.0.features")
    map_fpn("model.0.features_feat", "model.0.features_feat")

    # === Map Pose Head ===
    map_conv_only("model.1.final_layer", "model.1/final_layer")
    map_linear_as_conv("model.1.mlp.1", "model.1/mlp")
    map_linear_as_conv("model.1.gau.uv", "model.1/gau_proj")
    map_linear_as_conv("model.1.gau.o", "model.1/gau_out")
    map_linear_as_conv("model.1.cls_x", "model.1/cls_x")
    map_linear_as_conv("model.1.cls_y", "model.1/cls_y")

    # === Map ReID Head ===
    map_linear_as_conv("model.2.fc", "model.2/fc")

    num = p.write(out)
    print(f"Successfully converted. Wrote {num} tensors to: {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLOv7ReIDtiny PyTorch model to safetensors.")
    parser.add_argument("--weights", default="/home/seungbaek/Downloads/video/pose/pose_merged_v311.pt")
    parser.add_argument("--out", default="Applications/quick_ai/models/pose_reid/res/yoloreid.safetensors")
    args = parser.parse_args()
    main(args.weights, args.out)
