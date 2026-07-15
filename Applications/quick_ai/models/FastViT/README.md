# FastViT — nntrainer Inference Port (backbone)

Port of the **FastViT-S12 backbone** (general feature extractor) from
[deep-vision-models](https://github.sec.samsung.net/RS8-ARGlass-SW/deep-vision-models)
(branch `dev/keyword/train-inference`) to nntrainer.

This is the **general/open** part of the model — the FastViT-S12 feature
extractor shared across tasks (it is also the backbone of the
project-specific `FastViTKeyword` classifier app). This app runs the backbone
alone and emits its output feature map.

## Model Architecture

**FastViT-S12 backbone** (timm `fastvit_sa12.apple_dist_in1k`, fused/reparameterized inference form):

- 3-layer stem (Conv+GELU)
- Stage 0: 2× RepMixerBlock (64ch, 80×80)
- Stage 1: downsample + 2× RepMixerBlock (128ch, 40×40)
- Stage 2: downsample + 6× RepMixerBlock (256ch, 20×20)
- Stage 3: downsample + pos_emb + 2× AttentionBlock (512ch, 10×10, 16 heads, head_dim=32)
- Final conv: dw 3×3 (512→1024) + SE(1024, rd=64) + GELU
- Output: `[1, 1024, 10, 10]` feature map

All RepConv/BN pairs are **fused** (reparameterized) at inference time, so the
nntrainer graph uses single biased convolutions. Layer_scale gamma is folded
into conv weights; the pre-qkv BN is folded into the qkv conv (no `attn_norm`
layer).

## Files

```
FastViT/
├── jni/
│   ├── fastvit_backbone_graph.h     # Graph block builders (stem, stages, final_conv)
│   ├── fastvit_attention_layer.h    # Custom multi-head attention layer (header)
│   ├── fastvit_attention_layer.cpp  # Custom multi-head attention layer (impl)
│   ├── main.cpp                     # Model build + inference + verification
│   └── meson.build                  # Build definition
├── PyTorch/
│   ├── extract_reference.py         # Extract PyTorch reference outputs (.bin)
│   └── convert_weights.py           # Convert .pth → nntrainer safetensors (backbone only)
├── res/                             # Weights + reference bins (generated, untracked)
├── run_nntrainer.sh                 # Run script
└── README.md
```

## Build

```bash
# From nntrainer repo root
meson setup build -Denable-transformer=true -Denable-app=true
ninja -C build
```

This produces `build/fastvit_backbone_infer`.

## Weight Conversion & Reference Extraction

```bash
# Set path to deep-vision-models repo
export DEEP_VISION_MODELS_PATH=/home/seungbaek/projects/deep-vision-models

# Convert backbone weights (.pth → safetensors)
python PyTorch/convert_weights.py --weights /path/to/ckpt.pth

# Extract reference outputs (for verification)
python PyTorch/extract_reference.py --weights /path/to/ckpt.pth
```

The converter builds the full `FastViTKeyword` model (which wraps the standalone
FastViT backbone), fuses it, and exports **only** the backbone
(`_fastViT.model.*`) tensors — the project-specific head weights are not
exported. The extraction script runs the backbone via `forward_intermediates`
and saves input + per-stage features + the final `[1,1024,10,10]` output.

## Run

```bash
# Basic inference
./run_nntrainer.sh

# With verification against PyTorch references
FASTVIT_VERIFY=1 ./run_nntrainer.sh

# Custom input
./run_nntrainer.sh /path/to/input.bin
```

## Verification

When `FASTVIT_VERIFY=1` is set, the nntrainer backbone output is compared
against the PyTorch reference:

- `ref_backbone_out.bin` — `[1,1024,10,10]` final feature map

The `max_abs_diff` is printed. For FP32 inference, the difference should be
< 1e-4 (floating-point accumulation order differences).

## Architecture Details

### RepMixerBlock (fused)

```
x = dw_conv3x3(x)                          # token mixer (no act)
mlp_out = dw_conv7x7(x) + BN               # conv path (BN folded)
mlp_out = conv1x1(C, 4C) + GELU            # fc1
mlp_out = conv1x1(4C, C)                   # fc2 (layer_scale folded)
x = x + mlp_out                            # residual
```

### AttentionBlock (fused, stage 3)

```
qkv = conv1x1(512, 1536, x)                # pre-qkv BN folded into qkv
attn = fastvit_attention(qkv)              # 16-head attention (custom layer)
proj = conv1x1(512, 512, attn)             # output projection (layer_scale_1 folded)
x = x + proj                               # residual 1
mlp_out = dw_conv7x7(x) + BN
mlp_out = conv1x1(512, 2048) + GELU
mlp_out = conv1x1(2048, 512)               # (layer_scale_2 folded)
x = x + mlp_out                            # residual 2
```

### SE Module (final conv)

```
se = global_avg_pool(x)                    # [B, C, 1, 1]
se = conv1x1(C, C/16) + ReLU               # fc1
se = conv1x1(C/16, C) + Sigmoid            # fc2
out = x * se                               # channel reweighting
```
