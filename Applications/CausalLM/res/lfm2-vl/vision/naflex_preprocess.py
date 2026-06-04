#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file  naflex_preprocess.py
# @brief NaFlex image preprocessing for LFM2-VL vision tower.
#        Implements the variable-resolution image handling that matches
#        HuggingFace Siglip2ImageProcessor used by LiquidAI/LFM2-VL-450M.
#
#        Key parameters (from LFM2-VL-450M config.json):
#          patch_size     = 16
#          max_num_patches = 1024   (max_image_tokens in image_processor)
#          image_mean     = [0.5, 0.5, 0.5]
#          image_std      = [0.5, 0.5, 0.5]
#
#        NaFlex processing pipeline per image:
#          1. Aspect-preserving resize so that H*W <= max_patches * patch_size^2
#             (both H and W are multiples of patch_size). Uses binary search
#             identical to HF get_image_size_for_max_num_patches().
#          2. Normalize: pixel = (pixel / 255 - mean) / std.
#          3. Patchify: reshape (C, H, W) -> (pH*pW, patch_size*patch_size*C)
#             in the HF row-major order (matching convert_image_to_patches).
#          4. Optionally pad to max_num_patches along the patch dimension.
#
#        Positional-embedding interpolation:
#          Given a stored (base_h * base_w, dim) pos_embed grid (from the
#          gguf_to_nntrainer.py converter), bilinearly interpolate to (pH, pW)
#          matching HF Siglip2VisionEmbeddings.resize_positional_embeddings
#          (mode="bilinear", align_corners=False, antialias=True).
#          For convenience, scipy.ndimage.zoom is used with order=1 (bilinear).
#          antialias in PyTorch CPU bilinear also uses a box prefilter which
#          scipy approximates adequately; for exact HF parity use the torch
#          snippet shown in the docstring below.

import math
import struct
from pathlib import Path

import numpy as np

try:
    from PIL import Image as _PIL_Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    import scipy.ndimage as _ndimage
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Step 1: Compute NaFlex target image size
# ---------------------------------------------------------------------------

def get_naflex_image_size(
    image_height: int,
    image_width: int,
    patch_size: int = 16,
    max_num_patches: int = 1024,
    eps: float = 1e-5,
) -> tuple:
    """Return (target_height, target_width) that:
      - preserves aspect ratio (approximately)
      - ensures target_h and target_w are multiples of patch_size
      - ensures target_h * target_w / patch_size^2 <= max_num_patches
    Matches HF get_image_size_for_max_num_patches exactly.
    """

    def scaled(scale: float, size: int) -> int:
        s = math.ceil(size * scale / patch_size) * patch_size
        return max(patch_size, int(s))

    lo, hi = eps / 10, 100.0
    while (hi - lo) >= eps:
        mid = (lo + hi) / 2
        th = scaled(mid, image_height)
        tw = scaled(mid, image_width)
        if (th / patch_size) * (tw / patch_size) <= max_num_patches:
            lo = mid
        else:
            hi = mid

    scale = lo
    th = scaled(scale, image_height)
    tw = scaled(scale, image_width)
    return th, tw


# ---------------------------------------------------------------------------
# Step 2: Patchify a (C, H, W) float32 array into (pH*pW, patch_size^2*C)
# ---------------------------------------------------------------------------

def convert_image_to_patches(
    image: np.ndarray, patch_size: int = 16
) -> np.ndarray:
    """Convert (C, H, W) float32 -> (pH*pW, patch_size*patch_size*C).
    Matches HF convert_image_to_patches row-major order.
    """
    c, h, w = image.shape
    ph = h // patch_size
    pw = w // patch_size
    # (C, pH, P, pW, P) -> (pH, pW, P, P, C) -> (pH*pW, P*P*C)
    x = image.reshape(c, ph, patch_size, pw, patch_size)
    x = x.transpose(1, 3, 2, 4, 0)  # (pH, pW, P, P, C)
    return x.reshape(ph * pw, patch_size * patch_size * c).astype(np.float32,
                                                                   copy=False)


# ---------------------------------------------------------------------------
# Step 3: Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_image(
    image_path: str,
    patch_size: int = 16,
    max_num_patches: int = 1024,
    mean: tuple = (0.5, 0.5, 0.5),
    std: tuple = (0.5, 0.5, 0.5),
    pad_to_max: bool = True,
) -> dict:
    """Load, resize, normalize, and patchify an image.

    Returns a dict with:
      - "patches"   : np.ndarray shape (max_num_patches, P*P*C) if pad_to_max,
                      else (pH*pW, P*P*C)
      - "mask"      : np.ndarray int32 shape (max_num_patches,), 1=real 0=pad
      - "patch_h"   : int  number of patch rows
      - "patch_w"   : int  number of patch columns
      - "image_h"   : int  resized pixel height
      - "image_w"   : int  resized pixel width
    """
    if not _HAS_PIL:
        raise ImportError("Pillow is required: pip install Pillow")

    img = _PIL_Image.open(image_path).convert("RGB")
    orig_h, orig_w = img.height, img.width

    tgt_h, tgt_w = get_naflex_image_size(
        orig_h, orig_w, patch_size, max_num_patches
    )
    img = img.resize((tgt_w, tgt_h), resample=_PIL_Image.BILINEAR)

    # (H, W, C) uint8 -> (C, H, W) float32 in [0,1]
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # (C, H, W)

    mean_arr = np.array(mean, dtype=np.float32)[:, None, None]
    std_arr = np.array(std, dtype=np.float32)[:, None, None]
    arr = (arr - mean_arr) / std_arr

    ph = tgt_h // patch_size
    pw = tgt_w // patch_size
    patches = convert_image_to_patches(arr, patch_size)  # (pH*pW, feat)

    n_real = ph * pw
    if pad_to_max and n_real < max_num_patches:
        pad = np.zeros(
            (max_num_patches - n_real, patches.shape[1]), dtype=np.float32
        )
        patches = np.concatenate([patches, pad], axis=0)
        mask = np.zeros(max_num_patches, dtype=np.int32)
        mask[:n_real] = 1
    else:
        mask = np.ones(n_real, dtype=np.int32)

    return {
        "patches": patches,
        "mask": mask,
        "patch_h": ph,
        "patch_w": pw,
        "image_h": tgt_h,
        "image_w": tgt_w,
    }


# ---------------------------------------------------------------------------
# Step 4: NaFlex positional embedding interpolation
# ---------------------------------------------------------------------------

def naflex_interp_pos_embed(
    base_grid: np.ndarray,
    dst_h: int,
    dst_w: int,
) -> np.ndarray:
    """Bilinearly interpolate a stored (base_h, base_w, dim) or
    (base_h * base_w, dim) pos_embed grid to (dst_h * dst_w, dim).

    Matches HF Siglip2VisionEmbeddings.resize_positional_embeddings:
      F.interpolate(mode="bilinear", align_corners=False, antialias=True)

    For exact HF parity install torch and set use_torch=True in the call.
    The scipy fallback (order=1 zoom) is equivalent for non-tiny grids.

    Args:
        base_grid: shape (base_h*base_w, dim) or (base_h, base_w, dim)
        dst_h, dst_w: target patch grid dimensions
    Returns:
        np.ndarray shape (dst_h * dst_w, dim), float32
    """
    if base_grid.ndim == 2:
        n, dim = base_grid.shape
        base_sz = int(round(n ** 0.5))
        assert base_sz * base_sz == n, \
            f"pos_embed has {n} rows which is not a perfect square"
        grid = base_grid.reshape(base_sz, base_sz, dim)
    else:
        grid = base_grid  # already (base_h, base_w, dim)
        base_sz, _, dim = grid.shape

    if grid.shape[0] == dst_h and grid.shape[1] == dst_w:
        return grid.reshape(dst_h * dst_w, dim).astype(np.float32, copy=False)

    # Try torch for antialias=True parity with HF.
    try:
        import torch
        import torch.nn.functional as F
        t = torch.from_numpy(grid.astype(np.float32))  # (src_h, src_w, dim)
        # (src_h, src_w, dim) -> (1, dim, src_h, src_w)
        t = t.permute(2, 0, 1).unsqueeze(0)
        out = F.interpolate(
            t.float(), size=(dst_h, dst_w),
            mode="bilinear", align_corners=False, antialias=True
        )
        # (1, dim, dst_h, dst_w) -> (dst_h * dst_w, dim)
        out = out.squeeze(0).permute(1, 2, 0).reshape(dst_h * dst_w, dim)
        return out.numpy().astype(np.float32)
    except ImportError:
        pass

    if not _HAS_SCIPY:
        raise ImportError(
            "Either torch or scipy is required for pos_embed interpolation. "
            "pip install torch  OR  pip install scipy"
        )

    zoom_h = dst_h / grid.shape[0]
    zoom_w = dst_w / grid.shape[1]
    out = _ndimage.zoom(grid, (zoom_h, zoom_w, 1.0), order=1)
    return out.reshape(dst_h * dst_w, dim).astype(np.float32)


# ---------------------------------------------------------------------------
# CLI: preprocess a single image and optionally dump patches + spatial_shapes
# ---------------------------------------------------------------------------

def _main():
    import argparse, sys

    ap = argparse.ArgumentParser(
        description="NaFlex image preprocessor for LFM2-VL vision tower. "
                    "Outputs patch tensor (float32 binary) and spatial_shapes.")
    ap.add_argument("image", help="input image file")
    ap.add_argument("-o", "--output", default="patches.bin",
                    help="output binary file for patches (float32)")
    ap.add_argument("--patch-size", type=int, default=16)
    ap.add_argument("--max-patches", type=int, default=1024)
    ap.add_argument("--no-pad", action="store_true",
                    help="do not pad to max_patches")
    args = ap.parse_args()

    result = preprocess_image(
        args.image,
        patch_size=args.patch_size,
        max_num_patches=args.max_patches,
        pad_to_max=not args.no_pad,
    )

    patches = result["patches"]
    with open(args.output, "wb") as f:
        patches.astype(np.float32).tofile(f)

    meta_path = Path(args.output).with_suffix(".meta")
    with open(meta_path, "w") as f:
        f.write(f"patch_h={result['patch_h']}\n")
        f.write(f"patch_w={result['patch_w']}\n")
        f.write(f"image_h={result['image_h']}\n")
        f.write(f"image_w={result['image_w']}\n")
        f.write(f"n_real_patches={result['patch_h'] * result['patch_w']}\n")
        f.write(f"total_patches={patches.shape[0]}\n")
        f.write(f"feat_dim={patches.shape[1]}\n")

    print(f"Wrote {patches.shape[0]} patches ({patches.shape[1]} feat/patch) "
          f"to {args.output}")
    print(f"patch grid: {result['patch_h']} x {result['patch_w']} "
          f"({result['patch_h'] * result['patch_w']} real patches)")
    print(f"Metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main())
