#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Convert PyTorch PFLDInference weights to nntrainer MODEL_FORMAT_BIN.

PyTorch models are trained with NCHW tensors, but nntrainer's `conv2d` and
`fully_connected` layers use the same NCHW layout with the following weight
shapes:

  conv2d filter : (filters, input_channel, kh, kw)
  fc weight     : (1, 1, in_width, unit)   -> stored as (unit, in_width)
  bn weights    : mu, var, gamma, beta     each (1, channels, 1, 1)
  prelu alpha   : (1, channels, 1, 1)

For every Conv+BN+(PReLU) triplet the BN parameters are fused into the preceding
convolution bias so that nntrainer only needs a single conv filter+bias.
The standalone BN after the mixed depthwise convolutions is kept separate
because it is followed by a standalone PReLU layer.
"""

import os
import struct
import sys

import numpy as np
import torch


def nntrainer_conv_weight(tch_weight):
    """PyTorch (out,in,kh,kw) -> nntrainer filter layout is identical."""
    return tch_weight.astype(np.float32)


def nntrainer_fc_weight(tch_weight):
    """PyTorch linear weight (unit,in) -> nntrainer (in,unit)."""
    return tch_weight.T.astype(np.float32)


def nntrainer_bn_weights(mu, var, gamma, beta, eps=1e-5):
    """Return [mu, var, gamma, beta] each as (1,c,1,1) float32 arrays."""
    mu = mu.astype(np.float32).reshape(1, -1, 1, 1)
    var = var.astype(np.float32).reshape(1, -1, 1, 1)
    gamma = gamma.astype(np.float32).reshape(1, -1, 1, 1)
    beta = beta.astype(np.float32).reshape(1, -1, 1, 1)
    return mu, var, gamma, beta


def write_tensor(f, arr):
    """Append a contiguous float32 numpy array in binary."""
    f.write(arr.astype(np.float32).tobytes())


def get(sd, key):
    return sd[key].detach().cpu().numpy()


def fuse_bn_into_conv(tch_w, tch_b, bn_g, bn_b, bn_rm, bn_rv, eps=1e-5):
    """Fuse batchnorm into convolution bias (and optionally scale weights)."""
    scale = bn_g / np.sqrt(bn_rv + eps)
    fused_b = scale * (tch_b - bn_rm) + bn_b if tch_b is not None else -scale * bn_rm + bn_b
    fused_w = tch_w * scale.reshape(-1, 1, 1, 1)
    return fused_w, fused_b


def conv_with_bn_prelu(f, sd, conv_prefix, bn_prefix, prelu_prefix,
                       out_channels, has_bias=False):
    """Conv+BN+PReLU where BN is fused into conv.  Writes filter, bias, [prelu]."""
    w = get(sd, conv_prefix + ".weight")
    b = np.zeros(out_channels, dtype=np.float32)
    if has_bias:
        b = get(sd, conv_prefix + ".bias")

    bn_g = get(sd, bn_prefix + ".weight")
    bn_b = get(sd, bn_prefix + ".bias")
    bn_rm = get(sd, bn_prefix + ".running_mean")
    bn_rv = get(sd, bn_prefix + ".running_var")

    w, b = fuse_bn_into_conv(w, b, bn_g, bn_b, bn_rm, bn_rv)
    write_tensor(f, nntrainer_conv_weight(w))
    write_tensor(f, b.astype(np.float32).reshape(1, -1, 1, 1))
    if prelu_prefix is not None:
        alpha = get(sd, prelu_prefix + ".weight")
        write_tensor(f, alpha.astype(np.float32).reshape(1, -1, 1, 1))


def _conv_bn_prelu(f, sd, conv_prefix, bn_prefix, prelu_prefix, has_bias=False):
    """Write fused conv+bias, then prelu alpha."""
    out_c = sd[conv_prefix + ".weight"].shape[0]
    conv_with_bn_prelu(
        f, sd,
        conv_prefix=conv_prefix,
        bn_prefix=bn_prefix,
        prelu_prefix=prelu_prefix,
        out_channels=out_c,
        has_bias=has_bias,
    )


def conv_block(f, sd, prefix):
    """PyTorch Conv_block = Conv2d + BN + PReLU, written as conv+bias+pReLU."""
    _conv_bn_prelu(f, sd, prefix + ".conv", prefix + ".bn", prefix + ".prelu")


def bn_layer(f, sd, prefix):
    """Standalone BN layer: mu, var, gamma, beta."""
    mu, var, gamma, beta = nntrainer_bn_weights(
        get(sd, prefix + ".running_mean"),
        get(sd, prefix + ".running_var"),
        get(sd, prefix + ".weight"),
        get(sd, prefix + ".bias"),
    )
    write_tensor(f, mu)
    write_tensor(f, var)
    write_tensor(f, gamma)
    write_tensor(f, beta)


def prelu_weight(f, sd, prefix):
    alpha = get(sd, prefix + ".weight")
    write_tensor(f, alpha.astype(np.float32).reshape(1, -1, 1, 1))


def mdconv_weights(f, sd, prefix, splits):
    """Depthwise splits: each (c,1,k,k) stays as is, no BN fusion here."""
    for i in range(len(splits)):
        w = get(sd, f"{prefix}.mixed_depthwise_conv.{i}.weight")
        write_tensor(f, nntrainer_conv_weight(w))


def se_block(f, sd, prefix):
    """SE reduce conv+bias, expand conv+bias."""
    w = get(sd, prefix + ".se_reduce.weight")
    b = get(sd, prefix + ".se_reduce.bias")
    write_tensor(f, nntrainer_conv_weight(w))
    write_tensor(f, b.astype(np.float32).reshape(1, -1, 1, 1))

    w = get(sd, prefix + ".se_expand.weight")
    b = get(sd, prefix + ".se_expand.bias")
    write_tensor(f, nntrainer_conv_weight(w))
    write_tensor(f, b.astype(np.float32).reshape(1, -1, 1, 1))


def mixed_depthwise_block(f, sd, prefix, splits):
    """Expansion -> MDConv -> BN+PReLU -> SE -> project -> (residual add)

    nntrainer graph (main.cpp):
      - expand conv + bn (+ prelu if present) -> conv+bias (fused), prelu alpha
      - mdconv (split+dw+concat)              -> dw weights
      - dw bn + prelu                         -> bn weights, prelu alpha
      - SE                                     -> reduce/expand conv+bias
      - project conv + bn                      -> conv+bias (fused)
      - optional residual add                  -> no weights
    """
    # expansion conv + bn + prelu
    conv_block(f, sd, prefix + ".conv")

    # mdconv splits
    mdconv_weights(f, sd, prefix + ".conv_dw", splits)

    # conv_dw bn + prelu
    bn_layer(f, sd, prefix + ".conv_dw.bn")
    prelu_weight(f, sd, prefix + ".conv_dw.prelu")

    # squeeze-and-excite
    se_block(f, sd, prefix + ".squeeze_excite")

    # project conv + bn  (Linear_block, no prelu)
    _conv_bn_prelu(f, sd, prefix + ".project.conv", prefix + ".project.bn",
                   None)

    # The PyTorch mixed-depthwise blocks do NOT have a final prelu on the
    # residual path; they only use prelu inside conv_block and after mdconv.


def tail_conv_bn_relu(f, sd, conv7_prefix, conv8_prefix):
    """conv7/bn/relu and conv8/bn/relu are standard conv+bn+relu blocks.

    We keep BN separate because nntrainer batch_normalization with activation=relu
    handles both.  Weights: conv filter+bias, BN mu/var/gamma/beta.
    """
    for prefix in [conv7_prefix, conv8_prefix]:
        w = get(sd, prefix + ".0.weight")
        b = np.zeros(w.shape[0], dtype=np.float32)
        write_tensor(f, nntrainer_conv_weight(w))
        write_tensor(f, b.reshape(1, -1, 1, 1))
        bn_layer(f, sd, prefix + ".1")


def convert(model_path, out_path):
    model = torch.jit.load(model_path)
    sd = {k: v for k, v in model.state_dict().items() if v.dtype == torch.float}

    # Map from PyTorch parameter names to the nntrainer layer name list in
    # build order.  This must match Applications/FaceLandmark/jni/main.cpp
    # exactly.
    with open(out_path, "wb") as f:
        # conv1
        conv_block(f, sd, "conv1")

        # conv2_dw
        conv_block(f, sd, "conv2_dw")

        # conv_23
        mixed_depthwise_block(f, sd, "conv_23", [24, 12, 12])

        # conv_3 0..3
        for i in range(4):
            mixed_depthwise_block(f, sd, f"conv_3.model.{i}", [36, 12])

        # conv_34
        mixed_depthwise_block(f, sd, "conv_34", [48, 24, 24])

        # conv_4 0..5
        for i in range(6):
            mixed_depthwise_block(f, sd, f"conv_4.model.{i}", [72, 24])

        # conv_45
        mixed_depthwise_block(f, sd, "conv_45", [48, 48, 48, 48])

        # conv_5 0..1
        for i in range(2):
            mixed_depthwise_block(f, sd, f"conv_5.model.{i}", [32, 32, 32])

        # block6_2
        conv_block(f, sd, "block6_2")

        # conv7 / conv8  (stored in PyTorch as conv7.0.weight etc.)
        tail_conv_bn_relu(f, sd, "conv7", "conv8")

        # fc
        w = get(sd, "fc.weight")
        b = get(sd, "fc.bias")
        write_tensor(f, nntrainer_fc_weight(w))
        write_tensor(f, b.astype(np.float32).reshape(1, -1, 1, 1))

    total = os.path.getsize(out_path) // 4
    print(f"wrote {out_path}: {total} floats ({total * 4} bytes)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <face_landmark.pt> <out.bin>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
