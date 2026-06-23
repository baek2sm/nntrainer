// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Samsung Electronics Co., Ltd.
 *
 * @file   winograd_transform.h
 * @date   23 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Winograd F(2x2,3x3) FP32 conv fast path (cpu_backend)
 *
 * Forward-only fast path for 3x3 stride-1 dilation-1 groups==1 conv.
 * Works on nntrainer Tensor objects so the memory planner / pool lifecycle
 * is respected. Uses Tensor::dot (sgemm) for the 16 transform-point GEMMs.
 * Weight transform is cached by the caller (Conv2DLayer member).
 */
#ifndef __WINOGRAD_TRANSFORM_H__
#define __WINOGRAD_TRANSFORM_H__
#ifdef __cplusplus

#include <tensor.h>
#include <tensor_dim.h>

namespace nntrainer {

/**
 * @brief Transform conv weights (Cout, Cin, 3, 3) into Winograd domain
 *        U[16][Cout][Cin] = G · g · G^T. Called once per layer (weight cached).
 *
 * @param filter   weight tensor, dim {Cout, Cin, 3, 3} (NCHW, batch=Cout)
 * @param Cout, Cin output/input channel counts
 * @return Tensor  U, dim {16, Cout, Cin}, FP32
 */
Tensor winograd_transform_weight_f23x3(const Tensor &filter, unsigned int Cout,
                                       unsigned int Cin);

/**
 * @brief Winograd F(2x2,3x3) FP32 forward convolution.
 *
 * @param in        input  Tensor, NCHW {1, Cin, H, W} (single batch)
 * @param U         transformed weights, dim {16, Cout, Cin} (from
 *                  winograd_transform_weight_f23x3)
 * @param out       output Tensor, NCHW {1, Cout, OH, OW} (caller-allocated)
 * @param padH, padW  spatial padding (1,1 for same; 0,0 for valid)
 *
 * Eligibility (caller must guard): kernel 3x3, stride 1, dilation 1, groups 1.
 * Bias is NOT applied here — caller adds it uniformly (same as im2col path).
 */
void winograd_conv2d_f23x3_fp32(const Tensor &in, const Tensor &U, Tensor &out,
                                unsigned int padH, unsigned int padW);

/**
 * @brief Winograd F(2x2,3x3) Q8_0 forward convolution.
 *
 * Same as FP32 but STEP3 GEMM uses Q8_0 quantized U and V (per-point).
 * Transform (U, V) is still FP32; quantize/dequant happens before each
 * point-wise GEMM. MNN hybrid style: float transform, int8 GEMM domain.
 *
 * @param in, U, out, padH, padW  same as FP32 variant
 */
void winograd_conv2d_f23x3_q8_0(const Tensor &in, const Tensor &U, Tensor &out,
                                unsigned int padH, unsigned int padW);

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __WINOGRAD_TRANSFORM_H__ */
