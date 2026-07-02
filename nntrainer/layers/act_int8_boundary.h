// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   act_int8_boundary.h
 * @date   02 July 2026
 * @brief  Shared int8 activation boundary helpers for W4A8 Q8_0_TW tensors.
 *
 * These helpers are used at layer boundaries where a Q8_0_TW tensor must be
 * converted to/from a floating-point tensor. They are intentionally header-only
 * static inlines so that every call site (conv2d epilogue, attention,
 * depthwise, etc.) can specialise to its exact payload layout without adding
 * linkage.
 */

#ifndef __ACT_INT8_BOUNDARY_H__
#define __ACT_INT8_BOUNDARY_H__

#include <cmath>
#include <cstddef>
#include <cstdint>

#ifdef ENABLE_FP16
#include <tensor_dim.h>
#endif

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace nntrainer {
namespace act_int8 {

/**
 * @brief Re-quantize a contiguous FP32 buffer into Q8_0_TW int8.
 * @details out[i] = sat8(round(src[i] / s)), s > 0 required.  Uses symmetric
 * [-127, 127] saturation (the value -128 is never produced to keep the domain
 * compatible with existing int8 GEMM kernels).
 */
static inline void requant_q8_0_tw_from_fp32(const float *src, size_t n,
                                             float s, int8_t *dst) {
  const float id = (s > 0.0f) ? (1.0f / s) : 0.0f;
  for (size_t i = 0; i < n; ++i) {
    int q = (int)std::lroundf(src[i] * id);
    q = q > 127 ? 127 : (q < -127 ? -127 : q);
    dst[i] = static_cast<int8_t>(q);
  }
}

/**
 * @brief Re-quantize a contiguous FP16 buffer into Q8_0_TW int8.
 * @details Identical rounding/saturation semantics to the FP32 variant.
 */
#ifdef ENABLE_FP16
static inline void requant_q8_0_tw_from_fp16(const _FP16 *src, size_t n,
                                             float s, int8_t *dst) {
  const float id = (s > 0.0f) ? (1.0f / s) : 0.0f;

#ifdef __aarch64__
  const size_t vec_len = 8; /* 8 FP16 per 128-bit lane */
  const size_t n_vec = n / vec_len;
  const size_t rem = n % vec_len;

  for (size_t i = 0; i < n_vec; ++i) {
    float16x8_t v = vld1q_f16(reinterpret_cast<const __fp16 *>(src));
    float32x4_t lo = vcvt_f32_f16(vget_low_f16(v));
    float32x4_t hi = vcvt_f32_f16(vget_high_f16(v));
    lo = vmulq_n_f32(lo, id);
    hi = vmulq_n_f32(hi, id);

    /* Round-to-nearest (matches std::lroundf) before converting to int32. */
    int32x4_t q_lo = vcvtq_s32_f32(vrndnq_f32(lo));
    int32x4_t q_hi = vcvtq_s32_f32(vrndnq_f32(hi));

    int16x4_t q_lo16 = vqmovn_s32(q_lo);
    int16x4_t q_hi16 = vqmovn_s32(q_hi);
    int16x8_t q16 = vcombine_s16(q_lo16, q_hi16);
    int8x8_t q8 = vqmovn_s16(q16);

    /* Clamp symmetrically to [-127, 127]. NEON saturating narrowing already
     * clamps to [-128, 127]; we just need to turn a possible -128 into -127. */
    int8x8_t neg127 = vdup_n_s8(-127);
    uint8x8_t max_mask = vcgt_s8(q8, neg127); /* q8 > -127 */
    q8 = vbsl_s8(max_mask, q8, neg127);

    vst1_s8(dst, q8);
    src += vec_len;
    dst += vec_len;
  }

  for (size_t i = 0; i < rem; ++i) {
    int q = (int)std::lroundf(static_cast<float>(*src) * id);
    q = q > 127 ? 127 : (q < -127 ? -127 : q);
    *dst = static_cast<int8_t>(q);
    ++src;
    ++dst;
  }
#else
  for (size_t i = 0; i < n; ++i) {
    int q = (int)std::lroundf(static_cast<float>(src[i]) * id);
    q = q > 127 ? 127 : (q < -127 ? -127 : q);
    dst[i] = static_cast<int8_t>(q);
  }
#endif // __aarch64__
}
#endif // ENABLE_FP16

/**
 * @brief De-quantize a contiguous Q8_0_TW int8 buffer into FP16.
 */
#ifdef ENABLE_FP16
static inline void dequant_fp16_from_q8_0_tw(const int8_t *src, size_t n,
                                             float s, _FP16 *dst) {
#ifdef __aarch64__
  const size_t vec_len = 8; /* 8 int8 per 64-bit lane */
  const size_t n_vec = n / vec_len;
  const size_t rem = n % vec_len;

  const float32x4_t vs = vdupq_n_f32(s);

  for (size_t i = 0; i < n_vec; ++i) {
    int8x8_t v8 = vld1_s8(src);
    int16x8_t v16 = vmovl_s8(v8);
    int32x4_t vlo = vmovl_s16(vget_low_s16(v16));
    int32x4_t vhi = vmovl_s16(vget_high_s16(v16));

    float32x4_t flo = vmulq_f32(vcvtq_f32_s32(vlo), vs);
    float32x4_t fhi = vmulq_f32(vcvtq_f32_s32(vhi), vs);

    float16x4_t hlo = vcvt_f16_f32(flo);
    float16x4_t hhi = vcvt_f16_f32(fhi);
    float16x8_t h = vcombine_f16(hlo, hhi);

    vst1q_f16(reinterpret_cast<__fp16 *>(dst), h);
    src += vec_len;
    dst += vec_len;
  }

  for (size_t i = 0; i < rem; ++i) {
    *dst = static_cast<_FP16>(static_cast<float>(*src) * s);
    ++src;
    ++dst;
  }
#else
  for (size_t i = 0; i < n; ++i)
    dst[i] = static_cast<_FP16>(static_cast<float>(src[i]) * s);
#endif // __aarch64__
}
#endif // ENABLE_FP16

/**
 * @brief De-quantize a contiguous Q8_0_TW int8 buffer into FP32.
 */
static inline void dequant_fp32_from_q8_0_tw(const int8_t *src, size_t n,
                                             float s, float *dst) {
#ifdef __aarch64__
  const size_t vec_len = 16; /* 16 int8 per 128-bit lane */
  const size_t n_vec = n / vec_len;
  const size_t rem = n % vec_len;

  const float32x4_t vs = vdupq_n_f32(s);

  for (size_t i = 0; i < n_vec; ++i) {
    int8x16_t v8 = vld1q_s8(src);
    int16x8_t v16 = vmovl_s8(vget_low_s8(v8));
    int16x8_t v16_hi = vmovl_s8(vget_high_s8(v8));

    int32x4_t vlo = vmovl_s16(vget_low_s16(v16));
    int32x4_t vhi = vmovl_s16(vget_high_s16(v16));
    int32x4_t vlo2 = vmovl_s16(vget_low_s16(v16_hi));
    int32x4_t vhi2 = vmovl_s16(vget_high_s16(v16_hi));

    float32x4_t flo = vmulq_f32(vcvtq_f32_s32(vlo), vs);
    float32x4_t fhi = vmulq_f32(vcvtq_f32_s32(vhi), vs);
    float32x4_t flo2 = vmulq_f32(vcvtq_f32_s32(vlo2), vs);
    float32x4_t fhi2 = vmulq_f32(vcvtq_f32_s32(vhi2), vs);

    vst1q_f32(dst, flo);
    vst1q_f32(dst + 4, fhi);
    vst1q_f32(dst + 8, flo2);
    vst1q_f32(dst + 12, fhi2);
    src += vec_len;
    dst += vec_len;
  }

  for (size_t i = 0; i < rem; ++i) {
    *dst = static_cast<float>(*src) * s;
    ++src;
    ++dst;
  }
#else
  for (size_t i = 0; i < n; ++i)
    dst[i] = static_cast<float>(src[i]) * s;
#endif // __aarch64__
}

} // namespace act_int8
} // namespace nntrainer

#endif // __ACT_INT8_BOUNDARY_H__
