// SPDX-License-Identifier: Apache-2.0
/** @file turboquant_codec.c
 * @brief Portable codec; norm correction is computed once when appending.
 * @author NNTrainer contributors
 * @bug None known.
 */
#include "turboquant_codec.h"
#include <math.h>
#include <string.h>
#if defined(__hexagon__)
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#endif

static int finite_value(float value) {
  uint32_t bits;
  memcpy(&bits, &value, 4);
  return (bits & 0x7f800000u) != 0x7f800000u;
}

// Generated from the Gaussian Lloyd-Max solver (200 refinement steps,
// 200 trapezoid intervals) used by the independent C++ reference. Persist
// FP32 levels and boundaries to avoid solving the same codebook on DSP open.
static const float levels[5][16] = {
  {-0x1.5d8edc0000000p-2f, -0x1.08c2e00000000p-2f, -0x1.9e2d800000000p-3f,
   -0x1.419a180000000p-3f, -0x1.e288940000000p-4f, -0x1.5052900000000p-4f,
   -0x1.8d73fe0000000p-5f, -0x1.0705140000000p-6f, 0x1.0705140000000p-6f,
   0x1.8d73fe0000000p-5f, 0x1.5052900000000p-4f, 0x1.e288940000000p-4f,
   0x1.419a180000000p-3f, 0x1.9e2d800000000p-3f, 0x1.08c2e00000000p-2f,
   0x1.5d8edc0000000p-2f},
  {-0x1.ee59880000000p-3f, -0x1.766dce0000000p-3f, -0x1.24de2e0000000p-3f,
   -0x1.c6d0560000000p-4f, -0x1.5533e40000000p-4f, -0x1.dba1c20000000p-5f,
   -0x1.190ab20000000p-5f, -0x1.73f75c0000000p-7f, 0x1.73f75c0000000p-7f,
   0x1.190ab20000000p-5f, 0x1.dba1c20000000p-5f, 0x1.5533e40000000p-4f,
   0x1.c6d0560000000p-4f, 0x1.24de2e0000000p-3f, 0x1.766dce0000000p-3f,
   0x1.ee59880000000p-3f},
  {-0x1.5d8edc0000000p-3f, -0x1.08c2e00000000p-3f, -0x1.9e2d800000000p-4f,
   -0x1.419a180000000p-4f, -0x1.e288940000000p-5f, -0x1.5052900000000p-5f,
   -0x1.8d73fe0000000p-6f, -0x1.0705140000000p-7f, 0x1.0705140000000p-7f,
   0x1.8d73fe0000000p-6f, 0x1.5052900000000p-5f, 0x1.e288940000000p-5f,
   0x1.419a180000000p-4f, 0x1.9e2d800000000p-4f, 0x1.08c2e00000000p-3f,
   0x1.5d8edc0000000p-3f},
  {-0x1.ee59880000000p-4f, -0x1.766dce0000000p-4f, -0x1.24de2e0000000p-4f,
   -0x1.c6d0560000000p-5f, -0x1.5533e40000000p-5f, -0x1.dba1c20000000p-6f,
   -0x1.190ab20000000p-6f, -0x1.73f75c0000000p-8f, 0x1.73f75c0000000p-8f,
   0x1.190ab20000000p-6f, 0x1.dba1c20000000p-6f, 0x1.5533e40000000p-5f,
   0x1.c6d0560000000p-5f, 0x1.24de2e0000000p-4f, 0x1.766dce0000000p-4f,
   0x1.ee59880000000p-4f},
  {-0x1.5d8edc0000000p-4f, -0x1.08c2e00000000p-4f, -0x1.9e2d800000000p-5f,
   -0x1.419a180000000p-5f, -0x1.e288940000000p-6f, -0x1.5052900000000p-6f,
   -0x1.8d73fe0000000p-7f, -0x1.0705140000000p-8f, 0x1.0705140000000p-8f,
   0x1.8d73fe0000000p-7f, 0x1.5052900000000p-6f, 0x1.e288940000000p-6f,
   0x1.419a180000000p-5f, 0x1.9e2d800000000p-5f, 0x1.08c2e00000000p-4f,
   0x1.5d8edc0000000p-4f},
};
static const float boundaries[5][15] = {
  {-0x1.3328de0000000p-2f, -0x1.d7d9a00000000p-3f, -0x1.6fe3cc0000000p-3f,
   -0x1.196f320000000p-3f, -0x1.996d920000000p-4f, -0x1.0b86480000000p-4f,
   -0x1.087b440000000p-5f, -0x1.5800000000000p-53f, 0x1.087b440000000p-5f,
   0x1.0b86480000000p-4f, 0x1.996d920000000p-4f, 0x1.196f320000000p-3f,
   0x1.6fe3cc0000000p-3f, 0x1.d7d9a00000000p-3f, 0x1.3328de0000000p-2f},
  {-0x1.b263aa0000000p-3f, -0x1.4da5fe0000000p-3f, -0x1.04232c0000000p-3f,
   -0x1.8e021e0000000p-4f, -0x1.2182620000000p-4f, -0x1.7a563a0000000p-5f,
   -0x1.7608880000000p-6f, -0x1.0800000000000p-53f, 0x1.7608880000000p-6f,
   0x1.7a563a0000000p-5f, 0x1.2182620000000p-4f, 0x1.8e021e0000000p-4f,
   0x1.04232c0000000p-3f, 0x1.4da5fe0000000p-3f, 0x1.b263aa0000000p-3f},
  {-0x1.3328de0000000p-3f, -0x1.d7d9a00000000p-4f, -0x1.6fe3cc0000000p-4f,
   -0x1.196f320000000p-4f, -0x1.996d920000000p-5f, -0x1.0b86480000000p-5f,
   -0x1.087b440000000p-6f, -0x1.5800000000000p-54f, 0x1.087b440000000p-6f,
   0x1.0b86480000000p-5f, 0x1.996d920000000p-5f, 0x1.196f320000000p-4f,
   0x1.6fe3cc0000000p-4f, 0x1.d7d9a00000000p-4f, 0x1.3328de0000000p-3f},
  {-0x1.b263aa0000000p-4f, -0x1.4da5fe0000000p-4f, -0x1.04232c0000000p-4f,
   -0x1.8e021e0000000p-5f, -0x1.2182620000000p-5f, -0x1.7a563a0000000p-6f,
   -0x1.7608880000000p-7f, -0x1.0800000000000p-54f, 0x1.7608880000000p-7f,
   0x1.7a563a0000000p-6f, 0x1.2182620000000p-5f, 0x1.8e021e0000000p-5f,
   0x1.04232c0000000p-4f, 0x1.4da5fe0000000p-4f, 0x1.b263aa0000000p-4f},
  {-0x1.3328de0000000p-4f, -0x1.d7d9a00000000p-5f, -0x1.6fe3cc0000000p-5f,
   -0x1.196f320000000p-5f, -0x1.996d920000000p-6f, -0x1.0b86480000000p-6f,
   -0x1.087b440000000p-7f, -0x1.5800000000000p-55f, 0x1.087b440000000p-7f,
   0x1.0b86480000000p-6f, 0x1.996d920000000p-6f, 0x1.196f320000000p-5f,
   0x1.6fe3cc0000000p-5f, 0x1.d7d9a00000000p-5f, 0x1.3328de0000000p-4f},
};
int nntr_tq_init(nntr_tq_codec *c, uint32_t dim) {
  if (!c || dim < 64 || dim > 1024 || (dim & (dim - 1)))
    return -1;
  uint32_t index = 0;
  for (uint32_t d = 64; d < dim; d *= 2)
    ++index;
  c->dim = dim;
  memcpy(c->centroids, levels[index], sizeof(c->centroids));
  memcpy(c->boundaries, boundaries[index], sizeof(c->boundaries));
  return 0;
}

void nntr_tq_rotate(float *row, uint32_t dim) {
#if defined(__hexagon__)
  if (dim >= 32 && dim <= 1024 && !(dim & (dim - 1)) &&
      !((uintptr_t)row & 127u)) {
    static const int32_t lanes[32] __attribute__((aligned(128))) = {
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
      16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    for (uint32_t i = 0; i < dim; i += 32) {
      HVX_Vector v = *(HVX_Vector *)(row + i);
      for (uint32_t step = 1; step < 32; step *= 2) {
        const HVX_Vector bit = Q6_V_vsplat_R(step);
        const HVX_VectorPred upper =
          Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(*(const HVX_Vector *)lanes, bit), bit);
        const HVX_Vector sign =
          Q6_V_vmux_QVV(upper, Q6_V_vsplat_R(0x80000000u), Q6_V_vzero());
        const HVX_Vector other =
          Q6_V_vdelta_VV(v, Q6_V_vsplat_R(0x04040404u * step));
        v = Q6_Vsf_equals_Vqf32(
          Q6_Vqf32_vadd_VsfVsf(Q6_V_vxor_VV(v, sign), other));
      }
      *(HVX_Vector *)(row + i) = v;
    }
    for (uint32_t step = 32; step < dim; step *= 2)
      for (uint32_t base = 0; base < dim; base += 2 * step)
        for (uint32_t i = 0; i < step; i += 32) {
          const HVX_Vector a = *(HVX_Vector *)(row + base + i),
                           b = *(HVX_Vector *)(row + base + step + i);
          *(HVX_Vector *)(row + base + i) =
            Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
          *(HVX_Vector *)(row + base + step + i) =
            Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b));
        }
    const float scale = 1.0f / sqrtf((float)dim);
    uint32_t bits;
    memcpy(&bits, &scale, 4);
    for (uint32_t i = 0; i < dim; i += 32)
      *(HVX_Vector *)(row + i) = Q6_Vsf_equals_Vqf32(
        Q6_Vqf32_vmpy_VsfVsf(*(HVX_Vector *)(row + i), Q6_V_vsplat_R(bits)));
    return;
  }
#endif
  for (uint32_t span = 1; span < dim; span *= 2)
    for (uint32_t base = 0; base < dim; base += 2 * span)
      for (uint32_t i = 0; i < span; ++i) {
        const float a = row[base + i], b = row[base + span + i];
        row[base + i] = a + b;
        row[base + span + i] = a - b;
      }
  const float scale = 1.0f / sqrtf((float)dim);
  for (uint32_t i = 0; i < dim; ++i)
    row[i] *= scale;
}

#if defined(__hexagon__) && __HEXAGON_ARCH__ >= 79
static HVX_Vector tq_splat(float x) {
  uint32_t bits;
  memcpy(&bits, &x, 4);
  return Q6_V_vsplat_R(bits);
}
static int tq_any(HVX_Vector bits) {
  for (uint32_t shift = 4; shift <= 64; shift *= 2)
    bits = Q6_V_vor_VV(bits, Q6_V_vror_VR(bits, shift));
  return Q6_R_vextract_VR(bits, 0) != 0;
}
#endif
int nntr_tq_encode_key(const nntr_tq_codec *c, const float *row,
                       uint8_t *packed, float *scratch) {
  if (!c || !row || !packed || !scratch)
    return -1;
  double squared = 0;
  float norm = 0;
#if defined(__hexagon__) && __HEXAGON_ARCH__ >= 79
  int vector_norm = 0;
  HVX_Vector sum = Q6_V_vzero(), maximum = Q6_V_vzero();
  for (uint32_t i = 0; i < c->dim; i += 32) {
    const HVX_Vector x = *(const HVX_UVector *)(row + i);
    maximum =
      Q6_Vw_vmax_VwVw(maximum, Q6_V_vand_VV(x, Q6_V_vsplat_R(0x7fffffff)));
    sum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(
      sum, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x, x))));
  }
  for (uint32_t shift = 4; shift <= 64; shift *= 2) {
    maximum = Q6_Vw_vmax_VwVw(maximum, Q6_V_vror_VR(maximum, shift));
    sum =
      Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(sum, Q6_V_vror_VR(sum, shift)));
  }
  uint32_t bits = Q6_R_vextract_VR(maximum, 0);
  float largest;
  memcpy(&largest, &bits, 4);
  if (!finite_value(largest))
    return -1;
  if (largest > 1e-10f && largest < 1e10f) {
    bits = Q6_R_vextract_VR(sum, 0);
    float value;
    memcpy(&value, &bits, 4);
    norm = sqrtf(value);
    vector_norm = 1;
  }
  if (!vector_norm)
#endif
  {
    for (uint32_t i = 0; i < c->dim; ++i) {
      if (!finite_value(row[i]))
        return -1;
      squared += (double)row[i] * row[i];
    }
    norm = (float)sqrt(squared);
  }
  if (!finite_value(norm))
    return -1;
#if defined(__hexagon__) && __HEXAGON_ARCH__ >= 79
  if (norm > 1e-30f && norm < 1e30f && !((uintptr_t)scratch & 127u)) {
    const HVX_Vector inv = tq_splat(1.0f / norm);
    for (uint32_t i = 0; i < c->dim; i += 32)
      *(HVX_Vector *)(scratch + i) = Q6_Vsf_equals_Vqf32(
        Q6_Vqf32_vmpy_VsfVsf(*(const HVX_UVector *)(row + i), inv));
    nntr_tq_rotate(scratch, c->dim);
    uint32_t codes[1024] __attribute__((aligned(128)));
    uint16_t low[64] __attribute__((aligned(128))) = {0},
                     high[64] __attribute__((aligned(128))) = {0};
    for (uint32_t j = 0; j < 16; ++j) {
      uint32_t bits;
      memcpy(&bits, c->centroids + j, 4);
      low[2 * j] = (uint16_t)bits;
      high[2 * j] = (uint16_t)(bits >> 16);
    }
    HVX_Vector centroid_squared = Q6_V_vzero();
    HVX_Vector ambiguous = Q6_V_vzero();
    for (uint32_t i = 0; i < c->dim; i += 32) {
      const HVX_Vector x = *(HVX_Vector *)(scratch + i);
      HVX_Vector code = Q6_V_vzero();
      for (uint32_t j = 0; j < 15; ++j) {
        const HVX_Vector b = tq_splat(c->boundaries[j]);
        const HVX_VectorPred below = Q6_Q_vcmp_gt_VsfVsf(b, x);
        code = Q6_Vw_vadd_VwVw(
          code, Q6_V_vmux_QVV(below, Q6_V_vzero(), Q6_V_vsplat_R(1)));
        const HVX_Vector distance =
          Q6_V_vand_VV(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(x, b)),
                       Q6_V_vsplat_R(0x7fffffff));
        ambiguous = Q6_V_vor_VV(
          ambiguous,
          Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VsfVsf(tq_splat(2e-7f), distance),
                        Q6_V_vsplat_R(1), Q6_V_vzero()));
      }
      *(HVX_Vector *)(codes + i) = code;
      const HVX_Vector index = Q6_Vh_vpacke_VwVw(code, code);
      const HVX_Vector a =
        Q6_V_lo_W(Q6_Wh_vlut16_VbVhR(index, *(const HVX_Vector *)low, 0));
      const HVX_Vector b =
        Q6_V_lo_W(Q6_Wh_vlut16_VbVhR(index, *(const HVX_Vector *)high, 0));
      const HVX_Vector centroid =
        Q6_V_vor_VV(Q6_V_lo_W(Q6_Wuw_vunpack_Vuh(a)),
                    Q6_Vw_vasl_VwR(Q6_V_lo_W(Q6_Wuw_vunpack_Vuh(b)), 16));
      centroid_squared = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(
        centroid_squared,
        Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(centroid, centroid))));
    }
    // Close to a decision boundary, retain the scalar division/search so
    // reciprocal rounding cannot change a four-bit code.
    if (!tq_any(ambiguous)) {
      memset(packed, 0, c->dim / 2 + 8);
      squared = 0;
      for (uint32_t i = 0; i < c->dim; ++i) {
        const uint32_t code = codes[i];
        packed[i / 2] |= (uint8_t)(code << (4 * (i % 2)));
      }
      for (uint32_t shift = 4; shift <= 64; shift *= 2)
        centroid_squared = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(
          centroid_squared, Q6_V_vror_VR(centroid_squared, shift)));
      uint32_t bits = Q6_R_vextract_VR(centroid_squared, 0);
      float value;
      memcpy(&value, &bits, 4);
      const float corrected = norm / sqrtf(value);
      memcpy(packed + c->dim / 2, &norm, 4);
      memcpy(packed + c->dim / 2 + 4, &corrected, 4);
      return 0;
    }
  }
  if (vector_norm) {
    squared = 0;
    for (uint32_t i = 0; i < c->dim; ++i)
      squared += (double)row[i] * row[i];
    norm = (float)sqrt(squared);
  }
#endif
  for (uint32_t i = 0; i < c->dim; ++i)
    scratch[i] = norm > 0 ? row[i] / norm : 0;
  nntr_tq_rotate(scratch, c->dim);
  memset(packed, 0, c->dim / 2 + 8);
  squared = 0;
  for (uint32_t i = 0; i < c->dim; ++i) {
    uint32_t lo = 0, hi = 15;
    while (lo < hi) {
      const uint32_t mid = (lo + hi) / 2;
      if (scratch[i] >= c->boundaries[mid])
        lo = mid + 1;
      else
        hi = mid;
    }
    packed[i / 2] |= (uint8_t)(lo << (4 * (i % 2)));
    squared += (double)c->centroids[lo] * c->centroids[lo];
  }
  const float corrected = norm / (float)sqrt(squared);
  memcpy(packed + c->dim / 2, &norm, 4);
  memcpy(packed + c->dim / 2 + 4, &corrected, 4);
  return 0;
}

int nntr_tq_encode_value(const nntr_tq_codec *c, const float *row,
                         uint8_t *packed) {
  if (!c || !row || !packed)
    return -1;
  float lo = row[0], hi = row[0];
#if defined(__hexagon__) && __HEXAGON_ARCH__ >= 79
  HVX_Vector low = *(const HVX_UVector *)row, high = low,
             max_exp = Q6_V_vzero();
  for (uint32_t i = 0; i < c->dim; i += 32) {
    const HVX_Vector x = *(const HVX_UVector *)(row + i);
    low = Q6_Vsf_vmin_VsfVsf(low, x);
    high = Q6_Vsf_vmax_VsfVsf(high, x);
    max_exp =
      Q6_Vw_vmax_VwVw(max_exp, Q6_V_vand_VV(x, Q6_V_vsplat_R(0x7f800000u)));
  }
  for (uint32_t shift = 4; shift <= 64; shift *= 2) {
    low = Q6_Vsf_vmin_VsfVsf(low, Q6_V_vror_VR(low, shift));
    high = Q6_Vsf_vmax_VsfVsf(high, Q6_V_vror_VR(high, shift));
    max_exp = Q6_Vw_vmax_VwVw(max_exp, Q6_V_vror_VR(max_exp, shift));
  }
  if ((uint32_t)Q6_R_vextract_VR(max_exp, 0) == 0x7f800000u)
    return -1;
  uint32_t bits = Q6_R_vextract_VR(low, 0);
  memcpy(&lo, &bits, 4);
  bits = Q6_R_vextract_VR(high, 0);
  memcpy(&hi, &bits, 4);
#else
  for (uint32_t i = 0; i < c->dim; ++i) {
    if (!finite_value(row[i]))
      return -1;
    if (row[i] < lo)
      lo = row[i];
    if (row[i] > hi)
      hi = row[i];
  }
#endif
  const float scale = (float)(((double)hi - lo) / 15.0);
  if (!finite_value(scale))
    return -1;
#if defined(__hexagon__) && __HEXAGON_ARCH__ >= 79
  if (scale > 1e-30f && scale < 1e30f) {
    uint32_t codes[1024] __attribute__((aligned(128)));
    HVX_Vector ambiguous = Q6_V_vzero();
    for (uint32_t i = 0; i < c->dim; i += 32) {
      HVX_Vector value = Q6_Vsf_equals_Vqf32(
        Q6_Vqf32_vsub_VsfVsf(*(const HVX_UVector *)(row + i), tq_splat(lo)));
      value = Q6_Vsf_equals_Vqf32(
        Q6_Vqf32_vmpy_VsfVsf(value, tq_splat(1.0f / scale)));
      value = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(value, tq_splat(0.5f)));
      HVX_Vector code = Q6_Vw_equals_Vsf(value);
      const HVX_Vector fraction = Q6_Vsf_equals_Vqf32(
        Q6_Vqf32_vsub_VsfVsf(value, Q6_Vsf_equals_Vw(code)));
      const HVX_VectorPred near =
        Q6_Q_or_QQ(Q6_Q_vcmp_gt_VsfVsf(tq_splat(1e-5f), fraction),
                   Q6_Q_vcmp_gt_VsfVsf(fraction, tq_splat(0.99999f)));
      ambiguous = Q6_V_vor_VV(
        ambiguous, Q6_V_vmux_QVV(near, Q6_V_vsplat_R(1), Q6_V_vzero()));
      code =
        Q6_Vw_vmax_VwVw(Q6_V_vzero(), Q6_Vw_vmin_VwVw(code, Q6_V_vsplat_R(15)));
      *(HVX_Vector *)(codes + i) = code;
    }
    if (!tq_any(ambiguous)) {
      for (uint32_t i = 0; i < c->dim; i += 2)
        packed[i / 2] = (uint8_t)(codes[i] | (codes[i + 1] << 4));
      memcpy(packed + c->dim / 2, &scale, 4);
      memcpy(packed + c->dim / 2 + 4, &lo, 4);
      return 0;
    }
  }
#endif
  memset(packed, 0, c->dim / 2 + 8);
  for (uint32_t i = 0; i < c->dim; ++i) {
    double value = scale > 0 ? floor(((double)row[i] - lo) / scale + 0.5) : 0;
    if (value < 0)
      value = 0;
    if (value > 15)
      value = 15;
    packed[i / 2] |= (uint8_t)((uint32_t)value << (4 * (i % 2)));
  }
  memcpy(packed + c->dim / 2, &scale, 4);
  memcpy(packed + c->dim / 2 + 4, &lo, 4);
  return 0;
}
