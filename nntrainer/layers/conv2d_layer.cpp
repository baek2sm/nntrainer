// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   conv2d_layer.h
 * @date   02 June 2020
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Convolution Layer Class for Neural Network
 *
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include <conv2d_layer.h>
#include <cpu_backend.h>
#include <layer_context.h>
#include <lazy_tensor.h>
#include <nntr_threads.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <profiler.h>
#include <q8_0_tensor.h>
#include <tensor_dim.h>
#include <thread>
#include <thread_manager.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

namespace {

/**
 * @brief In-place SiLU / swish (x * sigmoid(x)) over a contiguous buffer.
 * @details Fuses what would otherwise be a separate Activation layer (a full
 * extra read+write pass over the conv output) into the conv epilogue. The
 * sigmoid is evaluated in fp32 and cast back to T so an FP16 activation graph
 * does not lose precision (or overflow exp) in the half domain. Called after
 * all conv compute completes, so it is never nested inside another
 * parallel_for.
 *
 * @note ThreadManager::parallel_for invokes its callback through a type-erased
 * std::function PER INDEX, so iterating it at element granularity would pay a
 * non-inlinable call per element (measured ~3x slower than a serial inlined
 * loop). Instead we parallelize over a handful of contiguous chunks (one per
 * compute thread) and run a tight, fully-inlined inner loop inside each — the
 * std::function is then hit only ~nthreads times while the exp stays inlined.
 */
template <typename T>
static inline void convApplySwishInplace(T *data, size_t n) {
  auto &tm = ThreadManager::Global();
  const size_t nthreads = std::max<size_t>(1, tm.getComputeThreadCount());
  const size_t chunk = (n + nthreads - 1) / nthreads;
  tm.parallel_for(0, nthreads, [&](size_t t) {
    const size_t start = t * chunk;
    if (start >= n)
      return;
    const size_t end = std::min(start + chunk, n);
    for (size_t i = start; i < end; ++i) {
      const float x = static_cast<float>(data[i]);
      data[i] = static_cast<T>(x / (1.0f + std::exp(-x)));
    }
  });
}

static TensorDim calcCol2ImOutputDim(const TensorDim &out,
                                     const TensorDim &kdim) {

  return TensorDim({kdim.getFeatureLen(), out.width() * out.height()},
                   out.getTensorType());
}

#ifdef ENABLE_FP16
/**
 * @brief Tensor-wise quantize FP16 NHWC [n_spatial, in_ch] -> tensor_q8_0.
 *
 * One fp16 scale is written at @a dst, followed by flat int8 quants.
 * Padding (if any) is zeroed out to a multiple of QK8_0.
 */
static inline void quantize_nhwc_q8_0_tensor(const _FP16 *src, int n_spatial,
                                              int in_ch, uint8_t *dst) {
  const size_t nelem = (size_t)n_spatial * in_ch;
  float amax = 0.0f;
  for (size_t i = 0; i < nelem; ++i) {
    float v = std::abs(static_cast<float>(src[i]));
    if (v > amax)
      amax = v;
  }

  float d = amax / 127.0f;
  if (d == 0.0f)
    d = 1.0f;
  const _FP16 d_h = static_cast<_FP16>(d);
  uint16_t d_u16;
  std::memcpy(&d_u16, &d_h, 2);
  std::memcpy(dst, &d_u16, sizeof(uint16_t));

  const float id = 1.0f / d;
  int8_t *qs = reinterpret_cast<int8_t *>(dst + sizeof(uint16_t));
  for (size_t i = 0; i < nelem; ++i) {
    float v = static_cast<float>(src[i]) * id;
    v = std::round(v);
    if (v > 127.0f)
      v = 127.0f;
    if (v < -127.0f)
      v = -127.0f;
    qs[i] = static_cast<int8_t>(v);
  }
  const size_t n_pad = (nelem + QK8_0 - 1) / QK8_0 * QK8_0;
  for (size_t i = nelem; i < n_pad; ++i)
    qs[i] = 0;
}

/**
 * @brief Dequantize tensor-wise Q8_0 activation to FP16 / FP32.
 *
 * @a src is the int8 quant buffer (the tensor's getData() pointer). The single
 * fp16 scale lives at src - sizeof(uint16_t). Output @a dst is a contiguous
 * buffer of @a nelem elements in the same logical order.
 */
static inline void dequantize_tensor_wise_q8_0_to_fp16(const int8_t *src,
                                                        size_t nelem,
                                                        _FP16 *dst) {
  const uint8_t *storage = reinterpret_cast<const uint8_t *>(src) -
                           sizeof(uint16_t);
  uint16_t d_u16;
  std::memcpy(&d_u16, storage, sizeof(uint16_t));
  float scale = static_cast<float>(*reinterpret_cast<const _FP16 *>(&d_u16));
  for (size_t i = 0; i < nelem; ++i)
    dst[i] = static_cast<_FP16>(static_cast<float>(src[i]) * scale);
}

static inline void dequantize_tensor_wise_q8_0_to_fp32(const int8_t *src,
                                                        size_t nelem,
                                                        float *dst) {
  const uint8_t *storage = reinterpret_cast<const uint8_t *>(src) -
                           sizeof(uint16_t);
  uint16_t d_u16;
  std::memcpy(&d_u16, storage, sizeof(uint16_t));
  float scale = static_cast<float>(*reinterpret_cast<const _FP16 *>(&d_u16));
  for (size_t i = 0; i < nelem; ++i)
    dst[i] = static_cast<float>(src[i]) * scale;
}

/**
 * @brief Tensor-wise quantize FP32 NHWC [n_spatial, in_ch] -> tensor_q8_0.
 */
static inline void quantize_nhwc_q8_0_tensor_fp32(const float *src,
                                                   int n_spatial, int in_ch,
                                                   uint8_t *dst) {
  const size_t nelem = (size_t)n_spatial * in_ch;
  float amax = 0.0f;
  for (size_t i = 0; i < nelem; ++i)
    amax = std::max(amax, std::abs(src[i]));

  float d = (amax == 0.0f) ? 1.0f : (amax / 127.0f);
  const _FP16 d_h = static_cast<_FP16>(d);
  uint16_t d_u16;
  std::memcpy(&d_u16, &d_h, 2);
  std::memcpy(dst, &d_u16, sizeof(uint16_t));

  const float id = 1.0f / d;
  int8_t *qs = reinterpret_cast<int8_t *>(dst + sizeof(uint16_t));
  for (size_t i = 0; i < nelem; ++i) {
    float v = src[i] * id;
    v = std::round(v);
    if (v > 127.0f)
      v = 127.0f;
    if (v < -127.0f)
      v = -127.0f;
    qs[i] = static_cast<int8_t>(v);
  }
  const size_t n_pad = (nelem + QK8_0 - 1) / QK8_0 * QK8_0;
  for (size_t i = nelem; i < n_pad; ++i)
    qs[i] = 0;
}

/**
 * @brief Fused bias + SiLU + tensor-wise Q8_0 quantize on FP16 scratch.
 *
 * @a src is [n_spatial, out_ch] row-major FP16. SiLU is applied element-wise
 * after bias addition, and the result is quantized into a tensor-wise Q8_0
 * whose scale is written to @a dst (storage base) and quants to dst+2.
 * SiLU is realized with a 256-entry LUT in the input quant domain to avoid
 * per-element exp() in the final pass.
 */
static inline void fused_bias_silu_quantize_tensor_wise_q8_0(
  _FP16 *src, int n_spatial, int out_ch, const _FP16 *bias, uint8_t *dst,
  const char *layer_name = nullptr) {
  const size_t nelem = (size_t)n_spatial * out_ch;
  auto &tm = ThreadManager::Global();
  const unsigned int chunk = 1024;
  const size_t loops = (n_spatial + chunk - 1) / chunk;

  struct LocalMax {
    float pre;
    float post;
  };
  std::vector<LocalMax> local_max(loops, {0.0f, 0.0f});
  const bool has_bias = bias != nullptr;

  // Pass 1: add bias, keep pre-SiLU values, and find per-chunk max abs of
  // pre/post SiLU values. SiLU is monotonic-ish; we keep the pre value and
  // compute the post value only for scale selection.
  tm.parallel_for(0, loops, [&](size_t idx) {
    unsigned int r0 = idx * chunk;
    unsigned int r1 = std::min(r0 + chunk, (unsigned int)n_spatial);
    float pre_max = 0.0f;
    float post_max = 0.0f;
    for (unsigned int r = r0; r < r1; ++r) {
      _FP16 *row = src + (size_t)r * out_ch;
      for (int c = 0; c < out_ch; ++c) {
        float v = static_cast<float>(row[c]);
        if (has_bias)
          v += static_cast<float>(bias[c]);
        row[c] = static_cast<_FP16>(v);
        pre_max = std::max(pre_max, std::abs(v));
        const float y = v / (1.0f + std::exp(-v));
        post_max = std::max(post_max, std::abs(y));
      }
    }
    local_max[idx] = {pre_max, post_max};
  });

  float amax_pre = 0.0f;
  float amax_post = 0.0f;
  for (const auto &lm : local_max) {
    amax_pre = std::max(amax_pre, lm.pre);
    amax_post = std::max(amax_post, lm.post);
  }

  float d_pre = amax_pre / 127.0f;
  float d_post = amax_post / 127.0f;
  if (d_pre == 0.0f)
    d_pre = 1.0f;
  if (d_post == 0.0f)
    d_post = 1.0f;

  if (layer_name != nullptr &&
      (std::strcmp(layer_name, "m4/cv2/conv") == 0 ||
       std::strcmp(layer_name, "m2/cv2/conv") == 0 ||
       std::getenv("NNTR_Q8_PROBE") != nullptr)) {
    float smin = src[0], smax = src[0];
    for (size_t i = 1; i < nelem; ++i) {
      smin = std::min(smin, static_cast<float>(src[i]));
      smax = std::max(smax, static_cast<float>(src[i]));
    }
    float bmin = 0.f, bmax = 0.f;
    if (has_bias) {
      bmin = bmax = static_cast<float>(bias[0]);
      for (int c = 1; c < out_ch; ++c) {
        bmin = std::min(bmin, static_cast<float>(bias[c]));
        bmax = std::max(bmax, static_cast<float>(bias[c]));
      }
    }
    std::cerr << "[Q8_PROBE] " << (layer_name ? layer_name : "?")
              << " FUSE_INPUT nelem=" << nelem << " amax_pre=" << amax_pre
              << " amax_post=" << amax_post << " d_pre=" << d_pre
              << " d_post=" << d_post << " scratch_min=" << smin
              << " scratch_max=" << smax << " bias_min=" << bmin
              << " bias_max=" << bmax << " has_bias=" << has_bias << std::endl;
  }

  const _FP16 d_post_h = static_cast<_FP16>(d_post);
  uint16_t d_post_u16;
  std::memcpy(&d_post_u16, &d_post_h, 2);
  std::memcpy(dst, &d_post_u16, sizeof(uint16_t));

  // Build 256-entry LUT: qi -> qo where qo = round(silu(qi*d_pre) / d_post).
  int8_t lut[256];
  for (int qi = -128; qi <= 127; ++qi) {
    const float x = qi * d_pre;
    const float y = x / (1.0f + std::exp(-x));
    float qo = std::round(y / d_post);
    if (qo > 127.0f)
      qo = 127.0f;
    if (qo < -127.0f)
      qo = -127.0f;
    lut[(uint8_t)qi] = static_cast<int8_t>(qo);
  }

  // Pass 2: quantize pre-SiLU values via LUT to tensor-wise Q8_0 quants.
  const float id_pre = 1.0f / d_pre;
  int8_t *qs = reinterpret_cast<int8_t *>(dst + sizeof(uint16_t));
  tm.parallel_for(0, loops, [&](size_t idx) {
    unsigned int r0 = idx * chunk;
    unsigned int r1 = std::min(r0 + chunk, (unsigned int)n_spatial);
    for (unsigned int r = r0; r < r1; ++r) {
      const _FP16 *row = src + (size_t)r * out_ch;
      const size_t base = (size_t)r * out_ch;
      for (int c = 0; c < out_ch; ++c) {
        float v = static_cast<float>(row[c]) * id_pre;
        int qi = static_cast<int>(std::round(v));
        if (qi > 127)
          qi = 127;
        if (qi < -127)
          qi = -127;
        qs[base + c] = lut[(uint8_t)qi];
      }
    }
  });
  const size_t n_pad = (nelem + QK8_0 - 1) / QK8_0 * QK8_0;
  for (size_t i = nelem; i < n_pad; ++i)
    qs[i] = 0;
}

/**
 * @brief Quantize FP16 NHWC [n_spatial, in_ch] -> plain block_q8_0 [n_spatial][in_ch/32].
 *
 * NHWC input is already row-major (channel innermost): src[r * in_ch + c].
 * No transpose needed — each row r has in_ch contiguous FP16 channels.
 * Q8_0 requires in_ch % 32 == 0 (caller must check).
 * dst must hold n_spatial * (in_ch/32) * sizeof(block_q8_0) bytes.
 */
static inline void quantize_nhwc_q8_0_rows(const _FP16 *src, int n_spatial,
                                            int in_ch,
                                            ::nntrainer::block_q8_0 *dst) {
  const int nb = in_ch / 32;
  auto &tm = ThreadManager::Global();
  const unsigned int chunk = 512;
  const size_t loops = ((size_t)n_spatial + chunk - 1) / chunk;
  tm.parallel_for(0, loops, [=](size_t idx) {
    unsigned r0 = (unsigned)idx * chunk;
    unsigned r1 = std::min(r0 + chunk, (unsigned)n_spatial);
    for (unsigned r = r0; r < r1; ++r) {
      const _FP16 *row = src + (size_t)r * in_ch;
      for (int b = 0; b < nb; ++b) {
        const _FP16 *blk = row + b * 32;
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) {
          float v = std::abs(static_cast<float>(blk[j]));
          if (v > amax) amax = v;
        }
        const float d = amax / 127.f;
        const float id = d ? 1.f / d : 0.f;
        _FP16 d_h = static_cast<_FP16>(d);
        uint16_t d_u16;
        std::memcpy(&d_u16, &d_h, 2);
        ::nntrainer::block_q8_0 &out_blk = dst[(size_t)r * nb + b];
        out_blk.d = d_u16;
        for (int j = 0; j < 32; ++j)
          out_blk.qs[j] = (int8_t)std::roundf(static_cast<float>(blk[j]) * id);
      }
    }
  });
}

/**
 * @brief Quantize FP16 NHWC [n_spatial, in_ch] directly into the
 *        block_q8_0x4 (4-row interleaved) layout Q8_0_Tensor::dot_prepacked_x4
 *        consumes, with NO intermediate plain block_q8_0 pass and NO separate
 *        interleave copy (fuses what used to be quantize + Q8_0_Tensor::dot's
 *        internal repack into a single write).
 *
 * NHWC input is row-major (channel innermost): src[r * in_ch + c]. Output
 * layout matches transpose_quantize_q8_0x4_act / Q8_0_Tensor::dot's QA buffer:
 * M4 = n_spatial/4 groups of block_q8_0x4, followed by (n_spatial%4) plain
 * block_q8_0 rows for the GEMV tail. dst must hold the same total bytes as
 * n_spatial * (in_ch/32) plain block_q8_0 (136B per 4-row group == 4*34B).
 */
static inline void quantize_nhwc_q8_0x4_rows(const _FP16 *src, int n_spatial,
                                              int in_ch, void *dst) {
  struct block_q8_0 {
    uint16_t d;
    int8_t qs[32];
  };
  struct block_q8_0x4 {
    uint16_t d[4];
    int8_t qs[128];
  };
  const int qk = 32;
  const int nb = in_ch / qk;
  const int M4 = n_spatial / 4;
  const int rem = n_spatial % 4;
  block_q8_0x4 *y4 = static_cast<block_q8_0x4 *>(dst);
  const size_t qa_4_rows_size = sizeof(block_q8_0x4) * nb;

  auto &tm = ThreadManager::Global();
  const unsigned int chunk = 256; // groups of 4 rows per task
  const size_t loops = (M4 + chunk - 1) / chunk;
  tm.parallel_for(0, loops, [=](size_t idx) {
    unsigned int g0 = idx * chunk;
    unsigned int g1 = std::min(g0 + chunk, (unsigned int)M4);
    for (unsigned int g = g0; g < g1; ++g) {
      unsigned int r0 = g * 4;
      for (int b = 0; b < nb; ++b) {
        block_q8_0x4 &dst_b = y4[g * nb + b];
        for (unsigned int row = 0; row < 4; ++row) {
          unsigned int r = r0 + row;
          const _FP16 *rowp = src + (size_t)r * in_ch + b * qk;
          float amax = 0.0f;
          for (int j = 0; j < qk; ++j) {
            float val = std::abs(static_cast<float>(rowp[j]));
            if (val > amax)
              amax = val;
          }
          const float d = amax / ((1 << 7) - 1);
          const float id = d ? 1.0f / d : 0.0f;
          _FP16 d_half = static_cast<_FP16>(d);
          uint16_t d_u16;
          std::memcpy(&d_u16, &d_half, 2);
          dst_b.d[row] = d_u16;
          for (int j = 0; j < qk; ++j) {
            float x0 = static_cast<float>(rowp[j]) * id;
            // qs[32*chunk + 8*row + lane], chunk = j/8, lane = j%8
            dst_b.qs[32 * (j / 8) + 8 * row + (j % 8)] =
              static_cast<int8_t>(std::roundf(x0));
          }
        }
      }
    }
  });

  if (rem > 0) {
    block_q8_0 *yrem = reinterpret_cast<block_q8_0 *>(
      reinterpret_cast<char *>(dst) + (size_t)M4 * qa_4_rows_size);
    for (int i = 0; i < rem; ++i) {
      unsigned int r = M4 * 4 + i;
      const _FP16 *rowp = src + (size_t)r * in_ch;
      for (int b = 0; b < nb; ++b) {
        const _FP16 *blk = rowp + b * qk;
        float amax = 0.0f;
        for (int j = 0; j < qk; ++j) {
          float val = std::abs(static_cast<float>(blk[j]));
          if (val > amax)
            amax = val;
        }
        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;
        _FP16 d_half = static_cast<_FP16>(d);
        uint16_t d_u16;
        std::memcpy(&d_u16, &d_half, 2);
        yrem[i * nb + b].d = d_u16;
        for (int j = 0; j < qk; ++j)
          yrem[i * nb + b].qs[j] =
            static_cast<int8_t>(std::roundf(static_cast<float>(blk[j]) * id));
      }
    }
  }
}

/**
 * @brief Transpose-and-quantize FP16 NCHW [in_ch, owoh] -> Q8_0 [owoh, in_ch]
 *        in a single fused pass (no intermediate transpose copy).
 *
 * Each output row r (a spatial position) is quantized per 32-channel block:
 * block (r, b) covers channels [b*32, b*32+32). The FP16 source is NCHW
 * (channel-major), so channel c at position r lives at src[c*owoh + r]. This
 * gathers a 32-wide channel run with a strided read and writes a packed
 * block_q8_0 (fp16 scale + 32 int8). Parallelized over spatial positions.
 *
 * dst must hold (owoh * in_ch/32) block_q8_0 = owoh*in_ch/32*34 bytes, laid out
 * row-major as [owoh][in_ch/32] blocks — exactly the [M,K] block_q8_0 layout
 * Q8_0_Tensor::dot / the indirect GEMM consumes (M=owoh, K=in_ch).
 */
static inline void transpose_quantize_q8_0_act(const _FP16 *src, int in_ch,
                                                int owoh, void *dst) {
  struct block_q8_0 {
    uint16_t d;
    int8_t qs[32];
  };
  block_q8_0 *y = static_cast<block_q8_0 *>(dst);
  const int qk = 32;
  const int nb = in_ch / qk;

  auto &tm = ThreadManager::Global();
  const unsigned int chunk = 1024;
  const size_t loops = (owoh + chunk - 1) / chunk;
  tm.parallel_for(0, loops, [=](size_t idx) {
    unsigned int r0 = idx * chunk;
    unsigned int r1 = std::min(r0 + chunk, (unsigned int)owoh);
    for (unsigned int r = r0; r < r1; ++r) {
      for (int b = 0; b < nb; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < qk; ++j) {
          int c = b * qk + j;
          float val =
            std::abs(static_cast<float>(src[c * owoh + r]));
          if (val > amax)
            amax = val;
        }
        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;
        _FP16 d_half = static_cast<_FP16>(d);
        uint16_t d_u16;
        std::memcpy(&d_u16, &d_half, 2);
        y[r * nb + b].d = d_u16;
        for (int j = 0; j < qk; ++j) {
          int c = b * qk + j;
          float x0 = static_cast<float>(src[c * owoh + r]) * id;
          y[r * nb + b].qs[j] = std::roundf(x0);
        }
      }
    }
  });
}

/**
 * @brief Fused transpose + quantize FP16 NCHW [in_ch, owoh] directly into the
 *        block_q8_0x4 (4-row interleaved) layout the SMMLA GEMM consumes, with
 *        NO intermediate plain-block pass and NO separate interleave copy.
 *
 * Outputs (M4 = owoh/4) groups of 4 rows; each group packs nb=in_ch/32
 * block_q8_0x4. block_q8_0x4.qs[128] layout = qs[32*j + 8*row + lane],
 * j=8-element chunk (0..3), row=0..3 (matches __ggml_quantize_mat_q8_0_4x8).
 * Remainder (owoh % 4) rows packed as plain block_q8_0 afterward for the GEMV
 * tail. dst must hold the block_q8_0x4 region followed by the remainder
 * block_q8_0 region (same total as Q8_0_Tensor::dot's QA buffer).
 */
static inline void
transpose_quantize_q8_0x4_act(const _FP16 *src, int in_ch, int owoh,
                               void *dst) {
  struct block_q8_0 {
    uint16_t d;
    int8_t qs[32];
  };
  struct block_q8_0x4 {
    uint16_t d[4];
    int8_t qs[128];
  };
  const int qk = 32;
  const int nb = in_ch / qk;
  const int M4 = owoh / 4;
  const int rem = owoh % 4;
  block_q8_0x4 *y4 = static_cast<block_q8_0x4 *>(dst);
  const size_t qa_4_rows_size = sizeof(block_q8_0x4) * nb;

  auto &tm = ThreadManager::Global();
  const unsigned int chunk = 256; // groups of 4 rows per task
  const size_t loops = (M4 + chunk - 1) / chunk;
  tm.parallel_for(0, loops, [=](size_t idx) {
    unsigned int g0 = idx * chunk;
    unsigned int g1 = std::min(g0 + chunk, (unsigned int)M4);
    for (unsigned int g = g0; g < g1; ++g) {
      unsigned int r0 = g * 4;
      for (int b = 0; b < nb; ++b) {
        block_q8_0x4 &dst_b = y4[g * nb + b];
        for (unsigned int row = 0; row < 4; ++row) {
          unsigned int r = r0 + row;
          float amax = 0.0f;
          for (int j = 0; j < qk; ++j) {
            int c = b * qk + j;
            float val = std::abs(static_cast<float>(src[c * owoh + r]));
            if (val > amax)
              amax = val;
          }
          const float d = amax / ((1 << 7) - 1);
          const float id = d ? 1.0f / d : 0.0f;
          _FP16 d_half = static_cast<_FP16>(d);
          uint16_t d_u16;
          std::memcpy(&d_u16, &d_half, 2);
          dst_b.d[row] = d_u16;
          for (int j = 0; j < qk; ++j) {
            int c = b * qk + j;
            float x0 = static_cast<float>(src[c * owoh + r]) * id;
            // qs[32*chunk + 8*row + lane], chunk = j/8, lane = j%8
            dst_b.qs[32 * (j / 8) + 8 * row + (j % 8)] =
              static_cast<int8_t>(std::roundf(x0));
          }
        }
      }
    }
  });

  // Remainder rows (owoh % 4): plain block_q8_0 for the GEMV tail.
  if (rem > 0) {
    block_q8_0 *yrem =
      reinterpret_cast<block_q8_0 *>(reinterpret_cast<char *>(dst) +
                                     (size_t)M4 * qa_4_rows_size);
    const unsigned int rchunk = 1024;
    const size_t rloops = (rem + rchunk - 1) / rchunk;
    tm.parallel_for(0, rloops, [=](size_t idx) {
      unsigned int i0 = idx * rchunk;
      unsigned int i1 = std::min(i0 + rchunk, (unsigned int)rem);
      for (unsigned int i = i0; i < i1; ++i) {
        unsigned int r = M4 * 4 + i;
        for (int b = 0; b < nb; ++b) {
          float amax = 0.0f;
          for (int j = 0; j < qk; ++j) {
            int c = b * qk + j;
            float val = std::abs(static_cast<float>(src[c * owoh + r]));
            if (val > amax)
              amax = val;
          }
          const float d = amax / ((1 << 7) - 1);
          const float id = d ? 1.0f / d : 0.0f;
          _FP16 d_half = static_cast<_FP16>(d);
          uint16_t d_u16;
          std::memcpy(&d_u16, &d_half, 2);
          yrem[i * nb + b].d = d_u16;
          for (int j = 0; j < qk; ++j) {
            int c = b * qk + j;
            float x0 = static_cast<float>(src[c * owoh + r]) * id;
            yrem[i * nb + b].qs[j] = static_cast<int8_t>(std::roundf(x0));
          }
        }
      }
    });
  }
}
#endif

/**
 * @brief     reconstruct image data from 2d column matrix
 *
 * @param[in] kdim kernel dimesion for define number of row
 * @param[in] padding padding information
 * @param[in] mstride stride value : x, y direction
 * @param[in] dilation kernel dilation factor : x, y each
 * @param[out] image image tensor to put
 */
static void col2im(const Tensor &col_matrix, const TensorDim &kdim,
                   const std::array<unsigned, 4> &padding,
                   const std::array<props::Stride, CONV2D_DIM> &mstride,
                   const std::array<props::Dilation, CONV2D_DIM> &dilation,
                   Tensor &image) {

  auto pt = padding[0];
  auto pb = padding[1];
  auto pl = padding[2];
  auto pr = padding[3];

  unsigned k_height = kdim.height();
  unsigned k_width = kdim.width();

  /// effective kernel height considering dilation
  unsigned eff_k_height = (k_height - 1) * dilation[0] + 1;
  /// effective kernel width considering dilation
  unsigned eff_k_width = (k_width - 1) * dilation[1] + 1;

  unsigned im_channel = image.channel();
  int im_height = image.height();
  int im_width = image.width();

  unsigned hstride = mstride[0];
  unsigned wstride = mstride[1];

  unsigned hdilation = dilation[0];
  unsigned wdilation = dilation[1];

  /// image considering padding
  unsigned im_eff_height = im_height + pt + pb;
  unsigned im_eff_width = im_width + pl + pr;
  image.setZero();

  int h_stride_end = im_eff_height - eff_k_height - pt;
  int w_stride_end = im_eff_width - eff_k_width - pl;

  /** @todo We need to implement way to use this kind of function to work inside
   * of Tensor. Then we could remove to access the getData or getValue which has
   * dependecy of data type.
   */
  auto apply_data = [&]<typename T>(T *val) {
    unsigned col_w = 0;
    for (int hs = -(int)pt; hs <= h_stride_end; hs += hstride) {
      for (int ws = -(int)pl; ws <= w_stride_end; ws += wstride) {
        unsigned col_h = 0;
        int patch_height_end = hs + eff_k_height;
        int patch_width_end = ws + eff_k_width;
        for (unsigned c = 0; c < im_channel; c++) {
          for (int h = hs; h < patch_height_end; h += hdilation) {
            if (h < 0 || im_height <= h) {
              col_h += k_width;
              continue;
            }
            for (int w = ws; w < patch_width_end; w += wdilation) {
              if (w < 0 || im_width <= w) {
                col_h++;
                continue;
              }

              val = image.getAddress<T>(0, c, h, w);
              *val += col_matrix.getValue<T>(0, 0, col_h, col_w);
              col_h++;
            }
          }
        }
        col_w++;
      }
    }
  };

  if (image.getDataType() == nntrainer::Tdatatype::FP32) {
    float val;
    apply_data(&val);
  }
#ifdef ENABLE_FP16
  else if (image.getDataType() == nntrainer::Tdatatype::FP16) {
    _FP16 val;
    apply_data(&val);
  }
#endif
  else {
    throw std::runtime_error("Conv2D::calcDerivative: Not supported datatype " +
                             std::to_string(static_cast<int>(image.getDataType())));
  }
}

/**
 * @brief     reform the data to 2d matrix
 * a region is sampled considering @a padding, @a mstride of unit @a kdim
 * Each region is mapped to one column,
 * if channel mode, kernel channel is considered part of kernel feature
 * if not, kernel channel is consider part of output dimension
 *
 * @param[in] in input data
 * @param[in] kdim kernel dimesion for define number of row
 * @param[in] padding padding information
 * @param[in] mstride stride value : x, y direction
 * @param[in] dilation kernel dilation factor : x, y each
 * @param[out] out out tensor, padding set each time for now
 * @note if out is initialized tensor, setting padding is skipped.
 */
static void im2col(const Tensor &in, const TensorDim &kdim,
                   const std::array<unsigned int, 4> &padding,
                   const std::array<props::Stride, CONV2D_DIM> &mstride,
                   const std::array<props::Dilation, CONV2D_DIM> &dilation,
                   Tensor &out) {
  /// for channel last mode, this is deprecated for now, leaving here on
  /// purpose.
  /** @code
  //   ================ initialize part ====================
  //   out_height -= 2;
  //   out =
  //     Tensor(k_height * k_width, in.channel() * (out_height) *
  //     (out_width));
  //   unsigned int im_w = 0;
  //   ================ loop part ====================
  //   if (eff_k_height > height || eff_k_width > width)
  //     throw std::runtime_error("Kernel shape bigger than input shape");

  //   for (unsigned int c = 0; c < channel; ++c) {
  //     for (unsigned int hs = 0; hs <= height - eff_k_height; hs +=
  //     mstride[0]) {
  //       for (unsigned int ws = 0; ws <= width - eff_k_width; ws +=
  //       mstride[1]) {
  //         unsigned int im_h = 0;
  //         unsigned int patch_height_end = eff_k_height + hs;
  //         unsigned int patch_width_end = eff_k_width + ws;

  //         for (unsigned int h = hs; h < patch_height_end; h += dilation[0]) {
  //           if (h < ph || in_height + ph <= h) {
  //             im_h += k_width;
  //             continue;
  //           }

  //           for (unsigned int w = ws; w < patch_width_end; w += dilation[1])
  //           {
  //             if (w < pw || in_width + pw <= w) {
  //               im_h++;
  //               continue;
  //             }

  //             float val = in.getValue(0, c, h - ph, w - pw);
  //             out.setValue(0, 0, im_h, im_w, val);
  //             im_h++;
  //           }
  //         }
  //         im_w++;
  //       }
  //     }
  //   }
  */

  auto pt = padding[0];
  auto pb = padding[1];
  auto pl = padding[2];
  auto pr = padding[3];

  unsigned int channel = in.channel();
  int in_height = in.height();
  int in_width = in.width();
  unsigned int height = in_height + pt + pb;
  unsigned int width = in_width + pl + pr;
  unsigned int k_height = kdim.height();
  unsigned int k_width = kdim.width();

  /// effective kernel height considering dilation
  unsigned int eff_k_height = (k_height - 1) * dilation[0] + 1;
  /// effective kernel width considering dilation
  unsigned int eff_k_width = (k_width - 1) * dilation[1] + 1;

  unsigned int out_height = (height - eff_k_height) / mstride[0] + 1;
  unsigned int out_width = (width - eff_k_width) / mstride[1] + 1;

  out.reshape(
    TensorDim({out_height * out_width, in.channel() * k_height * k_width},
              in.getTensorType()));
  // float *out_data = out.getData();

  auto apply_data = [&]<typename T>(T *out_data) {
    int w_stride_end = width - eff_k_width - pl;

    /// get a patch, size of kernel
    /// hs is height_strided, ws is width_strided
    unsigned int owidth = out.width();
    const int hstride = mstride[0];

    /// Raw contiguous-NCHW input base + inner strides. `in` is a (batch-sliced)
    /// contiguous NCHW tensor, so element (0,c,h,w) lives at
    /// in_base + c*inHW + h*inW + w. Hoisting these out of the inner loop turns
    /// the old per-element in.getValue() -- which re-fetched the data pointer
    /// (getData<T>()) and recomputed a 4-D linear offset (getIndex(): a format
    /// branch + 4 muls + 3 adds) plus a per-element padding branch -- into one
    /// contiguous run copy. im2col is pure data movement; the previous form ran
    /// far below memory bandwidth because that overhead dominated the 4-byte
    /// move. Padding columns are left untouched (the caller zeroes the buffer
    /// once before im2col), so the fast path only writes the valid span.
    const T *in_base = in.getData<T>();
    const size_t inW = (size_t)in_width;
    const size_t inHW = (size_t)in_height * (size_t)in_width;
    const bool unit_dil =
      ((unsigned int)dilation[0] == 1 && (unsigned int)dilation[1] == 1);
    const bool is_nhwc =
      (in.getFormat() == ml::train::TensorDim::Format::NHWC);

    /// Each output row (oh) writes a disjoint band of `out_width` columns
    /// (rows [oh*out_width, (oh+1)*out_width) of the [OH*OW, CRS] matrix), so
    /// the per-row work is independent and bit-identical when parallelized.
    /// hs and base_im_w are derived from oh directly (no sequential carry).
    auto fill_row = [&](size_t oh) {
      int hs = -(int)pt + (int)oh * hstride;
      unsigned int base_im_w = (unsigned int)oh * out_width;
      unsigned int base_im_h = 0;
      int patch_height_end = eff_k_height + hs;
      /// map the patch to a single line looping through channel
      for (unsigned int c = 0; c < channel; ++c) {
        for (int h = hs; h < patch_height_end; h += dilation[0]) {
          if (h < 0 || in_height <= h) {
            base_im_h += k_width;
            continue;
          }

          if (unit_dil) {
            /// Fast path (dilation == 1): for each output column position the
            /// kernel-width window maps a contiguous source run
            /// in_row[w_lo,w_hi) to a contiguous dest run; copy it in one
            /// memcpy.
            const T *in_row = in_base + (size_t)c * inHW + (size_t)h * inW;
            unsigned int im_w = base_im_w;
            for (int ws = -(int)pl; ws <= w_stride_end; ws += mstride[1]) {
              int w_lo = ws < 0 ? 0 : ws;
              int w_hi = ws + (int)k_width;
              if (w_hi > in_width)
                w_hi = in_width;
              if (w_hi > w_lo) {
                T *dst =
                  out_data + (size_t)im_w * owidth + base_im_h + (w_lo - ws);
                if (!is_nhwc) {
                  const T *in_row =
                    in_base + (size_t)c * inHW + (size_t)h * inW;
                  std::memcpy(dst, in_row + w_lo,
                              (size_t)(w_hi - w_lo) * sizeof(T));
                } else {
                  /// NHWC: channel is innermost, so the w-run is NOT
                  /// contiguous in source. Gather per (w, channel) element.
                  for (int w = w_lo; w < w_hi; ++w) {
                    dst[w - w_lo] =
                      in_base[((size_t)h * inW + (size_t)w) * channel + c];
                  }
                }
              }
              im_w++;
            }
          } else {
            /// General (dilated) path: original scalar gather, but via the
            /// hoisted base pointer (no per-element getData()/getIndex()).
            unsigned int im_w = base_im_w;
            for (int ws = -(int)pl; ws <= w_stride_end; ws += mstride[1]) {
              unsigned int im_h = base_im_h;
              int patch_width_end = eff_k_width + ws;

              for (int w = ws; w < patch_width_end; w += dilation[1]) {
                if (w < 0 || in_width <= w) {
                  im_h++;
                  continue;
                }
                if (!is_nhwc) {
                  out_data[(size_t)im_w * owidth + im_h] =
                    in_base[(size_t)c * inHW + (size_t)h * inW + w];
                } else {
                  out_data[(size_t)im_w * owidth + im_h] =
                    in_base[((size_t)h * inW + (size_t)w) * channel + c];
                }
                im_h++;
              }
              im_w++;
            }
          }
          base_im_h += k_width;
        }
      }
    };

    ThreadManager::Global().parallel_for(0, out_height, fill_row);
  };

  if (out.getDataType() == nntrainer::Tdatatype::FP32) {
    float *out_data = out.getData<float>();
    apply_data(out_data);
  }
#ifdef ENABLE_FP16
  else if (out.getDataType() == nntrainer::Tdatatype::FP16) {
    _FP16 *out_data = out.getData<_FP16>();
    apply_data(out_data);
  }
#endif
  else {
    throw std::runtime_error("Conv2D::im2col: Not supported datatype " +
                             std::to_string(static_cast<int>(out.getDataType())));
  }
}
} // namespace

enum ConvParams { weight, bias, im2col_scratch, qgemm_scratch, q8act_scratch, dw_in_scratch, dw_out_scratch };

Conv2DLayer::Conv2DLayer(
  const std::array<unsigned int, CONV2D_DIM * 2> &padding_) :
  LayerImpl(),
  padding(padding_),
  conv_props(props::FilterSize(), std::array<props::KernelSize, CONV2D_DIM>(),
             std::array<props::Stride, CONV2D_DIM>(), props::Padding2D(),
             std::array<props::Dilation, CONV2D_DIM>(), props::ConvGroups(),
             props::FusedActivation()) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

void Conv2DLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Convolution layer takes only one input";

  const TensorDim &in_dim = context.getInputDimensions()[0];

  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<props::BiasDecay>(*layer_impl_props);
  auto &bias_initializer = std::get<props::BiasInitializer>(*layer_impl_props);
  auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);

  unsigned int filter_size = std::get<props::FilterSize>(conv_props);
  auto &kernel_size =
    std::get<std::array<props::KernelSize, CONV2D_DIM>>(conv_props);
  auto &stride = std::get<std::array<props::Stride, CONV2D_DIM>>(conv_props);
  auto &dilation =
    std::get<std::array<props::Dilation, CONV2D_DIM>>(conv_props);

  auto &groups_prop = std::get<props::ConvGroups>(conv_props);
  unsigned int groups = groups_prop.empty() ? 1 : groups_prop.get();
  NNTR_THROW_IF(in_dim.channel() % groups != 0 || filter_size % groups != 0,
                std::invalid_argument)
    << "[Conv2D] input channels (" << in_dim.channel() << ") and filters ("
    << filter_size << ") must both be divisible by groups (" << groups << ")";

  auto in_t_type = in_dim.getTensorType();
  in_t_type.data_type = context.getWeightDataType();

  // Grouped/depthwise convolutions and convs whose output channel count is not
  // a multiple of 32 are not supported by the 2-D Q4_0_Tensor weight layout used
  // by the quantized GEMM paths. Keep their weights as FP32 so the standard
  // kernel runs, and let the Q8_0-activation bridge handle activation precision.
  if ((groups > 1 || filter_size % 32 != 0) &&
      (in_t_type.data_type == nntrainer::Tdatatype::Q4_0 ||
       in_t_type.data_type == nntrainer::Tdatatype::QINT4)) {
    in_t_type.data_type = nntrainer::Tdatatype::FP32;
  }

  // A quantized (Q4_0/QINT4) 1x1 conv is computed as a matmul, so its filter is
  // stored as a [in_ch, out_ch] (K, N) weight that the quantized GEMM consumes
  // directly (no im2col-style [out_ch, CRS] squeeze). Non-quantized or larger
  // kernels keep the standard [out_ch, in_ch/groups, kh, kw] layout.
  const bool quant_matmul_filter =
    (in_t_type.data_type == nntrainer::Tdatatype::Q4_0 ||
     in_t_type.data_type == nntrainer::Tdatatype::QINT4) &&
    groups == 1;

  // Real conv kernel geometry — used for padding/output-size computation even
  // when the quantized weight is stored flattened as [CRS, out_ch].
  TensorDim real_kernel_dim(filter_size, in_dim.channel() / groups,
                            kernel_size[0], kernel_size[1], in_t_type);
  // A quantized (groups==1) conv stores its filter as a [CRS, out_ch] matmul
  // weight (CRS = in_ch*kh*kw), consumed by the quantized GEMM after im2col.
  TensorDim kernel_dim = quant_matmul_filter
                           ? TensorDim(1, 1,
                                       in_dim.channel() * kernel_size[0].get() *
                                         kernel_size[1].get(),
                                       filter_size, in_t_type)
                           : real_kernel_dim;

  // Bias is never quantized (no dequantizer for add). Follow activation dtype
  // for FP32/FP16 graphs so a Q4_0/QINT4 weight does not force a Q4_0 bias.
  // For Q8_0-activation graphs keep bias as FP16; the fused quant conv epilogue
  // still operates on FP16 scratch and expects FP16 bias values.
  auto bias_t_type = in_dim.getTensorType();
  bias_t_type.data_type =
    context.getActivationDataType() == nntrainer::Tdatatype::Q8_0
      ? nntrainer::Tdatatype::FP16
      : context.getActivationDataType();
  TensorDim bias_dim = TensorDim(1, filter_size, 1, 1, bias_t_type);

  padding = std::get<props::Padding2D>(conv_props)
              .compute(in_dim, real_kernel_dim, {stride[0], stride[1]},
                       {dilation[0], dilation[1]});

  wt_idx[ConvParams::weight] = context.requestWeight(
    kernel_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "filter", true, 0);

  if (disable_bias.empty() || disable_bias.get() == false) {
    wt_idx[ConvParams::bias] =
      context.requestWeight(bias_dim, bias_initializer, WeightRegularizer::NONE,
                            1.0f, bias_decay, "bias", true, 0);
  }

  // this output_dim must be the same with dimension of hidden
  unsigned int eff_in_height = in_dim.height() + padding[0] + padding[1];
  unsigned int eff_in_width = in_dim.width() + padding[2] + padding[3];

  unsigned int eff_k_height = (kernel_size[0] - 1) * dilation[0] + 1;
  unsigned int eff_k_width = (kernel_size[1] - 1) * dilation[1] + 1;

  TensorDim out_dim;
  out_dim.batch(in_dim.batch());
  out_dim.channel(filter_size);
  out_dim.height((eff_in_height - eff_k_height) / stride[0] + 1);
  out_dim.width((eff_in_width - eff_k_width) / stride[1] + 1);

  // Output activation dtype follows the model activation dtype, not the input
  // dtype. For FP16 activation graphs this is identical to the input; for a
  // first-class Q8_0 activation graph it makes the conv emit a real Q8_0 tensor.
  auto out_t_type = in_dim.getTensorType();
  out_t_type.data_type = context.getActivationDataType();

  // YOLOv11 Detect head cls branch final 1x1 conv outputs a single-channel
  // class logit. Quantizing this to tensor-wise Q8_0 collapses the small logit
  // values; keep it in full precision so sigmoid/confidence stays accurate.
  const std::string &layer_name = context.getName();
  if (filter_size == 1 && layer_name.find("cv3_2/conv") != std::string::npos) {
    out_t_type.data_type = nntrainer::Tdatatype::FP32;
  }

  out_dim.setTensorType(out_t_type);

  context.setOutputDimensions({out_dim});

  NNTR_THROW_IF(eff_in_height < kernel_size[0] || eff_in_width < kernel_size[1],
                std::invalid_argument)
    << "Failed to initialize: in size + padding is smaller than effective "
       "kernel";

  unsigned int IM = std::numeric_limits<int>::max();

  NNTR_THROW_IF(eff_in_height - padding[0] - kernel_size[0] > IM ||
                  eff_in_width - padding[2] - kernel_size[1] > IM,
                std::invalid_argument)
    << "Failed to initialize: Calculated patch end is over int max";

  const bool nhwc_layout =
    (in_dim.getFormat() == ml::train::TensorDim::Format::NHWC);
  std::cerr << "[CONV_FIN] " << context.getName()
            << " nhwc=" << nhwc_layout
            << " quant_filt=" << quant_matmul_filter
            << " act_dtype=" << (int)context.getActivationDataType()
            << " out_dtype=" << (int)out_t_type.data_type << std::endl;

  // Forward scratch (groups==1 path only): the im2col column buffer and the
  // quantized-GEMM output are otherwise heap-allocated on every forwarding()
  // call. Request them once here (planned into the shared activation arena,
  // FORWARD_FUNC_LIFESPAN) and reuse — no per-forward malloc/free churn. The
  // grouped path keeps its local buffer. NOTE: the im2col buffer must still be
  // re-zeroed each forward (im2col skips padding positions and the arena is
  // reused across layers), so this saves the allocation, not the zeroing.
  wt_idx[ConvParams::im2col_scratch] = std::numeric_limits<unsigned int>::max();
  wt_idx[ConvParams::qgemm_scratch] = std::numeric_limits<unsigned int>::max();
  wt_idx[ConvParams::q8act_scratch] = std::numeric_limits<unsigned int>::max();
  wt_idx[ConvParams::dw_in_scratch] = std::numeric_limits<unsigned int>::max();
  wt_idx[ConvParams::dw_out_scratch] = std::numeric_limits<unsigned int>::max();
  if (groups == 1) {
    auto scratch_type = in_dim.getTensorType();
    const unsigned int owoh = out_dim.width() * out_dim.height();
    const bool is_1x1_s1 = kernel_size[0].get() == 1 &&
                           kernel_size[1].get() == 1 && stride[0].get() == 1 &&
                           stride[1].get() == 1;
    // im2col column buffer [batch, 1, CRS, OH*OW]. Unused by the quant paths
    // that never materialize a col buffer: the 1x1 path (im2col is an identity
    // handled by an input transpose) and, where the fused backend op exists,
    // the non-1x1 path (gather is fused into the q8_0 activation packing).
    const bool k_eligible_finalize = (in_dim.channel() % 32 == 0);
    const bool needs_im2col =
      !(quant_matmul_filter && (is_1x1_s1 ||
                                (NNTR_HAS_Q4_0_INDIRECT_CONV &&
                                 k_eligible_finalize)));
    if (needs_im2col) {
      // FP path or quant fallback: materialize the im2col column buffer
      // [batch, 1, CRS, OH*OW] once (planned into the activation arena). The
      // quant 1x1 path (identity input transpose) and the quant indirect path
      // (gather fused into the GEMM's q8_0 packing) never materialize a col
      // buffer, so they request no im2col_scratch here.
      TensorDim col_dim(in_dim.batch(), 1, real_kernel_dim.getFeatureLen(),
                        owoh, scratch_type);
      wt_idx[ConvParams::im2col_scratch] =
        context.requestTensor(col_dim, "im2col", Initializer::NONE, false,
                              TensorLifespan::MAX_LIFESPAN);
    }
    // Quantized-GEMM output scratch [batch, 1, OH*OW, out_ch].
    // The NHWC path may receive Q4_0/QINT4 weights at runtime even when
    // finalize() sees a different weight dtype (the app sets weight_dtype via
    // a per-layer property after the model is compiled/loaded). Always reserve
    // the scratch for layers that will use the quantized conv path: either
    // the weight dtype is already quantized, or the graph is NHWC / Q8_0-act.
    const bool needs_qgemm_scratch =
      quant_matmul_filter ||
      (groups == 1 &&
       (nhwc_layout ||
        context.getActivationDataType() == nntrainer::Tdatatype::Q8_0));
    if (needs_qgemm_scratch) {
      TensorDim tmp_dim(in_dim.batch(), 1, owoh, filter_size, scratch_type);
      wt_idx[ConvParams::qgemm_scratch] =
        context.requestTensor(tmp_dim, "qgemm_out", Initializer::NONE, false,
                              TensorLifespan::MAX_LIFESPAN);
      std::cerr << "[CONV_FIN] " << context.getName()
                << " requested qgemm idx="
                << wt_idx[ConvParams::qgemm_scratch] << std::endl;
    } else {
      std::cerr << "[CONV_FIN] " << context.getName()
                << " skipped qgemm scratch" << std::endl;
    }
    // Tensor-wise Q8_0 activation scratch for the NHWC W4A8/W4Q8 path.
    // This scratch stores an intermediate activation as tensor-wise Q8_0:
    //   uint16_t scale + int8_t qs[round_up(max_sp*in_ch, 32)]
    // It is used both for the first FP16-input conv (quantize once here) and
    // as a stable backing buffer when a Q8_0 input is fed to the indirect conv.
    // MAX_LIFESPAN is required: the shorter lifespan aliases activation memory
    // and corrupts live skip-connection tensors (see prior scratch lifespan bug).
    // Request it whenever the NHWC quantized path is possible, not just when
    // finalize's weight dtype is already Q4_0.
    if (nhwc_layout && groups == 1 && NNTR_HAS_Q4_0_INDIRECT_CONV) {
      const int in_ch_i = (int)in_dim.channel();
      const unsigned int max_sp =
        std::max(owoh, (unsigned int)(in_dim.height() * in_dim.width()));
      const size_t max_nelem = (size_t)max_sp * in_ch_i;
      const size_t q8_size =
        sizeof(uint16_t) + ((max_nelem + QK8_0 - 1) / QK8_0) * QK8_0;
      const unsigned int n_elems = static_cast<unsigned int>((q8_size + 1) / 2);
      TensorDim q8dim(1, 1, 1, n_elems, scratch_type);
      wt_idx[ConvParams::q8act_scratch] =
        context.requestTensor(q8dim, "q8act", Initializer::NONE, false,
                              TensorLifespan::MAX_LIFESPAN);
    }

  }

  // Depthwise scratch for NHWC Q8_0 -> FP16 bridge. True depthwise conv only
  // supports FP32/FP16 inputs; when activations are Q8_0, stage each batch
  // through FP16 scratch, run depthwise_conv2d_fp16, then quantize back.
  const bool is_true_depthwise =
    filter_size == groups && in_dim.channel() == groups && groups > 1;
  if (is_true_depthwise && nhwc_layout &&
      context.getActivationDataType() == nntrainer::Tdatatype::Q8_0) {
    auto fp16_type = in_dim.getTensorType();
    fp16_type.data_type = nntrainer::Tdatatype::FP16;
    TensorDim dw_in_dim(1, in_dim.channel(), in_dim.height(), in_dim.width(),
                        fp16_type);
    TensorDim dw_out_dim(1, out_dim.channel(), out_dim.height(),
                         out_dim.width(), fp16_type);
    wt_idx[ConvParams::dw_in_scratch] =
      context.requestTensor(dw_in_dim, "dw_in", Initializer::NONE, false,
                            TensorLifespan::MAX_LIFESPAN);
    wt_idx[ConvParams::dw_out_scratch] =
      context.requestTensor(dw_out_dim, "dw_out", Initializer::NONE, false,
                            TensorLifespan::MAX_LIFESPAN);
  }
}

void Conv2DLayer::forwarding(RunLayerContext &context, bool training) {
  int status = ML_ERROR_NONE;
  static volatile int __marker = 0;
  std::cerr << "[CONV_FWD] entering " << context.getName() << " marker=" << (__marker++) << std::endl;
  std::cerr << "[CONV_FWD] " << context.getName()
            << " dtype=" << (int)context.getInput(0).getDataType()
            << " fmt=" << (int)context.getInput(0).getFormat()
            << " in_ch=" << context.getInput(0).channel() << std::endl;
  std::cerr << "[CONV_FWD] " << context.getName()
            << " wt_idx_weight=" << wt_idx[ConvParams::weight] << std::endl;
  std::cerr << "[CONV_FWD] " << context.getName() << " stepA" << std::endl;
  std::cerr.flush();

  unsigned int filter_size = std::get<props::FilterSize>(conv_props);
  auto &stride = std::get<std::array<props::Stride, CONV2D_DIM>>(conv_props);
  auto &dilation =
    std::get<std::array<props::Dilation, CONV2D_DIM>>(conv_props);
  auto &kernel_size =
    std::get<std::array<props::KernelSize, CONV2D_DIM>>(conv_props);

  std::cerr << "[CONV_FWD] " << context.getName() << " stepB" << std::endl;
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  std::cerr << "[CONV_FWD] " << context.getName() << " stepC" << std::endl;
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  std::cerr << "[CONV_FWD] " << context.getName() << " stepD" << std::endl;

  // Force contiguous owned weight view; prevent a stale shared-data view (e.g.
  // after previous reshape/reuse from a different layer context) from being
  // dereferenced.
  Tensor filter_kernel = context.getWeight(wt_idx[ConvParams::weight]);
  std::cerr << "[CONV_FWD] " << context.getName() << " stepE" << std::endl;

  /** Calculate Convolution 2D
   *
   * This is the 2D Matrix Shape [ height ] x [ width ]
   *   . Height : filter_size
   *   . Width  : Input Channel * Kernel_size[0] * Kernel_size[1]
   *
   *                              imKernel
   *                        +------|------|------+
   *                        |------|------|------|
   * [filter_size (height)] |------|------|------|
   *                        |------|------|------|
   *                        +------|------|------+
   *                     [Input Channel * Kernel_size[0]
   *                       * Kernel_size[1] (width)]
   *
   *
   * After im2Col with channel_mode true (in : input)
   *
   * This is the 2D Matrix Shape [ height ] x [ width ]
   *   . Height : Input Channel * Kernel_size[0] * Kernel_size[1]
   *   . Width  : output_dim.height * output_dim.width
   *
   *                      +-|-|-|-|      |-|-|-|-+
   *   [Input Channel     | | | | |      | | | | |
   *   * Kernel_size[0]   |_|_|_|_|      |_|_|_|_|
   *  * Kenel_size[1]     | | | | | .... | | | | |
   *    (height)]         |_|_|_|_|      |_|_|_|_|
   *                      | | | | |      | | | | |
   *                      +_|_|_|_|      |_|_|_|_+
   *                     [ output_dim.height
   *                      * output_dim.width (width) ]
   *
   * Output Dimention
   *   -> [Channel ( = filter_size = output_dim.channel )]
   *       x [output_dim.height x output_dim.width]
   */
  const TensorDim &in_dim = input_.getDim();
  const TensorDim &out_dim = hidden_.getDim();
  const TensorDim &filter_dim = filter_kernel.getDim();
  auto &groups_prop = std::get<props::ConvGroups>(conv_props);
  unsigned int groups = groups_prop.empty() ? 1 : groups_prop.get();

  if (groups == 1) {
    // A quantized 1x1 conv stores its filter as a [in_ch, out_ch] matmul weight
    // (K, N). The quantized GEMM (dotQnK for Q4_0) takes the weight as the dot
    // *input* (activation is the receiver), so we keep that layout as-is and do
    // NOT squeeze it to [out_ch, CRS] like the FP32 path.
    const auto weight_dtype = filter_kernel.getDataType();
    const bool weight_is_quant = (weight_dtype == nntrainer::Tdatatype::Q4_0 ||
                                  weight_dtype == nntrainer::Tdatatype::QINT4);
    const unsigned int owoh = out_dim.width() * out_dim.height();
    const bool output_is_q8 =
      hidden_.getDataType() == nntrainer::Tdatatype::Q8_0;

    // The FP32 path below assumes the filter is 2-D [out_ch, CRS] for the dot.
    // The Q4_0 path keeps the [CRS, out_ch] matmul weight unchanged. Snapshot
    // the original real geometry before any reshape so im2col always sees the
    // correct kh/kw.
    const TensorDim real_filter_dim = filter_kernel.getDim();
    TensorDim filter_dim_squeezed{filter_kernel.batch(),
                                  filter_kernel.getDim().getFeatureLen()};
    filter_dim_squeezed.setTensorType(filter_kernel.getTensorType());
    if (!weight_is_quant) {
      filter_kernel.reshape(filter_dim_squeezed);
    }

    /**
     * Below sets the pad area values to zero
     * it is faster to do this way than seting selective area to zero
     */
    const bool is_1x1_s1 = kernel_size[0].get() == 1 &&
                           kernel_size[1].get() == 1 && stride[0].get() == 1 &&
                           stride[1].get() == 1;
    // Pre-allocated forward scratch (requested once in finalize). The im2col
    // column buffer is used by the FP32 path and the quant non-1x1 *fallback*;
    // the quant 1x1 path (identity input transpose) and the quant non-1x1 fused
    // path (gather folded into the GEMM) need no col buffer — finalize skips
    // im2col_scratch is materialized by finalize only for the FP/fallback path.
    // The quant 1x1 path (identity transpose) and the quant indirect path
    // (gather fused into the GEMM) requested no col buffer in finalize, so the
    // pointer stays null for them.
    const bool k_eligible_forward = (in_dim.channel() % 32 == 0);
    const bool use_im2col_scratch =
      !(weight_is_quant && (is_1x1_s1 ||
                            (NNTR_HAS_Q4_0_INDIRECT_CONV && k_eligible_forward)));
    Tensor *col_scratch =
      use_im2col_scratch
        ? &context.getTensor(wt_idx[ConvParams::im2col_scratch])
        : nullptr;
    Tensor *qgemm_scratch =
      weight_is_quant ? &context.getTensor(wt_idx[ConvParams::qgemm_scratch])
                      : nullptr;
    {
      const unsigned int qg_idx = wt_idx[ConvParams::qgemm_scratch];
      Tensor *raw_qgemm = (qg_idx != std::numeric_limits<unsigned int>::max())
                            ? &context.getTensor(qg_idx)
                            : nullptr;
      std::cerr << "[CONV_FWD] " << context.getName()
                << " col_ptr=" << col_scratch << " col_data="
                << (col_scratch ? col_scratch->getData() : nullptr)
                << " qgemm_idx=" << qg_idx << " qgemm_ptr=" << raw_qgemm
                << " qgemm_data="
                << (raw_qgemm ? raw_qgemm->getData() : nullptr) << std::endl;
    }
    if (col_scratch != nullptr) {
      col_scratch->setZero();
    }

    auto forwarding_job = [&](unsigned int s, unsigned int e, unsigned int pid,
                              void *user_data) {
      for (unsigned int b = s; b < e; ++b) {
        Tensor out = hidden_.getBatchSlice(b, 1);
        const TensorDim out_slice_orig = out.getDim();
        Tensor in_sub = input_.getBatchSlice(b, 1);

        if (weight_is_quant) {
          std::cerr << "[CONV_FWD] " << context.getName()
                    << " enter weight_is_quant branch" << std::endl;
          // Weight is stored as [CRS, out_ch]; im2col/indirect helpers synthesize
          // K=CRS from real kernel geometry, so the reshaped squeeze below is not
          // needed and would corrupt the geometry used by fallback im2col.
          if (in_sub.getFormat() == ml::train::TensorDim::Format::NHWC) {
            // NHWC channel-last quantized convolution for tensor-wise Q8_0
            // activation graphs (W4Q8) and FP16-activation graphs (W4A16).
            // The conv kernel always writes its FP16 result into qgemm_scratch;
            // when the output dtype is Q8_0 we run bias+SiLU on that scratch and
            // quantize to a single-scale tensor-wise Q8_0 output.
            Tensor tmp_fp16 = qgemm_scratch->getBatchSlice(b, 1);
            tmp_fp16.reshape(TensorDim(1, 1, owoh, filter_size,
                                       {ml::train::TensorDim::Format::NCHW,
                                        nntrainer::Tdatatype::FP16}));
            Tensor out_flat = output_is_q8 ? tmp_fp16 : out;
            if (!output_is_q8) {
              out_flat.reshape(TensorDim(1, 1, owoh, filter_size,
                                         {ml::train::TensorDim::Format::NCHW,
                                          out.getDataType()}));
            }

            const int in_ch_i = (int)in_dim.channel();
            const bool input_is_q8 =
              in_sub.getDataType() == nntrainer::Tdatatype::Q8_0;
            const bool k_eligible = (in_ch_i % 32 == 0);

            uint8_t *q8_scratch_base = nullptr;
            if (wt_idx[ConvParams::q8act_scratch] !=
                std::numeric_limits<unsigned int>::max()) {
              q8_scratch_base = reinterpret_cast<uint8_t *>(
                context.getTensor(wt_idx[ConvParams::q8act_scratch])
                  .getData());
            }

            std::cerr << "[CONV_FWD] " << context.getName()
                      << " weight_is_quant path is_1x1=" << is_1x1_s1
                      << " k_elig=" << k_eligible
                      << " input_is_q8=" << input_is_q8
                      << " output_is_q8=" << output_is_q8
                      << " q8_scratch_base="
                      << (q8_scratch_base ? "set" : "null") << std::endl;
            if (is_1x1_s1) {
#ifdef ENABLE_FP16
              if (k_eligible) {
                std::cerr << "[CONV_FWD] " << context.getName()
                          << " 1x1 k_eligible in_ch=" << in_ch_i
                          << " owoh=" << owoh << std::endl;
                // Build a [owoh, in_ch] tensor-wise Q8_0 view from the input.
                // For FP16 input we quantize once into the MAX_LIFESPAN scratch;
                // for Q8_0 input we reuse the already tensor-wise storage.
                TensorDim in_q8_dim({1, 1, owoh, (unsigned)in_ch_i},
                                    {ml::train::TensorDim::Format::NCHW,
                                     nntrainer::Tdatatype::Q8_0});
                Q8_0_Tensor *in_q8_ptr = nullptr;
                Q8_0_Tensor in_q8_view;
                if (input_is_q8) {
                  in_q8_view = Q8_0_Tensor(
                    in_q8_dim,
                    reinterpret_cast<uint8_t *>(in_sub.getData()) -
                      sizeof(uint16_t));
                  in_q8_ptr = &in_q8_view;
                } else if (q8_scratch_base != nullptr) {
                  quantize_nhwc_q8_0_tensor(in_sub.getData<_FP16>(), (int)owoh,
                                            in_ch_i, q8_scratch_base);
                  in_q8_view = Q8_0_Tensor(in_q8_dim, q8_scratch_base);
                  in_q8_ptr = &in_q8_view;
                }

                if (in_q8_ptr != nullptr) {
                  {
                    const int16_t *sd =
                      reinterpret_cast<const int16_t *>(in_q8_ptr->getData()) - 1;
                    uint16_t du;
                    std::memcpy(&du, sd, 2);
                    float sc = static_cast<float>(
                      *reinterpret_cast<const _FP16 *>(&du));
                    const int8_t *qq =
                      reinterpret_cast<const int8_t *>(in_q8_ptr->getData());
                    int mn = 127, mx = -128, z = 0;
                    size_t ne = (size_t)owoh * in_ch_i;
                    for (size_t i = 0; i < ne; ++i) {
                      int v = qq[i];
                      mn = std::min(mn, v);
                      mx = std::max(mx, v);
                      if (v == 0) ++z;
                    }
                    if (std::isnan(sc) || sc == 0.0f || z == ne ||
                        std::getenv("NNTR_Q8_PROBE")) {
                      std::cerr << "[Q8_PROBE] PRE_DOT " << context.getName()
                                << " in_scale=" << sc << " in_elems=" << ne
                                << " in_qmin=" << mn << " in_qmax=" << mx
                                << " zeros=" << z << std::endl;
                    }
                  }
                  std::cerr << "[CONV_FWD] " << context.getName()
                            << " about to dot in_q8_ptr in_data="
                            << in_q8_ptr->getData() << " out_data="
                            << out_flat.getData() << std::endl;
                  in_q8_ptr->dot(filter_kernel, out_flat, false, false);
                  std::cerr << "[CONV_FWD] " << context.getName()
                            << " returned from dot" << std::endl;
                  {
                    const _FP16 *s = out_flat.getData<_FP16>();
                    size_t ne = (size_t)owoh * filter_size;
                    float mn = static_cast<float>(s[0]);
                    float mx = mn;
                    size_t nan = 0;
                    for (size_t i = 0; i < ne; ++i) {
                      float v = static_cast<float>(s[i]);
                      if (std::isnan(v)) ++nan;
                      mn = std::min(mn, v);
                      mx = std::max(mx, v);
                    }
                    if (nan != 0 || std::getenv("NNTR_Q8_PROBE")) {
                      std::cerr << "[Q8_PROBE] POST_DOT " << context.getName()
                                << " scratch_min=" << mn << " scratch_max=" << mx
                                << " nans=" << nan << std::endl;
                    }
                  }
                } else if (input_is_q8) {
                  // TEMP: dequantize Q8_0 input to FP16 and use the proven
                  // Tensor::dot path to isolate whether Q8_0_Tensor::dot is
                  // corrupting the scratch for this layer.
                  std::cerr << "[CONV_FWD] " << context.getName()
                            << " Q8 input fallback dequant+dot" << std::endl;
                  const size_t in_nelem = (size_t)owoh * in_ch_i;
                  std::vector<_FP16> fp16_in(in_nelem);
                  const uint8_t *storage_in =
                    reinterpret_cast<const uint8_t *>(in_sub.getData()) -
                    sizeof(uint16_t);
                  uint16_t d_u16;
                  std::memcpy(&d_u16, storage_in, sizeof(uint16_t));
                  float in_scale = static_cast<float>(
                    *reinterpret_cast<const _FP16 *>(&d_u16));
                  const int8_t *qs = in_sub.getData<int8_t>();
                  for (size_t i = 0; i < in_nelem; ++i)
                    fp16_in[i] = static_cast<_FP16>(static_cast<float>(qs[i]) *
                                                    in_scale);
                  Tensor act(TensorDim(1, 1, owoh, in_dim.channel(),
                                       {ml::train::TensorDim::Format::NCHW,
                                        nntrainer::Tdatatype::FP16}),
                             fp16_in.data());
                  std::cerr << "[CONV_FWD] " << context.getName()
                            << " fallback act.getData()=" << act.getData()
                            << std::endl;
                  act.dot(filter_kernel, out_flat, false, false);
                  std::cerr << "[CONV_FWD] " << context.getName()
                            << " returned from fallback dot" << std::endl;
                } else {
                  std::cerr << "[CONV_FWD] " << context.getName()
                            << " about to dot in_sub fallback" << std::endl;
                  Tensor act = in_sub;
                  act.reshape(TensorDim(1, 1, owoh, in_dim.channel(),
                                        {ml::train::TensorDim::Format::NCHW,
                                         in_sub.getDataType()}));
                  std::cerr << "[CONV_FWD] " << context.getName()
                            << " in_sub.getData()=" << in_sub.getData()
                            << " dim=" << in_sub.batch() << "x"
                            << in_sub.channel() << "x" << in_sub.height()
                            << "x" << in_sub.width() << std::endl;
                  act.dot(filter_kernel, out_flat, false, false);
                  std::cerr << "[CONV_FWD] " << context.getName()
                            << " returned from dot fallback" << std::endl;
                }
              } else if (output_is_q8) {
                // Kernel K is not a multiple of 32: run FP16 matmul and quantize
                // the output. This only touches the output, not the input path.
                Tensor act = in_sub;
                act.reshape(TensorDim(1, 1, owoh, in_dim.channel(),
                                      {ml::train::TensorDim::Format::NCHW,
                                       in_sub.getDataType()}));
                act.dot(filter_kernel, out_flat, false, false);
              } else {
#endif
                Tensor act = in_sub;
                act.reshape(TensorDim(1, 1, owoh, in_dim.channel(),
                                      {ml::train::TensorDim::Format::NCHW,
                                       in_sub.getDataType()}));
                act.dot(filter_kernel, out_flat, false, false);
#ifdef ENABLE_FP16
              }
#endif
            } else if (NNTR_HAS_Q4_0_INDIRECT_CONV) {
              std::cerr << "[CONV_FWD] " << context.getName()
                        << " indirect conv path" << std::endl;
              ConvGatherParams geom;
              geom.in_ch = in_ch_i;
              geom.in_h = in_dim.height();
              geom.in_w = in_dim.width();
              geom.k_h = kernel_size[0].get();
              geom.k_w = kernel_size[1].get();
              geom.pad_t = padding[0];
              geom.pad_l = padding[2];
              geom.stride_h = stride[0].get();
              geom.stride_w = stride[1].get();
              geom.dil_h = dilation[0].get();
              geom.dil_w = dilation[1].get();
              geom.out_w = out_dim.width();

#ifdef ENABLE_FP16
              if (k_eligible) {
                const int n_sp = geom.in_h * geom.in_w;
                TensorDim in_q8_dim(
                  {1, 1, (unsigned)n_sp, (unsigned)in_ch_i},
                  {ml::train::TensorDim::Format::NCHW,
                   nntrainer::Tdatatype::Q8_0});
                Q8_0_Tensor *in_q8_ptr = nullptr;
                Q8_0_Tensor in_q8_view;
                if (input_is_q8) {
                  in_q8_view = Q8_0_Tensor(
                    in_q8_dim,
                    reinterpret_cast<uint8_t *>(in_sub.getData()) -
                      sizeof(uint16_t));
                  in_q8_ptr = &in_q8_view;
                } else if (q8_scratch_base != nullptr) {
                  quantize_nhwc_q8_0_tensor(in_sub.getData<_FP16>(), n_sp,
                                            in_ch_i, q8_scratch_base);
                  in_q8_view = Q8_0_Tensor(in_q8_dim, q8_scratch_base);
                  in_q8_ptr = &in_q8_view;
                }

                if (in_q8_ptr != nullptr) {
                  std::cerr << "[CONV_FWD] " << context.getName()
                            << " about to convQ4_0Indirect Q8 input" << std::endl;
                  geom.is_nhwc = true;
                  in_q8_ptr->convQ4_0Indirect(filter_kernel, out_flat, geom);
                  std::cerr << "[CONV_FWD] " << context.getName()
                            << " returned convQ4_0Indirect Q8 input" << std::endl;
                } else {
                  std::cerr << "[CONV_FWD] " << context.getName()
                            << " about to convQ4_0Indirect FP16 input" << std::endl;
                  geom.is_nhwc = true;
                  in_sub.convQ4_0Indirect(filter_kernel, out_flat, geom);
                  std::cerr << "[CONV_FWD] " << context.getName()
                            << " returned convQ4_0Indirect FP16 input" << std::endl;
                }
              } else {
                // Non-32-aligned K does not fit the block_q8_0x4 kernel. The
                // indirect GEMM kernels round K up to 32 and pad zeros, which is
                // valid for FP16/FP32 inputs (their per-block scale can absorb
                // the extra column). For tensor-wise Q8_0 we avoid padding the
                // input because the single tensor scale cannot represent the
                // mismatched K. Instead stage the NHWC FP16/Q8_0 input through
                // im2col to a contiguous [OH*OW, CRS] FP16 buffer, run the
                // existing FP16/Q4_0 matmul, then the Q8_0 epilogue handles bias
                // + SiLU and quantizes the output.
                std::cerr << "[CONV_FWD] " << context.getName()
                          << " non-32 fallback col_scratch="
                          << (col_scratch ? "ok" : "null") << std::endl;
                Tensor col = col_scratch->getBatchSlice(b, 1);
                TensorDim kdim(filter_size, in_dim.channel(),
                               kernel_size[0].get(), kernel_size[1].get(),
                               in_sub.getTensorType());
                std::cerr << "[CONV_FWD] " << context.getName()
                          << " kdim=" << (int)kdim.getFormat() << " "
                          << kdim.batch() << "x" << kdim.channel() << "x"
                          << kdim.height() << "x" << kdim.width()
                          << " dt=" << (int)kdim.getDataType() << std::endl;
                Tensor in_nchw = in_sub;
                if (in_sub.getFormat() ==
                    ml::train::TensorDim::Format::NHWC) {
                  // input is [H,W,C] but im2col below expects NCHW layout for
                  // the dims it uses; reshape only changes logical metadata.
                  in_nchw.reshape(
                    TensorDim(1, in_dim.channel(), in_dim.height(),
                              in_dim.width(), in_sub.getTensorType()));
                }
                std::cerr << "[CONV_FWD] " << context.getName()
                          << " pre-im2col" << std::endl;
                im2col(in_nchw, kdim, padding, stride, dilation, col);
                std::cerr << "[CONV_FWD] " << context.getName()
                          << " post-im2col" << std::endl;
                Tensor act = col;
                act.reshape(TensorDim(1, 1, owoh,
                                      (unsigned)in_ch_i * geom.k_h * geom.k_w,
                                      {ml::train::TensorDim::Format::NCHW,
                                       nntrainer::Tdatatype::FP16}));
                std::cerr << "[CONV_FWD] " << context.getName()
                          << " pre-dot act=" << (int)act.getFormat() << " "
                          << act.batch() << "x" << act.channel() << "x"
                          << act.height() << "x" << act.width()
                          << " out_flat=" << (int)out_flat.getFormat() << " "
                          << out_flat.batch() << "x" << out_flat.channel() << "x"
                          << out_flat.height() << "x" << out_flat.width()
                          << std::endl;
                act.dot(filter_kernel, out_flat, false, false);
                std::cerr << "[CONV_FWD] " << context.getName()
                          << " post-dot" << std::endl;
              }
#else
              geom.is_nhwc = true;
              in_sub.convQ4_0Indirect(filter_kernel, out_flat, geom);
#endif
            } else {
              throw std::runtime_error("Fallback quantized NHWC conv is not supported (requires indirect conv on ARM).");
            }

            // Tensor-wise Q8_0 output epilogue: bias + SiLU via LUT, then
            // quantize to a single scale for the whole output tensor.
            if (output_is_q8) {
              std::cerr << "[CONV_FWD] " << context.getName()
                        << " output_is_q8 epilogue" << std::endl;
              Tensor q8_out = out;
              q8_out.reshape(TensorDim(1, 1, owoh, filter_size,
                                       {ml::train::TensorDim::Format::NCHW,
                                        nntrainer::Tdatatype::Q8_0}));
              const _FP16 *bias = nullptr;
              if (auto &disable_bias =
                    std::get<props::DisableBias>(*layer_impl_props);
                  disable_bias.empty() || disable_bias.get() == false) {
                bias = context.getWeight(wt_idx[ConvParams::bias])
                         .getData<_FP16>();
              }
              _FP16 *scratch = tmp_fp16.getData<_FP16>();
              uint8_t *q8_storage =
                reinterpret_cast<uint8_t *>(q8_out.getData()) -
                sizeof(uint16_t);

              const bool fuse_silu = [&]() {
                const auto &act =
                  std::get<props::FusedActivation>(conv_props);
                return !act.empty() && act.get() == ActivationType::ACT_SWISH;
              }();

              if (fuse_silu) {
                std::cerr << "[CONV_FWD] " << context.getName()
                          << " about to fused_bias_silu_quantize" << std::endl;
                fused_bias_silu_quantize_tensor_wise_q8_0(
                  scratch, (int)owoh, (int)filter_size, bias, q8_storage,
                  context.getName().c_str());
                if (std::getenv("NNTR_Q8_PROBE")) {
                  uint16_t d_u16;
                  std::memcpy(&d_u16, q8_storage, 2);
                  float scale = static_cast<float>(
                    *reinterpret_cast<const _FP16 *>(&d_u16));
                  const int8_t *qs = reinterpret_cast<const int8_t *>(
                    q8_storage + sizeof(uint16_t));
                  int mnq = qs[0], mxq = qs[0];
                  size_t nz = 0, npos = 0, nneg = 0, n127 = 0;
                  const size_t nelem = (size_t)owoh * filter_size;
                  for (size_t i = 0; i < nelem; ++i) {
                    mnq = std::min(mnq, (int)qs[i]);
                    mxq = std::max(mxq, (int)qs[i]);
                    if (qs[i] == 0)
                      ++nz;
                    else if (qs[i] > 0)
                      ++npos;
                    else
                      ++nneg;
                    if (qs[i] == 127)
                      ++n127;
                  }
                  std::cerr << "[Q8_PROBE] " << context.getName()
                            << " fused scale=" << scale << " qmin=" << mnq
                            << " qmax=" << mxq << " zeros=" << nz
                            << " pos=" << npos << " neg=" << nneg
                            << " n127=" << n127 << std::endl;
                }
                std::cerr << "[CONV_FWD] " << context.getName()
                          << " returned fused_bias_silu_quantize" << std::endl;
              } else {
                if (bias != nullptr) {
                  const unsigned int C = filter_size;
                  const unsigned int HW = owoh;
                  auto &tm = ThreadManager::Global();
                  const unsigned int chunk = 1024;
                  const size_t loops = (HW + chunk - 1) / chunk;
                  tm.parallel_for(0, loops, [&](size_t idx) {
                    unsigned int r0 = idx * chunk;
                    unsigned int r1 = std::min(r0 + chunk, HW);
                    for (unsigned int r = r0; r < r1; ++r) {
                      _FP16 *row = scratch + (size_t)r * C;
                      for (unsigned int c = 0; c < C; ++c)
                        row[c] = static_cast<_FP16>(static_cast<float>(row[c]) +
                                                    static_cast<float>(bias[c]));
                    }
                  });
                }
                quantize_nhwc_q8_0_tensor(scratch, (int)owoh, (int)filter_size,
                                          q8_storage);
              }
            }
            std::cerr << "[CONV_FWD] " << context.getName()
                      << " end NHWC weight_is_quant batch body" << std::endl;
          } else {
            // Quantized conv as matmul: act [OH*OW, CRS] . weight [CRS, out_ch]
            // -> [OH*OW, out_ch] -> out [out_ch, OH*OW]. CRS = in_ch*kh*kw.
            // NOTE: col must outlive `act` (act aliases col's storage); here col
            // is a view into the context-owned scratch, so its storage outlives
            // the loop iteration regardless.
            //
            // For a Q8_0-activation graph the scratch stays FP16 so that bias +
            // SiLU can reuse the existing FP16 epilogue; we quantize to Q8_0
            // after the transpose.
            Tensor tmp = qgemm_scratch->getBatchSlice(b, 1);
            tmp.reshape(TensorDim(1, 1, owoh, filter_size,
                                  {ml::train::TensorDim::Format::NCHW,
                                   output_is_q8 ? nntrainer::Tdatatype::FP16
                                                : in_sub.getDataType()}));
            if (is_1x1_s1) {
              // 1x1 stride-1: im2col is an identity. The raw input is laid out as
              // [in_ch, OH*OW] (NCHW), so transpose to the act layout [OH*OW,
              // CRS] (CRS == in_ch here).
              in_sub.reshape({in_dim.channel(), owoh});
              Tensor act = in_sub.transpose("0:2:1");
              act.dot(filter_kernel, tmp, false, false);
            } else if (NNTR_HAS_Q4_0_INDIRECT_CONV) {
              // Quantized 3x3+ indirect: fold im2col gather into the q8_0
              // activation quantization so the activation matrix is never
              // materialized (the FP16 input is gathered on the fly and quantized
              // per tile inside the indirect GEMM). Output tmp is FP16
              // [OH*OW, out_ch].
              ConvGatherParams geom;
              geom.in_ch = in_dim.channel();
              geom.in_h = in_dim.height();
              geom.in_w = in_dim.width();
              geom.k_h = kernel_size[0].get();
              geom.k_w = kernel_size[1].get();
              geom.pad_t = padding[0];
              geom.pad_l = padding[2];
              geom.stride_h = stride[0].get();
              geom.stride_w = stride[1].get();
              geom.dil_h = dilation[0].get();
              geom.dil_w = dilation[1].get();
              geom.out_w = out_dim.width();
              geom.is_nhwc = false;
              in_sub.convQ4_0Indirect(filter_kernel, tmp, geom);
            } else {
              // Fallback (no fused backend op): materialize im2col into the col
              // scratch, then the standard quant GEMM.
              // build the real kernel geometry (filter is stored as [CRS,out_ch])
              TensorDim kdim(filter_size, in_dim.channel(), kernel_size[0].get(),
                             kernel_size[1].get(), in_sub.getTensorType());
              Tensor col = col_scratch->getBatchSlice(b, 1);
              // im2col reshapes col in place to [OH*OW, CRS] (spatial-major),
              // which is ALREADY the act layout — no transpose (unlike the
              // raw-input 1x1 branch above). Transposing here gives [CRS, OH*OW]
              // and makes the GEMM emit CRS rows into the owoh-row `tmp`,
              // overflowing it whenever CRS > owoh (deep convs) -> heap
              // corruption.
              im2col(in_sub, kdim, padding, stride, dilation, col);
              col.dot(filter_kernel, tmp, false, false);
            }
            // [OH*OW, out_ch] -> [out_ch, OH*OW] written straight into the
            // (memory-planned) output. `tmp` is a separate scratch buffer and
            // `out` is a separate output view, so there is no aliasing.
            if (output_is_q8) {
              // Transpose into a temporary FP16 channel-major view of `out`,
              // then quantize that to Q8_0. We borrow qgemm_scratch storage
              // for the transpose by reshaping it; it is no longer needed.
              Tensor tmp_chw = tmp;
              tmp_chw.reshape({filter_size, owoh});
              Tensor chw_fp16(out.getDim(), true);
              tmp_chw.transpose("0:2:1", chw_fp16);
              Tensor q8_out = out;
              q8_out.reshape(TensorDim(owoh, filter_size,
                                       {ml::train::TensorDim::Format::NCHW,
                                        nntrainer::Tdatatype::Q8_0}));
              quantize_nhwc_q8_0_rows(chw_fp16.getData<_FP16>(), (int)owoh,
                                      (int)filter_size,
                                      reinterpret_cast<::nntrainer::block_q8_0 *>(
                                        q8_out.getData()));
            } else {
              out.reshape({filter_size, owoh});
              tmp.transpose("0:2:1", out);
            }
          }
        } else if (in_sub.getDataType() == nntrainer::Tdatatype::Q8_0 ||
                   out.getDataType() == nntrainer::Tdatatype::Q8_0) {
          std::cerr << "[CONV_FWD] " << context.getName()
                    << " enter non-quant-weight Q8_0 branch" << std::endl;
#ifdef ENABLE_FP16
          // Non-quantized weights (e.g., a 1x1 head conv with out_ch=1) in a
          // Q8_0-activation graph. Stage through FP16 compute, then quantize
          // the output back to tensor-wise Q8_0.
          // Stage through contiguous FP32 compute: Q8_0 input -> FP32, FP32
          // im2col + FP32 filter dot -> FP32 output, then bias + SiLU and
          // quantize back to tensor-wise Q8_0.
          const bool input_is_q8 =
            in_sub.getDataType() == nntrainer::Tdatatype::Q8_0;
          std::cerr << "[CONV_FWD] " << context.getName()
                    << " input_is_q8=" << input_is_q8 << std::endl;
          Tensor in_work = in_sub;
          if (in_sub.getDataType() != nntrainer::Tdatatype::FP32) {
            TensorDim din32 = in_sub.getDim();
            din32.setDataType(nntrainer::Tdatatype::FP32);
            Tensor fp32_in(din32, true);
            const size_t nelem = in_sub.getDim().getDataLen();
            if (in_sub.getDataType() == nntrainer::Tdatatype::Q8_0) {
              dequantize_tensor_wise_q8_0_to_fp32(
                in_sub.getData<int8_t>(), nelem, fp32_in.getData<float>());
#ifdef ENABLE_FP16
            } else if (in_sub.getDataType() == nntrainer::Tdatatype::FP16) {
              const _FP16 *src = in_sub.getData<_FP16>();
              float *dst = fp32_in.getData<float>();
              for (size_t i = 0; i < nelem; ++i)
                dst[i] = static_cast<float>(src[i]);
#endif
            } else {
              throw std::runtime_error(
                "Conv2D Q8_0 non-quant fallback: unsupported input dtype " +
                std::to_string(static_cast<int>(in_sub.getDataType())));
            }
            in_work = fp32_in;
          }
          std::cerr << "[CONV_FWD] " << context.getName()
                    << " pre-col in_work_dt="
                    << (int)in_work.getDataType() << std::endl;

          TensorDim dout32 = out.getDim();
          dout32.setDataType(nntrainer::Tdatatype::FP32);
          Tensor fp32_out(dout32, true);

          TensorDim col_dim = calcCol2ImOutputDim(out_dim, real_filter_dim);
          col_dim.setDataType(nntrainer::Tdatatype::FP32);
          col_dim.setFormat(in_work.getFormat());
          Tensor result(col_dim, true);
          result.setZero();
          std::cerr << "[CONV_FWD] " << context.getName()
                    << " pre-im2col" << std::endl;
          std::cerr << "[CONV_FWD] " << context.getName()
                    << " in_work dim=" << in_work.batch() << "x"
                    << in_work.channel() << "x" << in_work.height() << "x"
                    << in_work.width() << " fmt=" << (int)in_work.getFormat()
                    << " dt=" << (int)in_work.getDataType()
                    << " bytes=" << in_work.getMemoryBytes() << std::endl;
          std::cerr << "[CONV_FWD] " << context.getName()
                    << " real_filter_dim=" << real_filter_dim.batch() << "x"
                    << real_filter_dim.channel() << "x"
                    << real_filter_dim.height() << "x"
                    << real_filter_dim.width() << " fmt="
                    << (int)real_filter_dim.getFormat() << std::endl;
          std::cerr << "[CONV_FWD] " << context.getName()
                    << " col_dim=" << col_dim.batch() << "x"
                    << col_dim.channel() << "x" << col_dim.height() << "x"
                    << col_dim.width() << " fmt=" << (int)col_dim.getFormat()
                    << " dt=" << (int)col_dim.getDataType()
                    << " bytes=" << col_dim.getDataLen() * sizeof(float)
                    << std::endl;
          std::cerr << "[CONV_FWD] " << context.getName()
                    << " result bytes=" << result.getMemoryBytes() << std::endl;
          std::cerr << "[CONV_FWD] " << context.getName()
                    << " out_dim=" << out_dim.batch() << "x" << out_dim.channel()
                    << "x" << out_dim.height() << "x" << out_dim.width()
                    << " owoh=" << owoh << std::endl;

          im2col(in_work, real_filter_dim, padding, stride, dilation, result);
          // restore filter_kernel to real geometry before any later code reads it
          // (e.g. the final reshape/filter_dim validation outside the batch loop).
          if (!weight_is_quant)
            filter_kernel.reshape(real_filter_dim);
          std::cerr << "[CONV_FWD] " << context.getName()
                    << " post-im2col" << std::endl;

          // Compute filter[filter_size, CRS] * im2col^T[CRS, owoh] directly with
          // explicit row-major sgemm, bypassing Tensor::dot which couples to the
          // NHWC TensorDim metadata in this fallback path.
          const unsigned int crs = real_filter_dim.getFeatureLen();
          float *im2col_data = result.getData<float>();
          float *weight_data = filter_kernel.getData<float>();
          float *out_data = fp32_out.getData<float>();
          nntrainer::getComputeOps()->sgemm_fp32(
            static_cast<unsigned int>(TensorDim::StorageOrder::ROW_MAJOR),
            false, true, owoh, filter_size, crs, 1.0f, im2col_data, crs,
            weight_data, crs, 0.0f, out_data, filter_size);
          std::cerr << "[CONV_FWD] " << context.getName()
                    << " post-dot" << std::endl;
          if (std::getenv("NNTR_Q8_PROBE")) {
            float mn = out_data[0], mx = out_data[0];
            for (size_t i = 1; i < (size_t)owoh * filter_size; ++i) {
              mn = std::min(mn, out_data[i]);
              mx = std::max(mx, out_data[i]);
            }
            std::cerr << "[Q8_PROBE] NONQUANT_POST_DOT " << context.getName()
                      << " out_min=" << mn << " out_max=" << mx << std::endl;
          }

          const _FP16 *bias_h = nullptr;
          if (auto &disable_bias =
                std::get<props::DisableBias>(*layer_impl_props);
              disable_bias.empty() || disable_bias.get() == false) {
            bias_h = context.getWeight(wt_idx[ConvParams::bias])
                      .getData<_FP16>();
          }
          const bool fuse_silu = [&]() {
            const auto &act =
              std::get<props::FusedActivation>(conv_props);
            return !act.empty() && act.get() == ActivationType::ACT_SWISH;
          }();

          // fp32_out is in NHWC/a contiguous flat layout after the per-format
          // write above; collapse it to [owoh, filter_size] for post-processing.
          Tensor scratch_view = fp32_out;
          scratch_view.reshape(
            TensorDim(1, 1, owoh, filter_size,
                      {ml::train::TensorDim::Format::NCHW,
                       nntrainer::Tdatatype::FP32}));
          float *scratch = scratch_view.getData<float>();
          const size_t nelem = (size_t)owoh * filter_size;
          if (bias_h != nullptr) {
            for (size_t r = 0; r < owoh; ++r) {
              float *row = scratch + r * filter_size;
              for (unsigned int c = 0; c < filter_size; ++c)
                row[c] += static_cast<float>(bias_h[c]);
            }
          }
          if (fuse_silu) {
            convApplySwishInplace(scratch, nelem);
          }
          if (std::getenv("NNTR_Q8_PROBE")) {
            float mn = scratch[0], mx = scratch[0];
            for (size_t i = 1; i < nelem; ++i) {
              mn = std::min(mn, scratch[i]);
              mx = std::max(mx, scratch[i]);
            }
            std::cerr << "[Q8_PROBE] NONQUANT_POST_BIAS_ACT " << context.getName()
                      << " scratch_min=" << mn << " scratch_max=" << mx
                      << std::endl;
          }

          if (out.getDataType() == nntrainer::Tdatatype::Q8_0) {
            uint8_t *qbase = reinterpret_cast<uint8_t *>(out.getData()) -
                             sizeof(uint16_t);
            if (context.getName() == "conv0/conv" ||
                std::getenv("NNTR_Q8_PROBE")) {
              float mn = scratch[0], mx = scratch[0];
              size_t n127 = 0, nn127 = 0;
              for (size_t i = 0; i < nelem; ++i) {
                mn = std::min(mn, scratch[i]);
                mx = std::max(mx, scratch[i]);
                if (scratch[i] >= 127.0f)
                  ++n127;
                if (scratch[i] <= -127.0f)
                  ++nn127;
              }
              std::cerr << "[Q8_PROBE] " << context.getName()
                        << " pre_quant min=" << mn << " max=" << mx
                        << " n>=127=" << n127 << " n<=-127=" << nn127
                        << std::endl;
            }
            quantize_nhwc_q8_0_tensor_fp32(scratch, (int)owoh, (int)filter_size,
                                           qbase);
            if (context.getName() == "conv0/conv" ||
                std::getenv("NNTR_Q8_PROBE")) {
              uint16_t d_u16;
              std::memcpy(&d_u16, qbase, 2);
              float scale = static_cast<float>(
                *reinterpret_cast<const _FP16 *>(&d_u16));
              const int8_t *qs = reinterpret_cast<const int8_t *>(qbase + 2);
              int mnq = qs[0], mxq = qs[0];
              size_t nz = 0, npos = 0, nneg = 0;
              for (size_t i = 0; i < nelem; ++i) {
                mnq = std::min(mnq, (int)qs[i]);
                mxq = std::max(mxq, (int)qs[i]);
                if (qs[i] == 0)
                  ++nz;
                else if (qs[i] > 0)
                  ++npos;
                else
                  ++nneg;
              }
              std::cerr << "[Q8_PROBE] " << context.getName()
                        << " scale=" << scale << " qmin=" << mnq
                        << " qmax=" << mxq << " zeros=" << nz
                        << " pos=" << npos << " neg=" << nneg << std::endl;
            }
          } else if (out.getDataType() == nntrainer::Tdatatype::FP32) {
            // Non-quantized head conv forced to full precision: copy the FP32
            // result into the planned output tensor.
            float *dst = out.getData<float>();
            std::memcpy(dst, scratch, nelem * sizeof(float));
          }
#else
          throw std::runtime_error(
            "Q8_0 conv with non-quantized weights requires FP16 support.");
#endif
        } else {
          Tensor result = col_scratch->getBatchSlice(b, 1);
          out.reshape({filter_size, owoh});
          im2col(in_sub, real_filter_dim, padding, stride, dilation, result);
          // filter kernel is (K, CRS), result is (CRS, OH*OW)
          if (out.getFormat() == ml::train::TensorDim::Format::NCHW) {
            filter_kernel.dot(result, out, false, true);
          } else {
            // NHWC: out's physical layout is [OH,OW,C] (channel innermost), so
            // a dot writing [out_ch, OH*OW] NCHW-order would land in the wrong
            // physical cells. Compute into a plain NCHW-order buffer (format
            // NCHW so dot writes channel-major), then scatter channel c of
            // spatial r to out[r*out_ch + c] (NHWC physical).
            // Compute into a plain NCHW-order buffer (same result feeding as the
            // NCHW path above — do NOT reshape result, im2col already laid it
            // out as the dot expects). nchw_out is format NCHW so dot writes
            // channel-major [out_ch, OH*OW]; then scatter channel c of spatial r
            // to out[r*out_ch + c] (NHWC physical).
            auto nchw_type = out.getTensorType();
            nchw_type.format = ml::train::TensorDim::Format::NCHW;
            TensorDim nchw_dim({1, filter_size, owoh, 1}, nchw_type);
            Tensor nchw_out(nchw_dim, true);
            filter_kernel.dot(result, nchw_out, false, true);
            if (out.getDataType() == nntrainer::Tdatatype::FP32) {
              const float *s = nchw_out.getData<float>();
              float *d = out.getData<float>();
              for (unsigned int oc = 0; oc < filter_size; ++oc)
                for (unsigned int r = 0; r < owoh; ++r)
                  d[r * filter_size + oc] = s[oc * owoh + r];
            }
#ifdef ENABLE_FP16
            else if (out.getDataType() == nntrainer::Tdatatype::FP16) {
              const _FP16 *s = nchw_out.getData<_FP16>();
              _FP16 *d = out.getData<_FP16>();
              for (unsigned int oc = 0; oc < filter_size; ++oc)
                for (unsigned int r = 0; r < owoh; ++r)
                  d[r * filter_size + oc] = s[oc * owoh + r];
            }
#endif
          }
        }
        if (out.getDim() != out_slice_orig) {
          out.reshape(out_slice_orig);
        }
      }
    };

    auto workers = ParallelBatch(forwarding_job, in_dim.batch(), nullptr);

    if (workers.getNumWorkers() > 1) {
      workers.run();
    } else {
      forwarding_job(0, in_dim.batch(), 0, nullptr);
    }

    // Final validation / restore: for the FP path the filter was reshaped to
    // [out_ch, CRS] inside each batch iteration, and already restored to the
    // original geometry. Nothing further to do.
    if (!weight_is_quant && filter_kernel.getDim() != filter_dim) {
      filter_kernel.reshape(filter_dim);
    }
  } else {
    // Grouped convolution: split channels into `groups` independent groups.
    const unsigned int ocg = filter_size / groups;      // out ch per group
    const unsigned int icg = in_dim.channel() / groups; // in ch per group
    const unsigned int fh = filter_dim.height(), fw = filter_dim.width();
    const unsigned int owoh = out_dim.width() * out_dim.height();
    const unsigned int ihw = in_dim.height() * in_dim.width();
    TensorDim fdim_g(ocg, icg, fh, fw, filter_dim.getTensorType());

    const bool is_true_depthwise = ocg == 1 && icg == 1;

    if (is_true_depthwise &&
        in_dim.getDataType() == nntrainer::Tdatatype::FP32) {
      // True depthwise (groups == channels): delegate to the CPU backend op so
      // the optimised kernel lives in the backend, not in the layer.
      nntrainer::getComputeOps()->depthwise_conv2d_fp32(
        input_.getData<float>(), filter_kernel.getData<float>(),
        hidden_.getData<float>(), in_dim.batch(), filter_size, in_dim.height(),
        in_dim.width(), out_dim.height(), out_dim.width(), fh, fw,
        stride[0].get(), stride[1].get(), padding[0], padding[2],
        dilation[0].get(), dilation[1].get());
#ifdef ENABLE_FP16
    } else if (is_true_depthwise &&
               in_dim.getDataType() == nntrainer::Tdatatype::FP16 &&
               hidden_.getDataType() == nntrainer::Tdatatype::FP16 &&
               filter_kernel.getDataType() == nntrainer::Tdatatype::FP32) {
      // FP16-activation depthwise: weights are never Q4_0 for groups>1 and stay
      // FP32 (BN-folded), so this is FP16 input/output x FP32 kernel. Keep it
      // on the tight channel-parallel direct-loop kernel instead of falling
      // into the generic grouped else-branch (per-channel im2col + FP16 GEMV),
      // which was ~2x slower for YOLOv11's detect-head depthwise convs.
      nntrainer::getComputeOps()->depthwise_conv2d_fp16(
        input_.getData<_FP16>(), filter_kernel.getData<float>(),
        hidden_.getData<_FP16>(), in_dim.batch(), filter_size, in_dim.height(),
        in_dim.width(), out_dim.height(), out_dim.width(), fh, fw,
        stride[0].get(), stride[1].get(), padding[0], padding[2],
        dilation[0].get(), dilation[1].get());
    } else if (is_true_depthwise &&
               in_dim.getDataType() == nntrainer::Tdatatype::Q8_0 &&
               hidden_.getDataType() == nntrainer::Tdatatype::Q8_0 &&
               filter_kernel.getDataType() == nntrainer::Tdatatype::FP32 &&
               in_dim.getFormat() ==
                 ml::train::TensorDim::Format::NHWC) {
      // Q8_0-activation depthwise bridge: dequant each batch to FP16 scratch,
      // run the optimised FP16 depthwise kernel, add bias, apply fused SiLU if
      // configured, then quantize back to tensor-wise Q8_0 output. This keeps
      // the graph end-to-end Q8_0 while reusing the existing FP16 depthwise
      // implementation; weights for depthwise convs are never Q4_0.
      Tensor &dw_in = context.getTensor(wt_idx[ConvParams::dw_in_scratch]);
      Tensor &dw_out = context.getTensor(wt_idx[ConvParams::dw_out_scratch]);
      const _FP16 *bias = nullptr;
      if (auto &disable_bias =
            std::get<props::DisableBias>(*layer_impl_props);
          disable_bias.empty() || disable_bias.get() == false) {
        bias = context.getWeight(wt_idx[ConvParams::bias]).getData<_FP16>();
      }
      const bool fuse_silu = [&]() {
        const auto &act = std::get<props::FusedActivation>(conv_props);
        return !act.empty() && act.get() == ActivationType::ACT_SWISH;
      }();

      for (unsigned int b = 0; b < in_dim.batch(); ++b) {
        Tensor in_b = input_.getBatchSlice(b, 1);
        Tensor out_b = hidden_.getBatchSlice(b, 1);
        Tensor dw_in_b = dw_in.getBatchSlice(b, 1);
        Tensor dw_out_b = dw_out.getBatchSlice(b, 1);

        const uint8_t *storage_in = reinterpret_cast<const uint8_t *>(
          in_b.getData()) - sizeof(uint16_t);
        uint16_t d_u16;
        std::memcpy(&d_u16, storage_in, sizeof(uint16_t));
        float in_scale = static_cast<float>(*reinterpret_cast<const _FP16 *>(&d_u16));
        const int8_t *qs = reinterpret_cast<const int8_t *>(in_b.getData());
        _FP16 *fp16_in = dw_in_b.getData<_FP16>();
        const size_t in_nelem = in_b.getDim().getDataLen();
        for (size_t i = 0; i < in_nelem; ++i)
          fp16_in[i] = static_cast<_FP16>(static_cast<float>(qs[i]) * in_scale);

        nntrainer::getComputeOps()->depthwise_conv2d_fp16(
          dw_in_b.getData<_FP16>(), filter_kernel.getData<float>(),
          dw_out_b.getData<_FP16>(), 1, filter_size, in_dim.height(),
          in_dim.width(), out_dim.height(), out_dim.width(), fh, fw,
          stride[0].get(), stride[1].get(), padding[0], padding[2],
          dilation[0].get(), dilation[1].get());

        _FP16 *scratch = dw_out_b.getData<_FP16>();
        uint8_t *q8_storage = reinterpret_cast<uint8_t *>(out_b.getData()) -
                              sizeof(uint16_t);
        if (fuse_silu) {
          fused_bias_silu_quantize_tensor_wise_q8_0(
            scratch, (int)(out_dim.height() * out_dim.width()),
            (int)filter_size, bias, q8_storage,
            context.getName().c_str());
        } else {
          if (bias != nullptr) {
            const unsigned int C = filter_size;
            const unsigned int HW = out_dim.height() * out_dim.width();
            auto &tm = ThreadManager::Global();
            const unsigned int chunk = 1024;
            const size_t loops = (HW + chunk - 1) / chunk;
            tm.parallel_for(0, loops, [&](size_t idx) {
              unsigned int r0 = idx * chunk;
              unsigned int r1 = std::min(r0 + chunk, HW);
              for (unsigned int r = r0; r < r1; ++r) {
                _FP16 *row = scratch + (size_t)r * C;
                for (unsigned int c = 0; c < C; ++c)
                  row[c] = static_cast<_FP16>(static_cast<float>(row[c]) +
                                              static_cast<float>(bias[c]));
              }
            });
          }
          quantize_nhwc_q8_0_tensor(scratch, (int)(out_dim.height() * out_dim.width()),
                                    (int)filter_size, q8_storage);
        }
      }
#endif
    } else {
      // getSharedDataTensor()/reshape() adopt the *passed* TensorDim's dtype
      // (TensorBase::getSharedDataTensor: ret->dim = dim_). A bare {..} dim
      // literal defaults to FP32, so on an FP16 activation graph every sub-view
      // below would silently relabel FP16-backed storage as FP32 -- im2col
      // would gather at the wrong precision and the dot would write 4-byte
      // floats into 2-byte half storage (overflow / garbage). Carry each
      // parent's real dtype onto its view. (For an all-FP32 graph these are
      // no-ops, so the historical path is unchanged.)
      // getSharedDataTensor()/reshape() adopt the *passed* TensorDim's dtype
      // and format (TensorBase::getSharedDataTensor: ret->dim = dim_). A bare
      // {..} dim literal defaults to NCHW/FP32, so on an NHWC FP16/Q8_0 graph
      // every sub-view below would silently relabel the storage and trigger a
      // format-mismatch assertion. Carry each parent's real dtype and format.
      const auto in_dt = input_.getDataType();
      const auto filt_dt = filter_kernel.getDataType();
      const auto out_dt = hidden_.getDataType();
      const auto in_fmt = in_dim.getFormat();
      const auto out_fmt = out_dim.getFormat();
      for (unsigned int b = 0; b < in_dim.batch(); ++b) {
        Tensor out = hidden_.getBatchSlice(b, 1);
        TensorDim out_rdim({filter_size, owoh});
        out_rdim.setDataType(out_dt);
        out_rdim.setFormat(out_fmt);
        out.reshape(out_rdim);
        Tensor in_sub = input_.getBatchSlice(b, 1);
        TensorDim col_dim = calcCol2ImOutputDim(out_dim, fdim_g);
        col_dim.setDataType(in_dt);
        col_dim.setFormat(in_fmt);
        Tensor result = Tensor(col_dim);
        for (unsigned int g = 0; g < groups; ++g) {
          TensorDim ing_dim({1, icg, in_dim.height(), in_dim.width()});
          ing_dim.setDataType(in_dt);
          ing_dim.setFormat(in_fmt);
          Tensor in_g =
            in_sub.getSharedDataTensor(ing_dim, (size_t)g * icg * ihw);
          TensorDim filtg_dim({ocg, (size_t)icg * fh * fw});
          filtg_dim.setDataType(filt_dt);
          filtg_dim.setFormat(in_fmt);
          Tensor filt_g = filter_kernel.getSharedDataTensor(
            filtg_dim, (size_t)g * ocg * icg * fh * fw);
          TensorDim outg_dim({ocg, owoh});
          outg_dim.setDataType(out_dt);
          outg_dim.setFormat(out_fmt);
          Tensor out_g =
            out.getSharedDataTensor(outg_dim, (size_t)g * ocg * owoh);
          result.setZero();
          im2col(in_g, fdim_g, padding, stride, dilation, result);
          filt_g.dot(result, out_g, false, true);
        }
        result.deallocate();
      }
    }
  }

  // Bias and fused SiLU are applied on the FP16 qgemm_scratch when the output
  // activation dtype is Q8_0, then quantized back to Q8_0 inside the loop.
  if (hidden_.getDataType() != nntrainer::Tdatatype::Q8_0) {
    if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
        disable_bias.empty() || disable_bias.get() == false) {
      Tensor &bias_kernel = context.getWeight(wt_idx[ConvParams::bias]);
      if (hidden_.getFormat() == ml::train::TensorDim::Format::NCHW) {
        status = hidden_.add_i(bias_kernel);
        if (status != ML_ERROR_NONE) {
          throw std::invalid_argument("[Conv2D] adding bias failed");
        }
      } else {
        // NHWC: channel is innermost. bias [out_ch] must be added per (n,h,w,c).
        // add_i assumes NCHW channel-major broadcast, so do it inline.
        const unsigned int C = out_dim.channel();
        const unsigned int HW = out_dim.height() * out_dim.width();
        const unsigned int B = out_dim.batch();
        if (hidden_.getDataType() == nntrainer::Tdatatype::FP32) {
          float *d = hidden_.getData<float>();
          const float *bias = bias_kernel.getData<float>();
          for (unsigned int b = 0; b < B; ++b)
            for (unsigned int p = 0; p < HW; ++p)
              for (unsigned int c = 0; c < C; ++c)
                d[((size_t)b * HW + p) * C + c] += bias[c];
        }
#ifdef ENABLE_FP16
        else if (hidden_.getDataType() == nntrainer::Tdatatype::FP16) {
          _FP16 *d = hidden_.getData<_FP16>();
          const _FP16 *bias = bias_kernel.getData<_FP16>();
          for (unsigned int b = 0; b < B; ++b)
            for (unsigned int p = 0; p < HW; ++p)
              for (unsigned int c = 0; c < C; ++c)
                d[((size_t)b * HW + p) * C + c] += bias[c];
        }
#endif
      }
    }

    // Fused activation epilogue. When the graph sets activation=swish on the
    // conv, apply SiLU in-place on the freshly written output instead of
    // materializing a separate Activation layer (which would read the conv
    // output back from memory and write a second full tensor). Only SiLU is
    // fused here (YOLOv11's conv activation); any other activation type is left
    // to a dedicated Activation layer in the graph.
    if (auto &act = std::get<props::FusedActivation>(conv_props);
        !act.empty() && act.get() == ActivationType::ACT_SWISH) {
      const size_t n = hidden_.size();
#ifdef ENABLE_FP16
      if (hidden_.getDataType() == nntrainer::Tdatatype::FP16) {
        convApplySwishInplace(hidden_.getData<_FP16>(), n);
      } else
#endif
      {
        convApplySwishInplace(hidden_.getData<float>(), n);
      }
    }
  }

  std::cerr << "[CONV_FWD] " << context.getName() << " returning hidden dt="
            << (int)hidden_.getDataType() << " fmt=" << (int)hidden_.getFormat()
            << " dim=" << hidden_.batch() << "x" << hidden_.channel() << "x"
            << hidden_.height() << "x" << hidden_.width();
  if (hidden_.getDataType() == nntrainer::Tdatatype::Q8_0) {
    const uint8_t *storage = reinterpret_cast<const uint8_t *>(hidden_.getData()) -
                             sizeof(uint16_t);
    uint16_t du;
    std::memcpy(&du, storage, sizeof(uint16_t));
    float sc = static_cast<float>(*reinterpret_cast<const _FP16 *>(&du));
    std::cerr << " q8_data=" << hidden_.getData() << " q8_scale=" << sc;
  }
  std::cerr << std::endl;

  if (std::getenv("NNTR_CONV_PROBE") &&
      context.getName() == "conv0/conv") {
    std::cerr << "[CONV_PROBE] " << context.getName() << " format="
              << (int)hidden_.getFormat() << " C=" << out_dim.channel()
              << " H=" << out_dim.height() << " W=" << out_dim.width()
              << std::endl;
    const float *p = hidden_.getData<float>();
    for (int i = 0; i < 8; ++i)
      std::cerr << "  [" << i << "]=" << p[i] << std::endl;
  }
}

void Conv2DLayer::calcDerivative(RunLayerContext &context) {
  NNTR_THROW_IF(!std::get<props::ConvGroups>(conv_props).empty() &&
                  std::get<props::ConvGroups>(conv_props).get() != 1,
                std::invalid_argument)
    << "[Conv2D] backward for grouped convolution (groups>1) is not yet "
       "implemented; only forward/inference is supported.";
  unsigned int filter_size = std::get<props::FilterSize>(conv_props);
  auto &stride = std::get<std::array<props::Stride, CONV2D_DIM>>(conv_props);
  auto &dilation =
    std::get<std::array<props::Dilation, CONV2D_DIM>>(conv_props);

  const Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  Tensor &filter_kernel = context.getWeight(wt_idx[ConvParams::weight]);

  TensorDim filter_dim = filter_kernel.getDim();
  TensorDim filter_dim_squeezed{filter_kernel.batch(),
                                filter_kernel.getDim().getFeatureLen()};

  filter_kernel.reshape(filter_dim_squeezed);

  /// for each batch
  /// filter_kernel^T X derivaitive  -> column matrix
  /// col2im(column matrix) to reconstruct the original image

  auto compute_derivative = [&](unsigned int s, unsigned int e,
                                unsigned int pid, void *user_data) {
    Tensor result =
      Tensor(calcCol2ImOutputDim(derivative.getDim(), filter_dim));

    for (unsigned int b = s; b < e; ++b) {
      Tensor deriv_sub = derivative.getBatchSlice(b, 1);
      Tensor in_deriv_sub = input_derivative.getBatchSlice(b, 1);
      deriv_sub.reshape(
        {filter_size, derivative.width() * derivative.height()});
      // filter_kernel is (K, CRS), deriv_sub is (K, OH*OW), result is (CRS,
      // OH*OW)
      filter_kernel.dot(deriv_sub, result, true, false);
      col2im(result, filter_dim, padding, stride, dilation, in_deriv_sub);
      // in_derv_sub is (C,H,W)
    }
    result.deallocate();
  };

  auto workers = ParallelBatch(compute_derivative, derivative.batch(), nullptr);

  if (workers.getNumWorkers() > 1) {
    workers.run();
  } else {
    compute_derivative(0, derivative.batch(), 0, nullptr);
  }

  filter_kernel.reshape(filter_dim);
}

void Conv2DLayer::calcGradient(RunLayerContext &context) {
  NNTR_THROW_IF(!std::get<props::ConvGroups>(conv_props).empty() &&
                  std::get<props::ConvGroups>(conv_props).get() != 1,
                std::invalid_argument)
    << "[Conv2D] backward for grouped convolution (groups>1) is not yet "
       "implemented; only forward/inference is supported.";
  unsigned int filter_size = std::get<props::FilterSize>(conv_props);
  auto &stride = std::get<std::array<props::Stride, CONV2D_DIM>>(conv_props);
  auto &dilation =
    std::get<std::array<props::Dilation, CONV2D_DIM>>(conv_props);

  const Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  Tensor &delK = context.getWeightGrad(wt_idx[ConvParams::weight]);
  delK.setZero();

  TensorDim filter_dim = delK.getDim();
  TensorDim filter_dim_squeezed{filter_dim.batch(), filter_dim.getFeatureLen()};

  delK.reshape(filter_dim_squeezed);

  /**
   * no need to set zero for im2col_result, as its lifespan is ITERATION,
   * so its zero padded values will still be zero
   */

  TensorDim out_dim_squeezed{filter_size,
                             derivative.width() * derivative.height()};
  auto workers = ParallelBatch(input_.batch());
  /// input -(im2col)-> column_matrix -> filter x (column_matrix) = output
  /// so delK = dy x column_matrix ^ T;
  if (workers.getNumWorkers() > 1) {

    TensorDim delK_ext = filter_dim_squeezed;
    delK_ext.batch(input_.batch());

    Tensor delK_par = Tensor(delK_ext);
    delK_par.setZero();

    auto calc_grad_job = [&](unsigned int s, unsigned int e, unsigned int pid,
                             void *user_data) {
      Tensor result =
        Tensor(calcCol2ImOutputDim(derivative.getDim(), filter_dim));
      result.setZero();
      for (unsigned int b = s; b < e; ++b) {
        Tensor deriv_sub = derivative.getBatchSlice(b, 1);
        Tensor delK_sub = delK_par.getBatchSlice(b, 1);
        deriv_sub.reshape(out_dim_squeezed);

        Tensor in_sub = input_.getBatchSlice(b, 1);

        /**
         * @todo this result can be cached from the forward iteration at the
         * expense of memory. In this case, memory of im2col_result must be
         * saved for the whole batch. try this while benchmarking.
         */
        // deriv_sub is (K, OH*OW) and result is (CRS, OH*OW)
        im2col(in_sub, filter_dim, padding, stride, dilation, result);
        deriv_sub.dot(result, delK_sub, false, false);
      }
      result.deallocate();
    };

    workers.setCallback(calc_grad_job, nullptr);

    workers.run();

    for (unsigned int b = 0; b < input_.batch(); ++b) {
      Tensor delK_sub = delK_par.getBatchSlice(b, 1);
      delK.add_i(delK_sub);
    }

  } else {
    Tensor result =
      Tensor(calcCol2ImOutputDim(derivative.getDim(), filter_dim));
    result.setZero();

    for (unsigned int b = 0; b < input_.batch(); ++b) {
      Tensor deriv_sub = derivative.getBatchSlice(b, 1);
      deriv_sub.reshape(out_dim_squeezed);

      Tensor in_sub = input_.getBatchSlice(b, 1);

      /**
       * @todo this result can be cached from the forward iteration at the
       * expense of memory. In this case, memory of im2col_result must be saved
       * for the whole batch. try this while benchmarking.
       */
      im2col(in_sub, filter_dim, padding, stride, dilation, result);
      deriv_sub.dot(result, delK, false, false, b == 0 ? 0.0f : 1.0f);
    }
    result.deallocate();
  }
  delK.reshape(filter_dim);
  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &delBias = context.getWeightGrad(wt_idx[ConvParams::bias]);
    delBias.setZero();
    derivative.sum({0, 2, 3}, delBias);
  }
}

void Conv2DLayer::setBatch(RunLayerContext &context, unsigned int batch) {
  // The forward scratch buffers (im2col column buffer and quantized-GEMM
  // output) are requested in finalize() sized to the batch present at init.
  // When the runtime batch changes the framework resizes inputs/outputs but
  // not these layer-private scratch tensors, so rebatch them here — otherwise
  // forwarding()'s getBatchSlice(b, 1) for b >= the init batch reads past the
  // planned storage and aborts ("shared tensor bigger than tensor memory").
  if (wt_idx[ConvParams::im2col_scratch] !=
      std::numeric_limits<unsigned int>::max())
    context.updateTensor(wt_idx[ConvParams::im2col_scratch], batch);
  if (wt_idx[ConvParams::qgemm_scratch] !=
      std::numeric_limits<unsigned int>::max())
    context.updateTensor(wt_idx[ConvParams::qgemm_scratch], batch);
}

void Conv2DLayer::exportTo(Exporter &exporter,
                           const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(conv_props, method, this);
}

void Conv2DLayer::save(std::ofstream &file, RunLayerContext &run_context,
                       bool opt_var, ml::train::ExecutionMode mode,
                       bool trainable, ml::train::TensorDim::DataType dtype,
                       ml::train::ISA target_isa) const {
  // Optimizer-variable save (training only) has no conv-specific layout, so
  // defer to the base implementation.
  if (opt_var) {
    Layer::save(file, run_context, opt_var, mode, trainable, dtype, target_isa);
    return;
  }

  for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
    if (!run_context.isGradientFirstAccess(i))
      continue;

    auto &weight = run_context.getWeight(i);

    // No conversion requested, or already the target dtype: save as-is.
    if (dtype == TensorDim::DataType::NONE || weight.getDataType() == dtype) {
      weight.save(file);
      continue;
    }

    NNTR_THROW_IF(dtype != TensorDim::DataType::Q4_0, std::runtime_error)
      << "[Conv2D] save: unsupported quantization dtype";
    NNTR_THROW_IF(weight.getDataType() != TensorDim::DataType::FP32,
                  std::runtime_error)
      << "[Conv2D] Q4_0 save only supports FP32 source weight.";

    // A conv FP32 filter is [out_ch, in_ch, kh, kw] in NCHW, i.e. already
    // row-major [out_ch, CRS] (CRS = in_ch*kh*kw) = [N rows, K cols]. This is
    // exactly the layout quantize_q4_0 consumes (N rows of K), so no transpose
    // is needed (the FC path in the base class transposes because its weight is
    // stored [K, N]). The bias and any non-matmul weight (CRS == 1 or
    // out_ch == 1) are kept FP32.
    const TensorDim dim = weight.getDim();
    const unsigned int out_ch = dim.batch();
    const unsigned int CRS = dim.channel() * dim.height() * dim.width();

    if (out_ch <= 1 || CRS <= 1 || out_ch % 32 != 0 || CRS % 32 != 0) {
      // Not Q4_0-eligible (bias, or block-misaligned): keep FP32 so the saved
      // tensor still matches what the runtime layer allocates for it.
      weight.save(file);
      continue;
    }

    // [1, 1, K=CRS, N=out_ch] is the matmul-weight shape the quantized conv
    // consumes at load (Conv2DLayer::finalize builds the same shape).
    Tensor quant_weight(1, 1, CRS, out_ch, {Tformat::NCHW, dtype});
    std::vector<char> tmp(quant_weight.size());

    quantize_q4_0(weight.getData<float>(), tmp.data(), out_ch, CRS, nullptr);
    repack_q4_0(quant_weight.getData<uint8_t>(), tmp.data(),
                quant_weight.size(), out_ch, CRS, target_isa);
    quant_weight.save(file);
  }
}

void Conv2DLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, conv_props);
  LayerImpl::setProperty(remain_props);
}

} /* namespace nntrainer */
