// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   c2psa_layer.cpp
 * @date   18 June 2026
 * @brief  PSA spatial multi-head attention custom layer (weight-free).
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs
 */

#include <cmath>

#include "c2psa_layer.h"

#include <layer_context.h>
#include <nntrainer_error.h>
#include <tensor_dim.h>

namespace yolov11 {

PSAAttentionLayer::PSAAttentionLayer() : dim_(0), scratch_idx_{}, sm() {}

void PSAAttentionLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "PSAAttentionLayer expects exactly 1 input (qkv)";

  const nntrainer::TensorDim &in_dim = context.getInputDimensions()[0];
  dim_ = in_dim.channel();

  const unsigned int expected = NUM_HEADS * KD * 2 + NUM_HEADS * VD; // 512
  NNTR_THROW_IF(dim_ != expected, std::invalid_argument)
    << "PSAAttentionLayer expects " << expected << " input channels, got "
    << dim_;

  const unsigned int B = in_dim.batch();
  const unsigned int N = in_dim.height() * in_dim.width();
  const unsigned int nh = NUM_HEADS;
  const unsigned int kd = KD;
  const unsigned int vd = VD;
  const unsigned int v_ch = nh * vd; // 256

  nntrainer::TensorDim out_dim = in_dim;
  out_dim.channel(v_ch);
  context.setOutputDimensions({out_dim});

  // Scratch tensors carry the activation dtype of this layer so the
  // dotBatched matmuls dispatch to the FP16 GEMM backend for FP16 models.
  // Layout is [B*nh, 1, N, kd|vd] / [B*nh, 1, N, N] — the leading axis is the
  // batch*heads collapse that dotBatched iterates over.
  //
  // The scratch layout is pinned to NCHW regardless of the model's input
  // format. These tensors are private to this layer's own matmuls (Q·Kᵀ,
  // score·V) and are filled via the format-aware getValue/setValue gather, so
  // they carry no external layout obligation. Pinning NCHW matters because the
  // dotBatched → HalfTensor::dot → dotHalf → hgemm kernel treats the operands
  // as contiguous NCHW (it reads the raw pointer and derives M/N/K from the
  // logical dims); an NHWC-typed scratch with non-trivial strides made hgemm
  // walk out of bounds under FP16-act + NHWC models (SEGV_ACCERR).
  const ml::train::TensorDim::TensorType act_type = {
    ml::train::TensorDim::Format::NCHW, in_dim.getDataType()};

  auto planar_dim = [&](unsigned int last) -> nntrainer::TensorDim {
    nntrainer::TensorDim d(act_type);
    d.batch(B * nh);
    d.channel(1);
    d.height(N);
    d.width(last);
    return d;
  };

  scratch_idx_[static_cast<unsigned int>(ScratchIdx::PROJ_Q)] =
    context.requestTensor(planar_dim(kd), "c2psa_q",
                          nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::ITERATION_LIFESPAN);
  scratch_idx_[static_cast<unsigned int>(ScratchIdx::PROJ_K)] =
    context.requestTensor(planar_dim(kd), "c2psa_k",
                          nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::ITERATION_LIFESPAN);
  scratch_idx_[static_cast<unsigned int>(ScratchIdx::PROJ_V)] =
    context.requestTensor(planar_dim(vd), "c2psa_v",
                          nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::ITERATION_LIFESPAN);
  scratch_idx_[static_cast<unsigned int>(ScratchIdx::SCORE)] =
    context.requestTensor(planar_dim(N), "c2psa_score",
                          nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::ITERATION_LIFESPAN);
  scratch_idx_[static_cast<unsigned int>(ScratchIdx::PROJ_OUT)] =
    context.requestTensor(planar_dim(vd), "c2psa_out",
                          nntrainer::Initializer::NONE, false,
                          nntrainer::TensorLifespan::ITERATION_LIFESPAN);

  // dtype-aware softmax: FP32 path uses ACT_SOFTMAX; the FP16 path uses the
  // FP16-typed softmax (FP16 in/out, FP32 accumulation internally), mirroring
  // multi_head_attention_layer.cpp:160-168.
#ifdef ENABLE_FP16
  if (in_dim.getDataType() == ml::train::TensorDim::DataType::FP16) {
    sm.setActiFunc<_FP16>(nntrainer::ActivationType::ACT_SOFTMAX);
  } else
#endif
  {
    sm.setActiFunc(nntrainer::ActivationType::ACT_SOFTMAX);
  }
}

void PSAAttentionLayer::forwarding(nntrainer::RunLayerContext &context,
                                   bool training) {
  (void)training;
  const nntrainer::Tensor &in_t = context.getInput(0);
  nntrainer::Tensor &out_t = context.getOutput(0);

  const nntrainer::Tdatatype dtype = in_t.getDataType();
#ifdef ENABLE_FP16
  const bool fp16 = (dtype == nntrainer::Tdatatype::FP16);
#else
  const bool fp16 = false;
#endif
  NNTR_THROW_IF(dtype != nntrainer::Tdatatype::FP32 && !fp16,
                std::invalid_argument)
    << "PSAAttentionLayer only implements the FP32 and FP16 activation paths";

  const unsigned int B = in_t.batch();
  const unsigned int H = in_t.height();
  const unsigned int W = in_t.width();
  const unsigned int N = H * W;
  const unsigned int nh = NUM_HEADS;
  const unsigned int kd = KD;
  const unsigned int vd = VD;
  const unsigned int v_ch = nh * vd; // 256

  // Ultralytics per-head interleaved qkv layout: channel h*128 + [0:32]=Q,
  // [32:64]=K, [64:128]=V. Gather into head-major planar scratch tensors
  // [B*nh, 1, N, kd|vd] at element (bn=h+nh*b, d, p). The gather/scatter is a
  // strided index rewrite (read qkv[b, cbase+d_off, p]), staged in the
  // requested scratch tensors — no heap vector, no raw sgemm.
  nntrainer::Tensor &Q = context.getTensor(
    scratch_idx_[static_cast<unsigned int>(ScratchIdx::PROJ_Q)]);
  nntrainer::Tensor &K = context.getTensor(
    scratch_idx_[static_cast<unsigned int>(ScratchIdx::PROJ_K)]);
  nntrainer::Tensor &V = context.getTensor(
    scratch_idx_[static_cast<unsigned int>(ScratchIdx::PROJ_V)]);
  nntrainer::Tensor &score = context.getTensor(
    scratch_idx_[static_cast<unsigned int>(ScratchIdx::SCORE)]);
  nntrainer::Tensor &attn = context.getTensor(
    scratch_idx_[static_cast<unsigned int>(ScratchIdx::PROJ_OUT)]);

  const unsigned int head_stride = 2 * kd + vd; // 128

  // Gather Q/K/V from the interleaved qkv into the planar scratch tensors
  // [B*nh, 1, N, kd|vd]. getValue/setValue take (b, c, h, w) and TensorBase::
  // getIndex maps those to the stored stride order for either NCHW or NHWC, so
  // the same (b, chead, h, w) indexing works for both formats: chead is the
  // qkv channel, h=p/W, w=p%W. No heap vector, no raw sgemm.
#ifdef ENABLE_FP16
  if (fp16) {
    for (unsigned int b = 0; b < B; ++b)
      for (unsigned int h = 0; h < nh; ++h) {
        const unsigned int cbase = h * head_stride;
        const unsigned int bn = b * nh + h;
        for (unsigned int p = 0; p < N; ++p) {
          const unsigned int hh = p / W, ww = p % W;
          for (unsigned int d = 0; d < kd; ++d) {
            _FP16 qv = in_t.getValue<_FP16>(b, cbase + d, hh, ww);
            _FP16 kv = in_t.getValue<_FP16>(b, cbase + kd + d, hh, ww);
            Q.setValue(bn, 0, p, d, qv);
            K.setValue(bn, 0, p, d, kv);
          }
          for (unsigned int d = 0; d < vd; ++d) {
            _FP16 vv = in_t.getValue<_FP16>(b, cbase + 2 * kd + d, hh, ww);
            V.setValue(bn, 0, p, d, vv);
          }
        }
      }
  } else
#endif
  {
    for (unsigned int b = 0; b < B; ++b)
      for (unsigned int h = 0; h < nh; ++h) {
        const unsigned int cbase = h * head_stride;
        const unsigned int bn = b * nh + h;
        for (unsigned int p = 0; p < N; ++p) {
          const unsigned int hh = p / W, ww = p % W;
          for (unsigned int d = 0; d < kd; ++d) {
            float qv = in_t.getValue<float>(b, cbase + d, hh, ww);
            float kv = in_t.getValue<float>(b, cbase + kd + d, hh, ww);
            Q.setValue(bn, 0, p, d, qv);
            K.setValue(bn, 0, p, d, kv);
          }
          for (unsigned int d = 0; d < vd; ++d) {
            float vv = in_t.getValue<float>(b, cbase + 2 * kd + d, hh, ww);
            V.setValue(bn, 0, p, d, vv);
          }
        }
      }
  }

  // Scaled dot-product attention, all in Tensor ops. Q,K are [B*nh,1,N,kd];
  // score = Q·Kᵀ -> [B*nh,1,N,N]. The scale is 1/sqrt(kd)=1/sqrt(32) (NOT
  // 1/sqrt(8)) — the per-head key dim, matching the prior implementation and
  // multi_head_attention_layer.cpp:458-459.
  Q.dotBatched(K, score, false, true);
  score.multiply_i(1.0f / std::sqrt(static_cast<float>(kd)));

  // softmax over the key axis (width=N) for each query row.
  sm.run_fn(score, score);

  // attn = score · V -> [B*nh,1,N,vd].
  score.dotBatched(V, attn, false, false);

  // Scatter [B*nh,1,N,vd] back to the planar output [B, v_ch, N]. The output
  // is NOT interleaved — head h's vd channels map contiguously to output
  // channels [h*vd : h*vd+vd] (the subsequent /proj conv consumes a standard
  // layout). The same (b, c, h, w) indexing handles NCHW and NHWC via getIndex.
#ifdef ENABLE_FP16
  if (fp16) {
    for (unsigned int b = 0; b < B; ++b)
      for (unsigned int h = 0; h < nh; ++h) {
        const unsigned int cbase = h * vd;
        const unsigned int bn = b * nh + h;
        for (unsigned int p = 0; p < N; ++p) {
          const unsigned int hh = p / W, ww = p % W;
          for (unsigned int d = 0; d < vd; ++d) {
            _FP16 av = attn.getValue<_FP16>(bn, 0, p, d);
            out_t.setValue(b, cbase + d, hh, ww, av);
          }
        }
      }
  } else
#endif
  {
    for (unsigned int b = 0; b < B; ++b)
      for (unsigned int h = 0; h < nh; ++h) {
        const unsigned int cbase = h * vd;
        const unsigned int bn = b * nh + h;
        for (unsigned int p = 0; p < N; ++p) {
          const unsigned int hh = p / W, ww = p % W;
          for (unsigned int d = 0; d < vd; ++d) {
            float av = attn.getValue<float>(bn, 0, p, d);
            out_t.setValue(b, cbase + d, hh, ww, av);
          }
        }
      }
  }
}

} // namespace yolov11
