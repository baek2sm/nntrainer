// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   c2psa_layer.h
 * @date   18 June 2026
 * @brief  PSA spatial multi-head attention custom layer (weight-free).
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __C2PSA_LAYER_H__
#define __C2PSA_LAYER_H__

#include <array>
#include <string>
#include <vector>

#include <acti_func.h>
#include <layer_devel.h>
#include <layer_impl.h>
#include <node_exporter.h>

namespace yolov11 {

/**
 * @class PSAAttentionLayer
 * @brief Spatial multi-head attention for YOLOv11 C2PSA (weight-free).
 *        Hardcoded nh=4, kd=32, vd=64: input 512 ch (Q128/K128/V256) -> 256 ch.
 *
 * Conformance: the attention math is expressed with nntrainer Tensor ops
 * (dotBatched for the QK^T and attention·V matmuls, ActiFunc softmax, in-place
 * scalar multiply for the 1/sqrt(kd) scale), with every intermediate staged in
 * MemoryPool scratch tensors requested in finalize() — no per-forward dynamic
 * allocation and no direct sgemm/kernel call. FP16 activations stay FP16 end
 * to end (the QK^T and AV matmuls dispatch to the FP16 GEMM backend through
 * Tensor::dotBatched); only the softmax reduction runs in FP32 internally, as
 * the framework softmax does.
 *
 * score[i,j] = softmax_j( sum_d Q[h,d,i]*K[h,d,j] / sqrt(kd) )
 * out[h,d,i] = sum_j score[i,j] * V[h,d,j]
 */
class PSAAttentionLayer : public nntrainer::LayerImpl {
public:
  PSAAttentionLayer();
  ~PSAAttentionLayer() override = default;

  void finalize(nntrainer::InitLayerContext &context) override;
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;
  void calcDerivative(nntrainer::RunLayerContext &context) override {}
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override {}
  const std::string getType() const override { return PSAAttentionLayer::type; }
  void setProperty(const std::vector<std::string> &values) override {
    nntrainer::LayerImpl::setProperty(values);
  }
  bool supportBackwarding() const override { return false; }

  static constexpr const char *type = "psa_attention";

private:
  /** @brief Scratch tensor slots requested in finalize and bound in
   *         forwarding. Kept in the same order as ScratchIdx. */
  enum class ScratchIdx : unsigned int {
    PROJ_Q = 0,  ///< [B*nh, 1, N, kd] gathered Q
    PROJ_K = 1,  ///< [B*nh, 1, N, kd] gathered K
    PROJ_V = 2,  ///< [B*nh, 1, N, vd] gathered V
    SCORE = 3,   ///< [B*nh, 1, N, N]  scaled QK^T / softmax weights
    PROJ_OUT = 4 ///< [B*nh, 1, N, vd] attention output before scatter
  };

  unsigned int dim_; ///< input channels
  static constexpr unsigned int NUM_HEADS = 4;

  std::array<unsigned int, 5> scratch_idx_; ///< scratch slot handles
  nntrainer::ActiFunc sm;                   ///< softmax (dtype-aware)
};

} // namespace yolov11

#endif // __C2PSA_LAYER_H__
