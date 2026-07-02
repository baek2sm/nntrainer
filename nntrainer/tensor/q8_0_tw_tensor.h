// SPDX-License-Identifier: Apache-2.0
/**
 * @file        q8_0_tw_tensor.h
 * @date        02 July 2026
 * @brief       Q8_0_TW_Tensor: tensor-wise static Q8_0 int8 activation tensor.
 * @see         https://github.com/nntrainer/nntrainer
 * @author      SeungBaek Hong <sb92.hong@samsung.com>
 * @bug         No known bugs except for NYI items
 *
 * Unlike Q8_0_Tensor (GGML block layout: fp16 scale + 32 int8 per 32-element
 * block), Q8_0_TW stores a *flat int8 payload only*. The per-tensor
 * quantization scale is graph metadata (kept in the scale table, not embedded
 * in the tensor memory), so the buffer is exactly getDataLen() bytes and the
 * shape carries no block-size (QK8_0) constraint. This is the activation
 * carrier for the W4A8 NHWC static-calibration path.
 *
 * Mutating / GEMM operations are NYI for now (throw); this class is introduced
 * as an unused dtype and wired in incrementally.
 */

#ifndef __Q8_0_TW_TENSOR_H__
#define __Q8_0_TW_TENSOR_H__
#ifdef __cplusplus

#include <quantizer.h>
#include <tensor_base.h>

namespace nntrainer {

/**
 * @class Q8_0_TW_Tensor
 * @brief Holder for a tensor whose data is a flat int8 payload; the per-tensor
 *        scale lives in graph metadata rather than in the buffer.
 */
class Q8_0_TW_Tensor : public TensorBase {

public:
  Q8_0_TW_Tensor(std::string name_ = "", Tformat fm = Tformat::NCHW);

  Q8_0_TW_Tensor(const TensorDim &d, bool alloc_now,
                 Initializer init = Initializer::NONE, std::string name = "");

  Q8_0_TW_Tensor(const TensorDim &d, const void *buf = nullptr);

  Q8_0_TW_Tensor(TensorBase &rhs) : TensorBase(rhs) {}

  void allocate() override;

  void deallocate() override {
    data = nullptr;
    offset = 0;
  }

  void *getData() const override;

  void *getData(size_t idx) const override;

  void *getAddress(unsigned int i) override;

  const void *getAddress(unsigned int i) const override;

  void setValue(float value) override {
    throw std::invalid_argument("Q8_0_TW_Tensor::setValue() is not supported.");
  }

  void setValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value) override {
    throw std::invalid_argument("Q8_0_TW_Tensor::setValue() is not supported.");
  }

  void addValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value, float beta) override {
    throw std::invalid_argument("Q8_0_TW_Tensor::addValue() is not supported.");
  }

  void setZero() override;

  void initialize(Initializer init) override {
    throw std::invalid_argument(
      "Q8_0_TW_Tensor::initialize(init) is not supported.");
  }

  void initialize() override;

  void print(std::ostream &out) const override {
    throw std::invalid_argument("Q8_0_TW_Tensor::print() is not supported.");
  }

  void copy(const Tensor &from) override {
    throw std::invalid_argument("Q8_0_TW_Tensor::copy() is not supported.");
  }

  void copyData(const Tensor &from) override {
    throw std::invalid_argument("Q8_0_TW_Tensor::copyData() is not supported.");
  }

  void copy_with_stride(const Tensor &input, Tensor &output) override {
    throw std::invalid_argument(
      "Q8_0_TW_Tensor::copy_with_stride() is not supported.");
  }

  float max_abs() const override {
    throw std::invalid_argument("Q8_0_TW_Tensor::max_abs() is not supported.");
  }

  float maxValue() const override {
    throw std::invalid_argument("Q8_0_TW_Tensor::maxValue() is not supported.");
  }

  float minValue() const override {
    throw std::invalid_argument("Q8_0_TW_Tensor::minValue() is not supported.");
  }

  size_t size() const override;

  size_t getMemoryBytes() const override;

  QScheme q_scheme() const override;

private:
  std::string getStringDataType() const override { return "Q8_0_TW"; }

  bool isValid() const override { return true; }
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __Q8_0_TW_TENSOR_H__ */
