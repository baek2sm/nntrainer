// SPDX-License-Identifier: Apache-2.0
/**
 * @file        q8_0_tw_tensor.cpp
 * @date        02 July 2026
 * @brief       Q8_0_TW_Tensor implementation (flat int8 payload, scale in graph
 *              metadata).
 * @see         https://github.com/nntrainer/nntrainer
 * @author      SeungBaek Hong <sb92.hong@samsung.com>
 * @bug         No known bugs except for NYI items
 */

#include <q8_0_tw_tensor.h>
#include <tensor.h>

/**
 * @brief Namespace for nntrainer core components
 */
namespace nntrainer {

Q8_0_TW_Tensor::Q8_0_TW_Tensor(std::string name_, Tformat fm) :
  TensorBase(name_, fm) {
  offset = 0;
}

Q8_0_TW_Tensor::Q8_0_TW_Tensor(const TensorDim &d, bool alloc_now,
                               Initializer init, std::string name) :
  TensorBase(d, false, init, name) {
  if (alloc_now)
    allocate();
  offset = 0;
}

Q8_0_TW_Tensor::Q8_0_TW_Tensor(const TensorDim &d, const void *buf) :
  Q8_0_TW_Tensor(d, true, Initializer::NONE, "") {
  if (d.getDataLen() != 0 && buf != nullptr) {
    NNTR_THROW_IF(!contiguous, std::invalid_argument)
      << getName() << " is not contiguous, cannot copy.";
    if (buf != getData())
      scopy(size(), (int8_t *)buf, 1, (int8_t *)getData(), 1);
  }
}

void Q8_0_TW_Tensor::allocate() {
  if (empty() || data)
    return;

  if (src_tensor) {
    allocateSrcTensor();
  } else {
    MemoryData *mem_data = new MemoryData((void *)(new int8_t[size()]{}));
    data = std::shared_ptr<MemoryData>(mem_data, [](auto *mem_data) {
      delete[] mem_data->template getAddr<int8_t>();
      delete mem_data;
    });
    offset = 0;
    initialize();
  }
}

void *Q8_0_TW_Tensor::getData() const {
  if (!data)
    return nullptr;
  data->validate();
  // Payload is a flat int8 array (1 byte per element) and `offset` is stored in
  // elements, so the element offset equals the byte offset directly.
  return data->getAddr<int8_t>() + offset;
}

void *Q8_0_TW_Tensor::getData(size_t idx) const {
  if (!data)
    return nullptr;
  data->validate();
  return data->getAddr<int8_t>() + offset + idx;
}

void *Q8_0_TW_Tensor::getAddress(unsigned int i) {
  return (void *)((int8_t *)getData() + i);
}

const void *Q8_0_TW_Tensor::getAddress(unsigned int i) const {
  return (const void *)((int8_t *)getData() + i);
}

size_t Q8_0_TW_Tensor::size() const { return dim.getDataLen(); }

size_t Q8_0_TW_Tensor::getMemoryBytes() const {
  return size() * sizeof(int8_t);
}

void Q8_0_TW_Tensor::setZero() {
  int8_t *d = (int8_t *)getData();
  std::fill(d, d + size(), 0);
}

void Q8_0_TW_Tensor::initialize() {
  if (empty() || !isAllocated())
    return;
  setZero();
  putData();
}

QScheme Q8_0_TW_Tensor::q_scheme() const { return QScheme::Q8_0; }

} // namespace nntrainer
