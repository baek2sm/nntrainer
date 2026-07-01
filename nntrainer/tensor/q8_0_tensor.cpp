// SPDX-License-Identifier: Apache-2.0
/**
 * @file        q8_0_tensor.cpp
 * @date        14 May 2026
 * @brief       Q8_0_Tensor implementation (mirror of q4_0_tensor.cpp).
 * @see         https://github.com/nntrainer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs except for NYI items
 */

#include <cpu_backend.h>
#include <q8_0_tensor.h>
#include <tensor.h>
#include <thread_manager.h>
#include <ggml_interface.h>
#include <nntr_ggml_impl.h>

namespace {
/**
 * @brief Local mirror of the block_q8_0x4 compute layout.
 *
 * Defined here instead of including nntr_ggml_impl_common.h to avoid the
 * Q8_0=32 macro colliding with Tdatatype::Q8_0 / QScheme::Q8_0.
 */
struct block_q8_0x4_local {
  uint16_t d[4];
  int8_t qs[128];
};

struct block_q4_0_local {
  uint16_t d;
  uint8_t qs[16];
};
} // namespace

/**
 * @brief Namespace for nntrainer core components
 */
namespace nntrainer {

Q8_0_Tensor::Q8_0_Tensor(std::string name_, Tformat fm) :
  TensorBase(name_, fm) {
  offset = 0;
}

Q8_0_Tensor::Q8_0_Tensor(const TensorDim &d, bool alloc_now, Initializer init,
                         std::string name) :
  TensorBase(d, false, init, name) {
  /// @note Tensor-wise Q8_0 stores a single scale for the whole tensor, so the
  /// only alignment requirement is that the total element count be padded to
  /// a multiple of QK8_0 for the int8 kernels. No per-axis divisibility rule.
  if (alloc_now)
    allocate();
  offset = 0;
}

Q8_0_Tensor::Q8_0_Tensor(const TensorDim &d, const void *buf) :
  Q8_0_Tensor(d, true, Initializer::NONE, "") {
  if (d.getDataLen() != 0) {
    if (buf != nullptr)
      copy_q80(buf);
  }
}

Q8_0_Tensor::Q8_0_Tensor(const TensorDim &d, void *external_buf) :
  TensorBase(d, false, Initializer::NONE, "") {
  data = std::make_shared<MemoryData>(external_buf);
  offset = 0;
}

void Q8_0_Tensor::allocate() {
  if (empty() || data)
    return;

  if (src_tensor) {
    allocateSrcTensor();
  } else {
    MemoryData *mem_data = new MemoryData((void *)(new uint8_t[size()]{}));
    data = std::shared_ptr<MemoryData>(mem_data, [](auto *mem_data) {
      delete[] mem_data->template getAddr<uint8_t>();
      delete mem_data;
    });
    offset = 0;
    initialize();
  }
}

void *Q8_0_Tensor::getData() const {
  if (!data)
    return nullptr;
  data->validate();
  // Tensor-wise Q8_0 layout: uint16_t d (2 bytes), then int8 qs[]. `offset` is
  // an element offset into qs[], so the byte offset is 2 (scale) + offset.
  // The tail of the tensor is padded to a multiple of 32; sub-tensors always
  // start at an element offset and share the parent's scale at offset 0, which
  // is fine because tensor-wise Q8_0 uses a single scale for the whole tensor.
  return data->getAddr<uint8_t>() + sizeof(uint16_t) + offset;
}

size_t Q8_0_Tensor::size() const {
  size_t nelem = getDim().getDataLen();
  size_t n_pad = (nelem + QK8_0 - 1) / QK8_0 * QK8_0;
  return sizeof(uint16_t) + n_pad;
}

size_t Q8_0_Tensor::getMemoryBytes() const {
  // TensorPlanner sizes memory from getMemoryBytes(); it must equal size() for
  // Q8_0_Tensor so requestTensor allocates exactly the byte buffer we write.
  return size();
}

void Q8_0_Tensor::copy_q80(const void *buf) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  if (buf == getData())
    return;
  scopy(size(), (uint8_t *)buf, 1, (uint8_t *)getData(), 1);
}

/**
 * @brief Tensor-wise quantize a contiguous FP16/FP32 buffer into this Q8_0 tensor.
 */
template <typename T>
static void quantize_tensor_wise(const T *src, size_t nelem, uint8_t *dst) {
  float amax = 0.0f;
  for (size_t i = 0; i < nelem; ++i) {
    float v = std::abs(static_cast<float>(src[i]));
    if (v > amax)
      amax = v;
  }
  float d = amax / 127.0f;
  if (d == 0.0f)
    d = 1.0f;
  _FP16 d_h = static_cast<_FP16>(d);
  uint16_t d_u16;
  std::memcpy(&d_u16, &d_h, 2);
  std::memcpy(dst, &d_u16, sizeof(uint16_t));

  float id = 1.0f / d;
  int8_t *qs = reinterpret_cast<int8_t *>(dst + sizeof(uint16_t));
  size_t i = 0;
  for (; i < nelem; ++i) {
    float v = static_cast<float>(src[i]) * id;
    v = std::round(v);
    if (v > 127.0f)
      v = 127.0f;
    if (v < -127.0f)
      v = -127.0f;
    qs[i] = static_cast<int8_t>(v);
  }
  size_t n_pad = (nelem + QK8_0 - 1) / QK8_0 * QK8_0;
  for (; i < n_pad; ++i)
    qs[i] = 0;
}

void Q8_0_Tensor::copyData(const Tensor &from) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copyData.";

  if (from.getDataType() == Tdatatype::Q8_0) {
    NNTR_THROW_IF(getDim() != from.getDim(), std::invalid_argument)
      << "Q8_0 copyData shape mismatch: " << getDim() << " vs "
      << from.getDim();
    const uint8_t *src = static_cast<const uint8_t *>(from.getData<void>());
    uint8_t *dst = static_cast<uint8_t *>(getData());
    size_t n = size();
    // Both tensors share the tensor-wise layout: 2-byte scale + qs[]. Since the
    // scale is the first two bytes at offset 0, getData() already points past
    // it. Copy from the very beginning of the source buffer so the scale moves.
    // Adjust pointers back by the scale header.
    std::memcpy(dst - sizeof(uint16_t), src - sizeof(uint16_t), n);
    return;
  }

  if (from.getDataType() == Tdatatype::FP16) {
    const _FP16 *src = from.getData<_FP16>();
    quantize_tensor_wise(src, getDim().getDataLen(),
                              static_cast<uint8_t *>(getData()));
    return;
  }

  NNTR_THROW_IF(from.getDataType() != Tdatatype::FP32, std::invalid_argument)
    << "Q8_0_Tensor::copyData supports FP16/FP32/Q8_0 source only";

  const float *src = from.getData<float>();
  quantize_tensor_wise(src, getDim().getDataLen(),
                            static_cast<uint8_t *>(getData()));
}

void Q8_0_Tensor::setZero() {
  uint8_t *d = (uint8_t *)getData();
  std::fill(d, d + size(), 0);
}

void Q8_0_Tensor::initialize() {
  if (empty() || !isAllocated())
    return;
  setZero();
  putData();
}

QScheme Q8_0_Tensor::q_scheme() const { return QScheme::Q8_0; }

Tensor &Q8_0_Tensor::convQ4_0Indirect(Tensor const &weight, Tensor &output,
                                     const ConvGatherParams &geom) const {
#ifdef ENABLE_FP16
  // `this` is a tensor-wise Q8_0 activation. The single scale lives at the
  // beginning of the byte buffer; getData() returns the qs[] base.
  uint8_t *storage = static_cast<uint8_t *>(getData()) - sizeof(uint16_t);
  uint16_t a_scale_u16;
  std::memcpy(&a_scale_u16, storage, sizeof(uint16_t));
  _FP16 a_scale_h;
  std::memcpy(&a_scale_h, &a_scale_u16, sizeof(uint16_t));

  const void *in = (const void *)getData();
  uint8_t *wdata = weight.getData<uint8_t>();
  _FP16 *rdata = output.getData<_FP16>();

  unsigned int M = output.getDim().height();
  unsigned int N = output.getDim().width();
  unsigned int K =
    (unsigned int)geom.in_ch * (unsigned int)geom.k_h * (unsigned int)geom.k_w;

  __ggml_q4_0_4x8_q8_0_indirect_GEMM_tq8_0(M, N, K, in, geom, (void *)wdata, N,
                                           rdata, N, a_scale_h);
  return output;
#else
  throw std::invalid_argument("Q8_0_Tensor::convQ4_0Indirect() is not supported on this platform.");
#endif
}

/**
 * @brief Tensor-wise Q8_0 act [M,K] x Q4_0 weight [K,N] -> FP16 out [M,N].
 *
 * The persistent Q8_0_Tensor stores one fp16 scale + flat int8 qs[]. This
 * function repacks qs[] into the compute-only block_q8_0x4 layout (qs only,
 * scales passed separately) and calls the tensor-wise int8 kernel. No FP16
 * dequant/re-quant happens outside the kernel.
 */
Tensor &Q8_0_Tensor::dot(Tensor const &input, Tensor &output, bool trans,
                         bool trans_in, float beta) const {
#ifdef ENABLE_FP16
  NNTR_THROW_IF(trans || trans_in, std::invalid_argument)
    << "Q8_0_Tensor::dot does not support trans/trans_in";
  (void)beta;

  uint8_t *storage = static_cast<uint8_t *>(getData()) - sizeof(uint16_t);
  uint16_t a_scale_u16;
  std::memcpy(&a_scale_u16, storage, sizeof(uint16_t));

  const int8_t *A_qs = static_cast<const int8_t *>(getData());
  const void *B = input.getData();
  _FP16 *C = output.getData<_FP16>();

  unsigned int M = getDim().height();
  unsigned int K = getDim().width();
  unsigned int N = input.getDim().width();

  const unsigned int nb = K / 32;
  const unsigned int M4 = M / 4;
  const unsigned int rem = M % 4;

  // Compute buffer: standard block_q8_0x4 (8-byte scale header + 128-byte qs).
  // Since activations are tensor-wise Q8_0, every block's d[0..3] is set to the
  // same scale, so the existing nntr_gemm_q4_0_4x8_q8_0_fp16 kernel can consume
  // it without a dedicated tensor-wise micro-kernel.
  const unsigned int qa_4_rows_size = sizeof(block_q8_0x4_local) * nb;
  const unsigned int qa_row_size = sizeof(block_q8_0) * nb;
  const unsigned int qa_size = qa_4_rows_size * M4 + qa_row_size * rem;
  std::vector<char> QA(qa_size);
  char *QA_ptr = QA.data();

  auto &tm = ThreadManager::Global();
  const unsigned int interleave_chunk = 256;
  const size_t interleave_loops =
    (M4 + interleave_chunk - 1) / interleave_chunk;
  tm.parallel_for(0, interleave_loops, [=](size_t idx) {
    unsigned int g0 = idx * interleave_chunk;
    unsigned int g1 = std::min(g0 + interleave_chunk, M4);
    for (unsigned int g = g0; g < g1; ++g) {
      unsigned int r0 = g * 4;
      char *group_base = QA_ptr + g * qa_4_rows_size;
      for (unsigned int b = 0; b < nb; ++b) {
        block_q8_0x4_local *dst = reinterpret_cast<block_q8_0x4_local *>(group_base) + b;
        for (int row = 0; row < 4; ++row) {
          dst->d[row] = a_scale_u16;
          unsigned int r = r0 + row;
          const int8_t *src = A_qs + r * K + b * 32;
          for (int j = 0; j < 4; ++j)
            for (int lane = 0; lane < 8; ++lane)
              dst->qs[32 * j + 8 * row + lane] = src[j * 8 + lane];
        }
      }
    }
  });
  if (rem > 0) {
    char *rem_dst = QA_ptr + M4 * qa_4_rows_size;
    for (unsigned int r = M4 * 4; r < M; ++r) {
      block_q8_0 *row_dst =
        reinterpret_cast<block_q8_0 *>(rem_dst) + (r - M4 * 4) * nb;
      for (unsigned int b = 0; b < nb; ++b) {
        row_dst[b].d = a_scale_u16;
        std::memcpy(row_dst[b].qs, A_qs + r * K + b * 32, 32);
      }
    }
  }

  const unsigned int B_step = sizeof(block_q4_0_local) * nb;

  if (M4 > 0) {
    const unsigned int row_chunk = 16;
    const size_t row_loop = (M4 * 4 + row_chunk - 1) / row_chunk;
    const unsigned int col_chunk = 16;
    const size_t col_loop = (N + col_chunk - 1) / col_chunk;
    tm.parallel_for(0, col_loop * row_loop, [=](size_t i) {
      unsigned int r = i / col_loop;
      unsigned int c = i % col_loop;
      unsigned int r0 = r * row_chunk;
      unsigned int r1 = std::min(row_chunk * (r + 1), M4 * 4);
      unsigned int c0 = c * col_chunk;
      unsigned int c1 = std::min(col_chunk * (c + 1), N);
      unsigned int t_rows = r1 - r0;
      unsigned int t_cols = c1 - c0;
      unsigned int t_rows4 = t_rows & ~3U;
      if (t_rows4 > 0) {
        nntr_gemm_q4_0_4x8_q8_0_fp16(
          K, C + r0 * N + c0, N,
          (void *)((char *)B + c0 * B_step),
          (void *)(QA_ptr + r0 / 4 * qa_4_rows_size), t_rows4, t_cols);
      }
      for (unsigned int rr = r0 + t_rows4; rr < r1; ++rr) {
        nntr_gemv_q4_0_4x8_q8_0_fp16(
          K, C + rr * N + c0, N,
          (void *)((char *)B + c0 * B_step),
          QA_ptr + M4 * qa_4_rows_size + (rr - M4 * 4) * qa_row_size, 1,
          t_cols);
      }
    });
  }
  if (rem > 0) {
    for (unsigned int pb = M4 * 4; pb < M; ++pb) {
      nntr_gemv_q4_0_4x8_q8_0_fp16(
        K, C + pb * N, N, (void *)B,
        QA_ptr + M4 * qa_4_rows_size + (pb - M4 * 4) * qa_row_size, 1, N);
    }
  }
  return output;
#else
  throw std::invalid_argument("Q8_0_Tensor::dot() is not supported on this platform.");
#endif
}

#ifdef ENABLE_FP16
void Q8_0_Tensor::dot_prepacked_x4(unsigned int M, unsigned int K,
                                   unsigned int N, const void *QA,
                                   float a_scale, const void *B, _FP16 *C,
                                   unsigned int ldc) {
  const char *QA_ptr = static_cast<const char *>(QA);
  const unsigned int nb = K / 32;
  const unsigned int M4 = M / 4;
  const unsigned int rem = M % 4;
  const unsigned int qa_4_rows_size = sizeof(block_q8_0x4_local) * nb;
  const unsigned int qa_row_size = sizeof(block_q8_0) * nb;
  const unsigned int B_step = sizeof(block_q4_0_local) * nb;
  auto &tm = ThreadManager::Global();

  (void)a_scale; // scale is already embedded in the pre-packed QA buffer

  if (M4 > 0) {
    const unsigned int row_chunk = 16;
    const size_t row_loop = (M4 * 4 + row_chunk - 1) / row_chunk;
    const unsigned int col_chunk = 16;
    const size_t col_loop = (N + col_chunk - 1) / col_chunk;
    tm.parallel_for(0, col_loop * row_loop, [=](size_t i) {
      unsigned int r = i / col_loop;
      unsigned int c = i % col_loop;
      unsigned int r0 = r * row_chunk;
      unsigned int r1 = std::min(row_chunk * (r + 1), M4 * 4);
      unsigned int c0 = c * col_chunk;
      unsigned int c1 = std::min(col_chunk * (c + 1), N);
      unsigned int t_rows = r1 - r0;
      unsigned int t_cols = c1 - c0;
      unsigned int t_rows4 = t_rows & ~3U;
      if (t_rows4 > 0) {
        nntr_gemm_q4_0_4x8_q8_0_fp16(
          K, C + r0 * ldc + c0, ldc,
          (void *)((const char *)B + c0 * B_step),
          (void *)(QA_ptr + r0 / 4 * qa_4_rows_size), t_rows4, t_cols);
      }
      for (unsigned int rr = r0 + t_rows4; rr < r1; ++rr) {
        nntr_gemv_q4_0_4x8_q8_0_fp16(
          K, C + rr * ldc + c0, ldc,
          (void *)((const char *)B + c0 * B_step),
          QA_ptr + M4 * qa_4_rows_size + (rr - M4 * 4) * qa_row_size, 1,
          t_cols);
      }
    });
  }
  if (rem > 0) {
    for (unsigned int pb = M4 * 4; pb < M; ++pb) {
      nntr_gemv_q4_0_4x8_q8_0_fp16(
        K, C + pb * ldc, ldc, (void *)B,
        QA_ptr + M4 * qa_4_rows_size + (pb - M4 * 4) * qa_row_size, 1, N);
    }
  }
}
#endif

} // namespace nntrainer
