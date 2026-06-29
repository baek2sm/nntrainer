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
  NNTR_THROW_IF(d.batch() != 1 || d.channel() != 1 || d.width() % QK8_0 != 0,
                std::invalid_argument)
    << "Q8_0_Tensor must be 2-dimensional with batch=1, channel=1 and "
       "width divisible by "
    << QK8_0;

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
  // `offset` is stored in *elements* (see TensorBase::allocateSrcTensor, which
  // copies the element offset passed to getSharedDataTensor; TensorDim::
  // getDataLen is element-count). Q8_0 packs 32 elements into Q8_0_SIZE bytes,
  // so convert the element offset to a byte offset before indexing the uint8_t
  // buffer. Slices are always row-aligned (width % QK8_0 == 0), so offset is a
  // whole multiple of QK8_0. Without this, a shared Q8_0 sub-tensor (e.g. an
  // activation slice in mha_core / rms_norm / lm_head) reads the wrong bytes
  // and produces all-zero logits.
  size_t byte_offset = (offset / QK8_0) * Q8_0_SIZE;
  return data->getAddr<uint8_t>() + byte_offset;
}

size_t Q8_0_Tensor::size() const {
  size_t num_blocks = height() * width() / QK8_0;
  return Q8_0_SIZE * num_blocks;
}

size_t Q8_0_Tensor::getMemoryBytes() const { return size() * sizeof(uint8_t); }

void Q8_0_Tensor::copy_q80(const void *buf) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  if (buf == getData())
    return;
  scopy(size(), (uint8_t *)buf, 1, (uint8_t *)getData(), 1);
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
  // `this` is the pre-quantized Q8_0 activation matrix of blocks, `weight` is
  // the Q4_0 filter [CRS, out_ch], `output` is [OH*OW, out_ch] FP16. We call the
  // direct Q8_0 backend op (Task B2), bypassing all FP16 dequantization and
  // re-quantization steps.
  const void *in = (const void *)getData();
  uint8_t *wdata = weight.getData<uint8_t>();
  _FP16 *rdata = output.getData<_FP16>();

  unsigned int M = output.getDim().height();
  unsigned int N = output.getDim().width();
  unsigned int K =
    (unsigned int)geom.in_ch * (unsigned int)geom.k_h * (unsigned int)geom.k_w;

  getOps()->gemm_q4_0_indirect_conv_q8_0(M, N, K, in, geom, (void *)wdata, N,
                                         rdata, N);
  return output;
#else
  throw std::invalid_argument("Q8_0_Tensor::convQ4_0Indirect() is not supported on this platform.");
#endif
}

Tensor &Q8_0_Tensor::dot(Tensor const &input, Tensor &output, bool trans,
                         bool trans_in, float beta) const {
#ifdef ENABLE_FP16
  // `this` is Q8_0 activation matrix [M, K], `input` is Q4_0 filter [K, N].
  // `output` is [M, N] FP16. We call the direct Q8_0 backend op (Task B2),
  // bypassing all FP16 dequantization and re-quantization steps.
  NNTR_THROW_IF(trans || trans_in, std::invalid_argument)
    << "Q8_0_Tensor::dot does not support trans/trans_in";

  const void *A = getData();
  const void *B = input.getData();
  _FP16 *C = output.getData<_FP16>();

  unsigned int M = getDim().height();
  unsigned int K = getDim().width();
  unsigned int N = input.getDim().width();

  const unsigned int blocks_per_4_rows = (K + 32 - 1) / 32;
  const unsigned int qa_4_rows_size = (sizeof(uint16_t) * 4 + 128) * blocks_per_4_rows; // block_q8_0x4 size = 136 bytes
  const size_t qa_row_size = 34 * K / 32; // block_q8_0 size = 34 bytes
  const unsigned int M4 = M / 4;
  const unsigned int rem = M % 4;

  const unsigned int qa_size =
    qa_4_rows_size * M4 + static_cast<unsigned int>(qa_row_size) * rem;
  std::vector<char> QA(qa_size);
  char *QA_ptr = QA.data();

  struct local_block_q8_0 {
    uint16_t d;
    int8_t qs[32];
  };

  struct local_block_q8_0x4 {
    uint16_t d[4];
    int8_t qs[128];
  };

  // Interleave Q8_0 activation rows directly into QA_ptr
  const local_block_q8_0 *in_q80 = (const local_block_q8_0 *)A;
  local_block_q8_0x4 *y_q80x4 = (local_block_q8_0x4 *)QA_ptr;

  unsigned int max_blocks = (M / 4) * 4 * blocks_per_4_rows;

  auto &tm = ThreadManager::Global();

  const unsigned int interleave_chunk_size = 256;
  const size_t interleave_loops = (M4 + interleave_chunk_size - 1) / interleave_chunk_size;

  tm.parallel_for(0, interleave_loops, [=, &y_q80x4](size_t idx) {
    unsigned int r_start = idx * interleave_chunk_size;
    unsigned int r_end = std::min(r_start + interleave_chunk_size, M4);

    for (unsigned int r = r_start; r < r_end; ++r) {
      for (unsigned int b = 0; b < blocks_per_4_rows; ++b) {
        unsigned int dst_idx = r * blocks_per_4_rows + b;
        local_block_q8_0x4 &dst = y_q80x4[dst_idx];
        for (unsigned int i = 0; i < 4; ++i) {
          unsigned int src_idx = (r * 4 + i) * blocks_per_4_rows + b;
          if (src_idx >= max_blocks) {
            continue;
          }
          const local_block_q8_0 &src = in_q80[src_idx];
          dst.d[i] = src.d;
          std::memcpy(&dst.qs[32 * i], src.qs, 32);
        }
      }
    }
  });

  // Remainder rows: copy them directly
  if (rem > 0) {
    char *rem_dst = QA_ptr + M4 * qa_4_rows_size;
    const char *rem_src = (const char *)A + M4 * 4 * qa_row_size;
    std::memcpy(rem_dst, rem_src, rem * qa_row_size);
  }

  const unsigned int A_step = sizeof(local_block_q8_0) * (K / 32);
  const unsigned int B_step = 18 * (K / 32); // block_q4_0 size = 18 bytes

  if (M4 > 0) {
    const unsigned int row_chunk_size = 16;
    const size_t row_loop = (M4 * 4 + row_chunk_size - 1) / row_chunk_size;
    const unsigned int col_chunk_size = 16;
    const size_t col_loop = (N + col_chunk_size - 1) / col_chunk_size;

    tm.parallel_for(0, col_loop * row_loop, [=](size_t i) {
      unsigned int r = i / col_loop;
      unsigned int c = i % col_loop;

      unsigned int r_start = r * row_chunk_size;
      unsigned int r_end = std::min(row_chunk_size * (r + 1), M4 * 4);

      unsigned int c_start = c * col_chunk_size;
      unsigned int c_end = std::min(col_chunk_size * (c + 1), N);

#if defined(__ARM_NEON)
      nntr_gemm_q4_0_4x8_q8_0_fp16(K, (_FP16 *)(C + r_start * N + c_start), N,
                                   (void *)((char *)B + c_start * B_step),
                                   (void *)(QA_ptr + r_start * A_step),
                                   r_end - r_start, c_end - c_start);
#else
      unsigned int t_rows = r_end - r_start;
      unsigned int t_cols = c_end - c_start;
      std::vector<float> tile(t_rows * t_cols);
      nntr_gemm_q4_0_4x8_q8_0(K, tile.data(), t_cols,
                              (void *)((char *)B + c_start * B_step),
                              (void *)(QA_ptr + r_start * A_step), t_rows,
                              t_cols);
      for (unsigned int ii = 0; ii < t_rows; ++ii) {
        for (unsigned int jj = 0; jj < t_cols; ++jj) {
          C[(r_start + ii) * N + c_start + jj] = (_FP16)tile[ii * t_cols + jj];
        }
      }
#endif
    });
  }

  // Leftover rows (M % 4) GEMV
  if (rem > 0) {
    std::vector<float> tail32(rem * (size_t)N);
    unsigned int chunk_size = 16;
    unsigned int loop = (N + chunk_size - 1) / chunk_size;
    for (unsigned int pb = M4 * 4; pb < M; pb++) {
      tm.parallel_for(0, loop, [=, &tail32](size_t idx) {
        unsigned int M_step_start = chunk_size * idx;
        unsigned int M_step_end = std::min(chunk_size * (idx + 1), (size_t)N);

        nntr_gemv_q4_0_4x8_q8_0(
          K, (float *)(tail32.data() + (pb - M4 * 4) * N) + M_step_start, N,
          (void *)((char *)B + M_step_start * B_step),
          QA_ptr + (M4 * qa_4_rows_size) + (pb - M4 * 4) * qa_row_size, 1,
          M_step_end - M_step_start);
      });
    }
    for (unsigned int ii = 0; ii < rem; ++ii) {
      for (unsigned int jj = 0; jj < N; ++jj) {
        C[(M4 * 4 + ii) * N + jj] = (_FP16)tail32[ii * N + jj];
      }
    }
  }

  return output;
#else
  throw std::invalid_argument("Q8_0_Tensor::dot() is not supported on this platform.");
#endif
}

} // namespace nntrainer
