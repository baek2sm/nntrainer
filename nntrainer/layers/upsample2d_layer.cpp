// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 heka1024 <heka1024@gmail.com>
 *
 * @file   upsample2d_layer.h
 * @date   15 June 2024
 * @brief  It is a implementation of upsample layer for given size and
 * interpolation method
 * @see    https://github.com/nntrainer/nntrainer
 * @author heka1024 <heka1024@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#include <layer_context.h>
#include <node_exporter.h>
#include <upsample2d_layer.h>

#include <cstring>

namespace nntrainer {
static constexpr size_t SINGLE_INOUT_IDX = 0;

Upsample2dLayer::Upsample2dLayer() :
  Layer(),
  upsample2d_props(props::UpsampleMode(),
                   std::array<props::KernelSize, UPSAMPLE2D_DIM>()) {}

void Upsample2dLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();

  const auto &kernel_size =
    std::get<std::array<props::KernelSize, UPSAMPLE2D_DIM>>(upsample2d_props);

  for (unsigned int i = 0; i < dim.size(); ++i) {
    if (dim[i].getDataLen() == 0) {
      throw std::invalid_argument("Input dimension is not set");
    } else {
      dim[i].channel(dim[i].channel());
      dim[i].height(dim[i].height() * kernel_size[0]);
      dim[i].width(dim[i].width() * kernel_size[1]);
    }
  }

  context.setOutputDimensions(dim);
}

/**
 * @brief dtype-correct upsample forwarding. getValue()/setValue() default to
 * T=float, so on an FP16 tensor a float read reinterprets two 2-byte halves as
 * one 4-byte float (and indexes at 2x stride) -> garbage. Dispatch on the
 * actual tensor dtype so element access stays correct for any precision. The
 * bilinear interpolation math is carried out in float regardless of storage.
 */
template <typename T>
static void upsampleForwardT(
  const Tensor &in, Tensor &out,
  props::UpsampleModeInfo::Interpolation upsampling_type,
  const std::array<props::KernelSize, UPSAMPLE2D_DIM> &kernel_size) {
  switch (upsampling_type) {
  case props::UpsampleModeInfo::Interpolation::nearest: {
    // Fast path: contiguous NCHW. nearest upsample maps
    // out[b,c, kh*ih+ry, kw*iw+rx] = in[b,c,ih,iw] for ry in [0,kh), rx in
    // [0,kw). Same element mapping as the scalar loop below, written per
    // output row: each input row ih feeds kh output rows, and within a row
    // each input element is repeated kw times. Bit-identical to the scalar
    // path.
    const unsigned int kh = kernel_size[0].get();
    const unsigned int kw = kernel_size[1].get();
    const bool can_fast =
      in.getContiguous() && out.getContiguous() &&
      in.getFormat() == Tformat::NCHW && out.getFormat() == Tformat::NCHW;
    if (can_fast) {
      const T *in_d = in.getData<T>();
      T *out_d = out.getData<T>();
      const unsigned int B = out.batch();
      const unsigned int C = out.channel();
      const unsigned int iH = in.height();
      const unsigned int iW = in.width();
      const unsigned int oW = out.width(); // == iW * kw
      const size_t in_plane = (size_t)iH * iW;   // [H*W] per (b,c)
      const size_t out_plane = (size_t)iH * kh * oW; // [oH*oW] per (b,c)
      // scratch row for the kw-expanded output row (reused per input row)
      std::vector<T> expanded(oW);
      for (unsigned int b = 0; b < B; ++b) {
        for (unsigned int c = 0; c < C; ++c) {
          const T *in_bc = in_d + (size_t)b * C * in_plane + (size_t)c * in_plane;
          T *out_bc = out_d + (size_t)b * C * out_plane + (size_t)c * out_plane;
          for (unsigned int ih = 0; ih < iH; ++ih) {
            const T *in_row = in_bc + (size_t)ih * iW;
            // expand in_row[iW] -> expanded[iW*kw], each elt repeated kw
            T *e = expanded.data();
            for (unsigned int iw = 0; iw < iW; ++iw) {
              const T v = in_row[iw];
              for (unsigned int rx = 0; rx < kw; ++rx)
                e[iw * kw + rx] = v;
            }
            // write the expanded row into the kh output rows it feeds
            for (unsigned int ry = 0; ry < kh; ++ry) {
              T *out_row = out_bc + (size_t)(ih * kh + ry) * oW;
              std::memcpy(out_row, e, oW * sizeof(T));
            }
          }
        }
      }
      return;
    }

    const bool can_fast_nhwc =
      in.getContiguous() && out.getContiguous() &&
      in.getFormat() == Tformat::NHWC && out.getFormat() == Tformat::NHWC;
    if (can_fast_nhwc) {
      const T *in_d = in.getData<T>();
      T *out_d = out.getData<T>();
      const unsigned int B = out.batch();
      const unsigned int Co = out.channel();
      const unsigned int iH = in.height();
      const unsigned int iW = in.width();
      const unsigned int oH = out.height();
      const unsigned int oW = out.width();
      const size_t in_hwc = (size_t)iH * iW * Co;
      const size_t out_hwc = (size_t)oH * oW * Co;

      std::vector<T> expanded(oW * Co);
      for (unsigned int b = 0; b < B; ++b) {
        const T *in_b = in_d + b * in_hwc;
        T *out_b = out_d + b * out_hwc;
        for (unsigned int ih = 0; ih < iH; ++ih) {
          const T *in_row = in_b + (size_t)ih * iW * Co;
          T *e = expanded.data();
          for (unsigned int iw = 0; iw < iW; ++iw) {
            const T *v = in_row + iw * Co;
            for (unsigned int rx = 0; rx < kw; ++rx) {
              std::memcpy(e + (iw * kw + rx) * Co, v, Co * sizeof(T));
            }
          }
          for (unsigned int ry = 0; ry < kh; ++ry) {
            T *out_row = out_b + (size_t)(ih * kh + ry) * oW * Co;
            std::memcpy(out_row, e, oW * Co * sizeof(T));
          }
        }
      }
      return;
    }
    // Direct NHWC scalar fallback: avoids getValue<T> on NHWC, whose index
    // path may use NCHW assumptions and crash. NHWC physical layout is dense
    // [B,H,W,C]; contiguous flag can be unset after reshape, so skip it.
    if (in.getFormat() == Tformat::NHWC && out.getFormat() == Tformat::NHWC) {
      const unsigned int B = out.batch();
      const unsigned int C = out.channel();
      const unsigned int iH = in.height();
      const unsigned int iW = in.width();
      const unsigned int oH = out.height();
      const unsigned int oW = out.width();
      const T *src = in.getData<T>();
      T *dst = out.getData<T>();
      for (unsigned int b = 0; b < B; ++b) {
        for (unsigned int oh = 0; oh < oH; ++oh) {
          unsigned int ih = oh / kh;
          for (unsigned int ow = 0; ow < oW; ++ow) {
            unsigned int iw = ow / kw;
            size_t in_off = (((size_t)b * iH + ih) * iW + iw) * C;
            size_t out_off = (((size_t)b * oH + oh) * oW + ow) * C;
            const T *s = src + in_off;
            T *d = dst + out_off;
            for (unsigned int c = 0; c < C; ++c)
              d[c] = s[c];
          }
        }
      }
      return;
    }
    for (int b = 0; b < (int)out.batch(); b++) {
      for (int c = 0; c < (int)out.channel(); c++) {
        for (int h = 0; h < (int)out.height(); h++) {
          for (int w = 0; w < (int)out.width(); w++) {
            out.getValue<T>(b, c, h, w) =
              in.getValue<T>(b, c, h / kernel_size[0], w / kernel_size[1]);
          }
        }
      }
    }
  } break;
  case props::UpsampleModeInfo::Interpolation::bilinear: {
    float scale_h = (float)kernel_size[0];
    float scale_w = (float)kernel_size[1];

    for (int b = 0; b < (int)out.batch(); b++) {
      for (int c = 0; c < (int)out.channel(); c++) {
        for (int h = 0; h < (int)out.height(); h++) {
          for (int w = 0; w < (int)out.width(); w++) {
            float x_in = (w + 0.5f) / scale_w - 0.5f;
            float y_in = (h + 0.5f) / scale_h - 0.5f;

            if (x_in < 0) {
              x_in = 0.0f;
            }
            if (y_in < 0) {
              y_in = 0.0f;
            }

            int x0 = static_cast<int>(floor(x_in));
            int y0 = static_cast<int>(floor(y_in));
            int x1 = std::min(x0 + 1, (int)in.width() - 1);
            int y1 = std::min(y0 + 1, (int)in.height() - 1);

            float dx = x_in - x0;
            float dy = y_in - y0;

            float top =
              (1.0f - dx) * static_cast<float>(in.getValue<T>(b, c, y1, x0)) +
              dx * static_cast<float>(in.getValue<T>(b, c, y1, x1));
            float bottom =
              (1.0f - dx) * static_cast<float>(in.getValue<T>(b, c, y0, x0)) +
              dx * static_cast<float>(in.getValue<T>(b, c, y0, x1));
            float v = (1.0f - dy) * bottom + dy * top;
            out.getValue<T>(b, c, h, w) = static_cast<T>(v);
          }
        }
      }
    }
  } break;
  default:
    throw std::runtime_error("Error: Unknown Upsample Mode Type");
  }
}

/**
 * @brief Nearest-neighbor upsample for tensor-wise Q8_0 tensors.
 *
 * Nearest replication preserves the exact value range, so the output can share
 * the input's single FP16 scale and simply replicate the int8 qs values.
 */
static void upsampleNearestQ8_0(
  const Tensor &in, Tensor &out,
  const std::array<props::KernelSize, UPSAMPLE2D_DIM> &kernel_size) {
  const unsigned int kh = kernel_size[0].get();
  const unsigned int kw = kernel_size[1].get();

  // Copy the single tensor-wise scale (2 bytes at offset -2).
  const uint8_t *in_storage =
    static_cast<const uint8_t *>(in.getData<void>()) - sizeof(uint16_t);
  uint8_t *out_storage =
    static_cast<uint8_t *>(out.getData<void>()) - sizeof(uint16_t);
  std::memcpy(out_storage, in_storage, sizeof(uint16_t));

  const int8_t *src = in.getData<int8_t>();
  int8_t *dst = out.getData<int8_t>();

  const unsigned int B = in.batch();
  const unsigned int C = in.channel();
  const unsigned int iH = in.height();
  const unsigned int iW = in.width();
  const unsigned int oH = out.height();
  const unsigned int oW = out.width();

  if (in.getFormat() == Tformat::NHWC && out.getFormat() == Tformat::NHWC) {
    for (unsigned int b = 0; b < B; ++b) {
      const size_t in_batch_offset = (size_t)b * iH * iW * C;
      const size_t out_batch_offset = (size_t)b * oH * oW * C;
      for (unsigned int ih = 0; ih < iH; ++ih) {
        for (unsigned int iw = 0; iw < iW; ++iw) {
          const size_t in_off = in_batch_offset + ((size_t)ih * iW + iw) * C;
          for (unsigned int ry = 0; ry < kh; ++ry) {
            const size_t out_off =
              out_batch_offset +
              ((size_t)(ih * kh + ry) * oW + (size_t)iw * kw) * C;
            for (unsigned int rx = 0; rx < kw; ++rx) {
              std::memcpy(dst + out_off + rx * C, src + in_off, C * sizeof(int8_t));
            }
          }
        }
      }
    }
    return;
  }

  // NCHW fallback: channels are planar.
  const size_t in_plane = (size_t)iH * iW;
  const size_t out_plane = (size_t)oH * oW;
  for (unsigned int b = 0; b < B; ++b) {
    for (unsigned int c = 0; c < C; ++c) {
      const int8_t *in_bc = src + ((size_t)b * C + c) * in_plane;
      int8_t *out_bc = dst + ((size_t)b * C + c) * out_plane;
      for (unsigned int ih = 0; ih < iH; ++ih) {
        for (unsigned int iw = 0; iw < iW; ++iw) {
          int8_t v = in_bc[ih * iW + iw];
          for (unsigned int ry = 0; ry < kh; ++ry) {
            size_t out_row = ((size_t)(ih * kh + ry) * oW + (size_t)iw * kw);
            for (unsigned int rx = 0; rx < kw; ++rx) {
              out_bc[out_row + rx] = v;
            }
          }
        }
      }
    }
  }
}

void Upsample2dLayer::forwarding(nntrainer::RunLayerContext &context,
                                 bool training) {
  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);

  const auto &upsampling_type =
    std::get<props::UpsampleMode>(upsample2d_props).get();
  const auto &kernel_size =
    std::get<std::array<props::KernelSize, UPSAMPLE2D_DIM>>(upsample2d_props);

  if (out.getDataType() == ml::train::TensorDim::DataType::Q8_0) {
    NNTR_THROW_IF(upsampling_type != props::UpsampleModeInfo::Interpolation::nearest,
                  std::invalid_argument)
      << "Q8_0 upsample only supports nearest mode";
    upsampleNearestQ8_0(in, out, kernel_size);
    return;
  }

#ifdef ENABLE_FP16
  if (out.getDataType() == ml::train::TensorDim::DataType::FP16) {
    upsampleForwardT<_FP16>(in, out, upsampling_type, kernel_size);
    return;
  }
#endif
  upsampleForwardT<float>(in, out, upsampling_type, kernel_size);
}

void Upsample2dLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  const nntrainer::Tensor &derivative_ =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);

  nntrainer::Tensor &dx = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  const auto &kernel_size =
    std::get<std::array<props::KernelSize, UPSAMPLE2D_DIM>>(upsample2d_props);
  const auto &upsampling_type =
    std::get<props::UpsampleMode>(upsample2d_props).get();

  switch (upsampling_type) {
  case props::UpsampleModeInfo::Interpolation::nearest: {
    float val = 0;
    for (int b = 0; b < (int)derivative_.batch(); b++) {
      for (int c = 0; c < (int)derivative_.channel(); c++) {
        for (int h = 0; h < (int)derivative_.height(); h++) {
          for (int w = 0; w < (int)derivative_.width(); w++) {
            if (h % kernel_size[0] == 0 && w % kernel_size[1] == 0) {
              dx.setValue(b, c, h / kernel_size[0], w / kernel_size[1], 0);
            }

            val = dx.getValue(b, c, h / kernel_size[0], w / kernel_size[1]) +
                  derivative_.getValue(b, c, h, w);
            dx.setValue(b, c, h / kernel_size[0], w / kernel_size[1], val);
          }
        }
      }
    }
  } break;
  case props::UpsampleModeInfo::Interpolation::bilinear: {
    dx.setZero();

    int input_height = dx.height();
    int input_width = dx.width();

    for (int b = 0; b < (int)derivative_.batch(); b++) {
      for (int c = 0; c < (int)derivative_.channel(); c++) {
        for (int h = 0; h < (int)derivative_.height(); h++) {
          for (int w = 0; w < (int)derivative_.width(); w++) {
            float in_h = (h + 0.5f) / kernel_size[0] - 0.5f;
            float in_w = (w + 0.5f) / kernel_size[1] - 0.5f;

            if (in_h < 0) {
              in_h = 0.0f;
            }
            if (in_w < 0) {
              in_w = 0.0f;
            }

            int y0 = static_cast<int>(floor(in_h));
            int x0 = static_cast<int>(floor(in_w));
            int y1 = std::min(y0 + 1, input_height - 1);
            int x1 = std::min(x0 + 1, input_width - 1);

            float dx_ = (in_w - x0); // Due to name conflict with dx
            float dy_ = (in_h - y0);

            float top_left_weight = (1.0 - dy_) * (1.0 - dx_);
            float top_right_weight = (1.0 - dy_) * dx_;
            float bottom_left_weight = dy_ * (1.0 - dx_);
            float bottom_right_weight = dy_ * dx_;

            float grad = derivative_.getValue(b, c, h, w);

            dx.addValue(b, c, y0, x0, top_left_weight * grad, 1.0f);
            dx.addValue(b, c, y0, x1, top_right_weight * grad, 1.0f);
            dx.addValue(b, c, y1, x0, bottom_left_weight * grad, 1.0f);
            dx.addValue(b, c, y1, x1, bottom_right_weight * grad, 1.0f);
          }
        }
      }
    }
  } break;
  default:
    throw std::runtime_error("Error: Unknown Upsample Mode Type");
  }
}

void Upsample2dLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, upsample2d_props);

  if (!remain_props.empty()) {
    std::string msg = "[Upsample2dLayer] Unknown properties set with count" +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}
} // namespace nntrainer
