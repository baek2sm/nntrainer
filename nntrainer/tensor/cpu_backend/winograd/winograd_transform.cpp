// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Samsung Electronics Co., Ltd.
 *
 * @file   winograd_transform.cpp
 * @date   23 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Winograd F(2x2,3x3) FP32 conv transform implementation
 *
 * Lavin & Winograd 2015 minimal algorithm. Multiplications reduced 2.25x.
 * Uses Tensor::dot (sgemm) for the 16 transform-point GEMMs. All data lives
 * inside Tensor objects so the memory planner / pool lifecycle is respected.
 */
#include <winograd_transform.h>

#include <cstring>
#include <vector>

namespace nntrainer {

// F(2,2,3,3) constant matrices (Lavin & Winograd 2015).
static constexpr float Bt[4][4] = {
  {1, 0, -1, 0}, {0, 1, 1, 0}, {0, -1, 1, 0}, {0, 1, 0, -1}};
static constexpr float G[4][3] = {
  {1, 0, 0}, {0.5f, 0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0, 0, 1}};
static constexpr float At[2][4] = {{1, 1, 1, 0}, {0, 1, -1, -1}};

Tensor winograd_transform_weight_f23x3(const Tensor &filter, unsigned int Cout,
                                       unsigned int Cin) {
  // U: {16, Cout, Cin} row-major. Store as Tensor {1, 16, Cout, Cin}.
  TensorDim udim(1, 16, Cout, Cin);
  Tensor U(udim);
  U.setZero();
  float *ud = U.getData<float>();

  const float *fd = filter.getData<float>();
  for (unsigned int co = 0; co < Cout; ++co) {
    for (unsigned int ci = 0; ci < Cin; ++ci) {
      const float *g = fd + ((size_t)co * Cin + ci) * 9;
      float Gg[4][3];
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 3; ++j) {
          float s = 0;
          for (int k = 0; k < 3; ++k)
            s += G[i][k] * g[k * 3 + j];
          Gg[i][j] = s;
        }
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
          float s = 0;
          for (int k = 0; k < 3; ++k)
            s += Gg[i][k] * G[j][k];
          ud[(i * 4 + j) * (size_t)Cout * Cin + (size_t)co * Cin + ci] = s;
        }
    }
  }
  return U;
}

void winograd_conv2d_f23x3_fp32(const Tensor &in, const Tensor &U, Tensor &out,
                                unsigned int padH, unsigned int padW) {
  const TensorDim &in_dim = in.getDim();
  const unsigned int Cin = in_dim.channel();
  const unsigned int H = in_dim.height(), W = in_dim.width();
  const unsigned int Cout = out.getDim().channel();
  const unsigned int OH = H + 2 * padH - 2;
  const unsigned int OW = W + 2 * padW - 2;
  const unsigned int nTH = (OH + 1) / 2;
  const unsigned int nTW = (OW + 1) / 2;
  const unsigned int T = nTH * nTW;

  const float *id = in.getData<float>();
  const float *ud = U.getData<float>();

  // Step 2: V = B^T d B.  Pack as {1, 16, Cin, T} for dot per point.
  TensorDim vdim(1, 16, Cin, T);
  Tensor V(vdim);
  V.setZero();
  float *vd = V.getData<float>();
  for (unsigned int t = 0; t < T; ++t) {
    const unsigned int th = t / nTW, tw = t % nTW;
    const unsigned int oh0 = th * 2, ow0 = tw * 2;
    for (unsigned int ci = 0; ci < Cin; ++ci) {
      float d[4][4];
      for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
          int ih = (int)oh0 + r - (int)padH;
          int iw = (int)ow0 + c - (int)padW;
          d[r][c] = (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W)
                      ? id[ci * H * W + ih * W + iw]
                      : 0.0f;
        }
      float Bd[4][4];
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
          float s = 0;
          for (int k = 0; k < 4; ++k)
            s += Bt[i][k] * d[k][j];
          Bd[i][j] = s;
        }
      float Vch[4][4];
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
          float s = 0;
          for (int k = 0; k < 4; ++k)
            s += Bd[i][k] * Bt[j][k];
          Vch[i][j] = s;
        }
      for (int p = 0; p < 16; ++p)
        vd[p * (size_t)Cin * T + ci * T + t] = Vch[p / 4][p % 4];
    }
  }

  // Step 3: M[p] = U[p](Cout,Cin) × V[p](Cin,T) -> (Cout, T) via Tensor::dot.
  // For each point, slice U and V along dim[1]=point, dot → M slice.
  TensorDim mdim(1, 16, Cout, T);
  Tensor M(mdim);
  M.setZero();
  for (int p = 0; p < 16; ++p) {
    // U slice: {1,1,Cout,Cin} from {1,16,Cout,Cin} at channel offset p.
    Tensor U_p =
      U.getSharedDataTensor(TensorDim(1, 1, Cout, Cin), p * Cout * Cin);
    Tensor V_p = V.getSharedDataTensor(TensorDim(1, 1, Cin, T), p * Cin * T);
    Tensor M_p = M.getSharedDataTensor(TensorDim(1, 1, Cout, T), p * Cout * T);
    // M = U × V^T? No: M(Cout,T) = U(Cout,Cin) × V(Cin,T).
    // Tensor::dot(input, out, trans, trans_in): out = this × input.
    // U_p.dot(V_p, M_p, false, false) → M = U × V. But dot interprets
    // dims as (first_three, last). U{1,1,Cout,Cin}: first3=Cout,last=Cin.
    // V{1,1,Cin,T}: first3=Cin,last=T. !trans&&!trans_in: check Cin==Cin ✓.
    // M=Cout × N=T. Correct.
    U_p.dot(V_p, M_p, false, false);
  }

  // Step 4: Y = A^T M A per (co, tile) -> 2x2. Write directly into out Tensor.
  float *od = out.getData<float>();
  const float *md = M.getData<float>();
  for (unsigned int t = 0; t < T; ++t) {
    for (unsigned int co = 0; co < Cout; ++co) {
      float Mtile[4][4];
      for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
          Mtile[r][c] = md[(r * 4 + c) * (size_t)Cout * T + (size_t)co * T + t];
      float AtM[2][4];
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j) {
          float s = 0;
          for (int k = 0; k < 4; ++k)
            s += At[i][k] * Mtile[k][j];
          AtM[i][j] = s;
        }
      const unsigned int oh0 = (t / nTW) * 2, ow0 = (t % nTW) * 2;
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
          float s = 0;
          for (int k = 0; k < 4; ++k)
            s += AtM[i][k] * At[j][k];
          unsigned int oh = oh0 + i, ow = ow0 + j;
          if (oh < OH && ow < OW)
            od[co * OH * OW + oh * OW + ow] = s;
        }
    }
  }
}

} // namespace nntrainer
