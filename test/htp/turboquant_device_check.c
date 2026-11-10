// SPDX-License-Identifier: Apache-2.0
/** @file turboquant_device_check.c
 * @brief DSP packed attention against the portable scalar cache, including
 * native M1 reuse, sliding ring wrap and decode timing. Compile together with
 * hvx_turboquant.c (without __hexagon__) and turboquant_codec.c on the host.
 * @author NNTrainer contributors
 * @bug Synthetic kernel/dispatch check; not a checkpoint quality evaluation.
 */
#include "htp_graph_desc.h"
#include "hvx_turboquant.h"
#include "nntr_hvx.h"
#include <AEEStdErr.h>
#include <math.h>
#include <remote.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define TQ_FLAG 0x04000000u
uint32_t hvx_worker_pool_workers(const hvx_worker_pool *p) {
  (void)p;
  return 3;
}
void hvx_worker_pool_run(hvx_worker_pool *p, hvx_worker_pool_func f, void *data,
                         uint32_t units) {
  (void)p;
  const uint32_t n = units < 4 ? units : 4;
  for (uint32_t i = 0; i < n; ++i)
    f(n, i, data);
}
static uint32_t rng = 71;
static float random_value(void) {
  rng = rng * 1664525u + 1013904223u;
  return ((float)(rng >> 8) / 16777216.0f - 0.5f) * 2;
}
static void noise(float *x, size_t n) {
  for (size_t i = 0; i < n; ++i)
    x[i] = random_value();
}
static double milliseconds(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}
static int check(int rc, const char *what) {
  if (rc)
    printf("FAIL %s rc=0x%x\n", what, (unsigned)rc);
  return rc != 0;
}
static int compare(const float *a, const float *b, size_t n, float tolerance,
                   const char *name) {
  float worst = 0;
  for (size_t i = 0; i < n; ++i) {
    if (!isfinite(a[i]) || !isfinite(b[i])) {
      printf("FAIL %s nonfinite index=%zu actual=%g expected=%g\n", name, i,
             a[i], b[i]);
      return 1;
    }
    float d = fabsf(a[i] - b[i]);
    if (d > worst)
      worst = d;
  }
  printf("COMPARE %s elements=%zu max_abs=%.8g %s\n", name, n, worst,
         worst <= tolerance ? "PASS" : "FAIL");
  return worst > tolerance;
}
static int shape_check(remote_handle64 session, uint32_t hd, uint32_t window,
                       uint32_t gqa, uint32_t count) {
  const uint32_t nk = 2 * hd, nq = 2 * gqa * hd;
  const uint32_t end = 4 * count, max_seq = end + 32 < 256 ? 256 : end + 32;
  int rc = 0, error = 0;
  uint32_t info[8] = {0};
  if (check(nntr_hvx_session_info(session, info, 8), "shape capabilities"))
    return 1;
  // HMX includes FP16 operands, logits and partial outputs; native HVX
  // decode is compared separately at the tighter 3e-4 tolerance.
  const float prefill_tolerance = info[7] ? 0.002f : 0.0003f;
  uint32_t stats[3];
  float *q = malloc((size_t)count * nq * 4),
        *k = malloc((size_t)count * nk * 4),
        *v = malloc((size_t)count * nk * 4);
  float *actual = malloc((size_t)count * nq * 4),
        *expected = malloc((size_t)count * nq * 4);
  hvx_tq_cache *reference = hvx_tq_create(1, 2, gqa, hd, max_seq, NULL, &error);
  if (!q || !k || !v || !actual || !expected || !reference) {
    rc = 1;
    goto done;
  }
  rc |=
    check(nntr_hvx_attn_m1_register(session, 1, 2, gqa, hd, max_seq | TQ_FLAG),
          "register");
  if (rc)
    goto done;
  for (uint32_t from = 0; from < end; from += count) {
    noise(q, (size_t)count * nq);
    noise(k, (size_t)count * nk);
    noise(v, (size_t)count * nk);
    rc |= check(hvx_tq_step(reference, 0, from, count, window,
                            1 / sqrtf((float)hd), q, k, v, expected),
                "scalar step");
    rc |= check(nntr_hvx_attn_tq_step(session, 0, from, count, window,
                                      1 / sqrtf((float)hd), q, count * nq, k,
                                      count * nk, v, count * nk, actual,
                                      count * nq, stats, 3),
                "DSP step");
    if (rc)
      break;
    rc |= check((int)stats[2], "packed HMX tile roundtrip");
    char name[64];
    snprintf(name, sizeof(name), "hd%u window%u from%u", hd, window, from);
    rc |=
      compare(actual, expected, (size_t)count * nq, prefill_tolerance, name);
    if (rc)
      break;
  }
  if (!rc && !window) {
    noise(q, nq);
    noise(k, nk);
    noise(v, nk);
    rc |= check(hvx_tq_step(reference, 0, end, 1, 0, 1 / sqrtf((float)hd), q, k,
                            v, expected),
                "scalar decode");
    rc |=
      check(nntr_hvx_attn_m1_forward(session, 0, end, 1 / sqrtf((float)hd), q,
                                     nq, k, nk, v, nk, actual, nq, NULL, 0),
            "native M1 decode reuses prefill");
    rc |= compare(actual, expected, nq, 3e-4f, "native M1 reuse");
  }
  if (window == 17 && !rc) {
    const uint32_t bytes = stats[1];
    // Packed pages retain the window and the current prefill chunk.
    const uint32_t bound =
      2u * ((window + count + 31u) / 32u) * (hd * 32u + 512u);
    printf("RING hd=%u bytes=%u bound=%u %s\n", hd, bytes, bound,
           bytes <= bound ? "PASS" : "FAIL");
    rc |= bytes > bound;
  }
  rc |= check(nntr_hvx_attn_m1_release(session), "release");
done:
  hvx_tq_free(reference);
  free(q);
  free(k);
  free(v);
  free(actual);
  free(expected);
  return rc;
}
static int benchmark(remote_handle64 session, uint32_t rows) {
  const uint32_t hd = 128, nk = 256, nq = 512, chunk = 512;
  int rc = 0, error = 0;
  float *k = malloc((size_t)rows * nk * 4), *v = malloc((size_t)rows * nk * 4);
  float q[512], key[256], value[256], actual[512], expected[512];
  hvx_tq_cache *reference = hvx_tq_create(1, 2, 2, hd, rows + 64, NULL, &error);
  if (!k || !v || !reference) {
    rc = 1;
    goto done;
  }
  noise(k, (size_t)rows * nk);
  noise(v, (size_t)rows * nk);
  noise(q, nq);
  noise(key, nk);
  noise(value, nk);
  rc |= check(
    nntr_hvx_attn_m1_register(session, 1, 2, 2, hd, (rows + 64) | TQ_FLAG),
    "benchmark register");
  for (uint32_t from = 0; !rc && from < rows; from += chunk) {
    uint32_t count = rows - from < chunk ? rows - from : chunk;
    rc |= check(hvx_tq_append(reference, 0, from, count, 0,
                              k + (size_t)from * nk, v + (size_t)from * nk),
                "reference bulk append");
    rc |= check(nntr_hvx_attn_m1_kv_append(session, 0, from, count,
                                           k + (size_t)from * nk, count * nk,
                                           v + (size_t)from * nk, count * nk),
                "DSP bulk append");
  }
  if (rc)
    goto release;
  rc |= check(hvx_tq_step(reference, 0, rows, 1, 0, 1 / sqrtf((float)hd), q,
                          key, value, expected),
              "reference benchmark decode");
  rc |=
    check(nntr_hvx_attn_m1_forward(session, 0, rows, 1 / sqrtf((float)hd), q,
                                   nq, key, nk, value, nk, actual, nq, NULL, 0),
          "benchmark decode");
  rc |= compare(actual, expected, nq, 3e-4f, "long-context decode");
  if (!rc) {
    double start = milliseconds();
    for (int i = 0; i < 10; ++i)
      rc |= check(nntr_hvx_attn_m1_forward(session, 0, rows,
                                           1 / sqrtf((float)hd), q, nq, key, nk,
                                           value, nk, actual, nq, NULL, 0),
                  "timed decode");
    double wall = (milliseconds() - start) / 10;
    uint32_t stats[2];
    rc |= check(nntr_hvx_attn_tq_step(session, 0, rows, 1, 0,
                                      1 / sqrtf((float)hd), q, nq, key, nk,
                                      value, nk, actual, nq, stats, 2),
                "timed DSP step");
    printf("BENCH context=%u wall_ms=%.6f dsp_us=%u packed_bytes=%u "
           "reference_bytes=%zu\n",
           rows + 1, wall, stats[0], stats[1], hvx_tq_payload_bytes(reference));
  }
release:
  rc |= check(nntr_hvx_attn_m1_release(session), "benchmark release");
done:
  hvx_tq_free(reference);
  free(k);
  free(v);
  return rc;
}
static int graph_check(remote_handle64 session) {
  const uint32_t hd = 64, nq = 256, nk = 128, width = 512, seq = 256;
  uint32_t words[2048] = {0}, count = 0, resume = 0;
  int rc = 0, error = 0;
  words[0] = HTP_GRAPH_MAGIC;
  words[1] = HTP_GRAPH_VERSION;
  words[2] = 1;
  words[3] = 4;
  words[4] = nq;
  words[5] = 32;
  words[6] = seq;
  words[HTP_GRAPH_HEADER_WORDS] = HTP_GRAPH_LAYER_ATTN;
  float epsilon = 1e-6f, scale = 0.125f;
  for (uint32_t i = 0; i < 4; ++i) {
    htp_graph_op *op = htp_graph_op_at(words, i);
    op->kind = i == 0   ? HTP_OP_QK_NORM
               : i == 1 ? HTP_OP_ROPE
               : i == 2 ? HTP_OP_ATTN_M1
                        : HTP_OP_LM_HEAD;
    op->layer = i == 3 ? 1 : 0;
    op->resident = i < 3;
    op->K = i < 3 ? width : nq;
    op->N = i == 2 ? nq : i == 3 ? 32 : width;
    op->in_slot = i < 3 ? 2 : 1;
    op->out_slot = i < 2 ? 2 : 1;
    op->next_mm = HTP_GRAPH_NO_OP;
    if (i < 3) {
      op->head_dim = hd;
      op->n_kv = 2;
      op->gqa = 2;
    }
    if (!i)
      memcpy(&op->eps_bits, &epsilon, 4);
    if (i == 2)
      memcpy(&op->eps_bits, &scale, 4);
  }
  rc |= check(
    nntr_hvx_graph_init(session, words, htp_graph_words_for(1, 4), &count),
    "native graph init");
  if (rc)
    return rc;
  float gamma[128], rope[256 * 64];
  for (uint32_t i = 0; i < 128; ++i)
    gamma[i] = 1;
  for (uint32_t r = 0; r < seq; ++r)
    for (uint32_t d = 0; d < hd; ++d)
      rope[r * hd + d] = d < hd / 2 ? 1 : 0;
  rc |= check(
    nntr_hvx_graph_set_param(session, 0, HTP_GRAPH_PARAM_GAMMA, gamma, 128),
    "graph QK gamma");
  rc |= check(nntr_hvx_graph_set_param(session, 1, HTP_GRAPH_PARAM_ROPE_TABLE,
                                       rope, 256 * 64),
              "graph identity RoPE");
  rc |= check(nntr_hvx_attn_m1_register(session, 1, 2, 2, hd, seq | TQ_FLAG),
              "graph packed cache");
  float k[33 * 128], v[33 * 128], input[512], normalized[512], actual[256],
    expected[256];
  noise(k, 33 * nk);
  noise(v, 33 * nk);
  noise(input, width);
  memcpy(normalized, input, sizeof(input));
  for (uint32_t head = 0; head < 6; ++head) {
    double squared = 0;
    for (uint32_t d = 0; d < hd; ++d)
      squared += (double)input[head * hd + d] * input[head * hd + d];
    float inv = 1 / sqrtf((float)(squared / hd) + epsilon);
    for (uint32_t d = 0; d < hd; ++d)
      normalized[head * hd + d] *= inv;
  }
  hvx_tq_cache *reference = hvx_tq_create(1, 2, 2, hd, seq, NULL, &error);
  if (!reference) {
    rc = 1;
    goto done;
  }
  rc |=
    check(hvx_tq_append(reference, 0, 0, 33, 0, k, v), "graph reference seed");
  rc |=
    check(nntr_hvx_attn_m1_kv_append(session, 0, 0, 33, k, 33 * nk, v, 33 * nk),
          "graph DSP seed");
  rc |= check(hvx_tq_step(reference, 0, 33, 1, 0, scale, normalized,
                          normalized + nq, normalized + nq + nk, expected),
              "graph scalar attention");
  if (!rc)
    rc |= check(nntr_hvx_forward(session, 0, 33, NULL, 0, NULL, 0, NULL, 0,
                                 input, width, actual, nq, &resume),
                "resident QK/RoPE/packed attention stretch");
  if (!rc) {
    rc |= resume != 3;
    rc |= compare(actual, expected, nq, 3e-4f, "resident graph packed KV");
  }
  hvx_tq_free(reference);
done:
  rc |= check(nntr_hvx_attn_m1_release(session), "graph cache release");
  rc |= check(nntr_hvx_graph_release(session), "graph release");
  return rc;
}

static int baseline_q8(remote_handle64 session, uint32_t heads, uint32_t gqa,
                       uint32_t hd, uint32_t rows, uint32_t window,
                       const float *q, const float *k, const float *v,
                       float *out) {
  const uint32_t nk = heads * hd, nq = nk * gqa,
                 enc_count = heads + nk + 4 * heads * gqa;
  uint16_t *k16 = malloc((size_t)rows * nk * 2),
           *v16 = malloc((size_t)rows * nk * 2),
           *out16 = malloc((size_t)rows * nq * 2);
  float *enc = malloc(enc_count * 4u);
  uint32_t handle = 0, stats[12] = {0};
  int rc = 0, registered = 0;
  if (!k16 || !v16 || !out16 || !enc) {
    rc = 1;
    goto done;
  }
  for (size_t i = 0; i < (size_t)rows * nk; ++i) {
    _Float16 kh = k[i], vh = v[i];
    memcpy(k16 + i, &kh, 2);
    memcpy(v16 + i, &vh, 2);
  }
  rc = check(nntr_hvx_kv_register_q(session, 0, rows, heads, hd, &handle),
             "baseline Q8 register");
  if (rc)
    goto done;
  registered = 1;
  rc = check(nntr_hvx_attn_q2_step_fqc(session, handle, 0, k16, rows * nk, v16,
                                       rows * nk, rows, 0, rows, heads * gqa,
                                       window, q, rows * nq, enc, enc_count,
                                       out16, rows * nq, stats, 12),
             "baseline calibrated Q8 prefill");
  if (!rc)
    for (size_t i = 0; i < (size_t)rows * nq; ++i) {
      const uint32_t h = (i / hd) % (heads * gqa);
      const float *e = enc + heads + nk + 2 * heads * gqa + 2 * h;
      out[i] = e[0] * ((int32_t)out16[i] - (int32_t)e[1]);
    }
done:
  if (registered)
    rc |= check(nntr_hvx_kv_release_q(session, handle), "baseline Q8 release");
  free(k16);
  free(v16);
  free(out16);
  free(enc);
  return rc;
}

// Independent scalar fixed-Q8 calibration and packing. The DSP retains
// these original Q8 values, so the reference must not quantize the prefix
// to TQ or compare against the former all-TQ cache.
static int reference_q8_prefix(hvx_tq_cache *cache, uint32_t from,
                               uint32_t rows, uint32_t heads, uint32_t hd,
                               uint32_t window, const float *k,
                               const float *v) {
  const uint32_t stride = (rows + 63u) / 32u * 32u;
  const size_t n = (size_t)heads * stride * hd;
  uint8_t *k8 = malloc(n), *v8 = malloc(n);
  float *sk = calloc(heads, 4), *sv = calloc((size_t)heads * hd, 4);
  if (!k8 || !v8 || !sk || !sv) {
    free(k8);
    free(v8);
    free(sk);
    free(sv);
    return 1;
  }
  memset(k8, 128, n);
  memset(v8, 128, n);
  for (uint32_t h = 0; h < heads; ++h) {
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t d = 0; d < hd; ++d) {
        const size_t at = ((size_t)r * heads + h) * hd + d;
        sk[h] = fmaxf(sk[h], fabsf((float)(_Float16)k[at]));
        sv[h * hd + d] = fmaxf(sv[h * hd + d], fabsf((float)(_Float16)v[at]));
      }
    sk[h] = sk[h] > 0 ? sk[h] / 127.0f : 1;
    for (uint32_t d = 0; d < hd; ++d)
      sv[h * hd + d] = sv[h * hd + d] > 0 ? sv[h * hd + d] / 127.0f : 1;
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t d = 0; d < hd; ++d) {
        const size_t at = ((size_t)r * heads + h) * hd + d;
        const size_t dst = ((size_t)h * stride + r) * hd + d;
        k8[dst] =
          (int)fmaxf(-127,
                     fminf(127, rintf((float)(_Float16)k[at] * (1 / sk[h])))) +
          128;
        v8[dst] = (int)fmaxf(-127, fminf(127, rintf((float)(_Float16)v[at] *
                                                    (1 / sv[h * hd + d])))) +
                  128;
      }
  }
  int rc = hvx_tq_adopt_q8_prefix(cache, 0, from, rows, stride, window, k8, v8,
                                  sk, sv);
  if (rc) {
    free(k8);
    free(v8);
    free(sk);
    free(sv);
  }
  return rc;
}

// Exercise the long-global byte-dot dispatch against the scalar Q8/TQ
// reference, independently of model output and MoE expert selection.
static int long_hybrid_check(remote_handle64 session, uint32_t hd,
                             uint32_t heads, uint32_t gqa) {
  const uint32_t nk = heads * hd;
  const float scale = 1 / sqrtf((float)hd);
  const uint32_t nq = nk * gqa, prefix = 32, rows = 8192, chunk = 512;
  int error = 0, rc = 0;
  hvx_tq_cache *reference =
    hvx_tq_create(1, heads, gqa, hd, rows + 64, NULL, &error);
  float *k = malloc((size_t)chunk * nk * sizeof(float));
  float *v = malloc((size_t)chunk * nk * sizeof(float));
  float *q = malloc((size_t)prefix * nq * sizeof(float));
  float *actual = malloc((size_t)prefix * nq * sizeof(float));
  float *expected = malloc((size_t)nq * sizeof(float));
  if (!reference || !k || !v || !q || !actual || !expected) {
    rc = 1;
    goto done;
  }
  noise(k, (size_t)prefix * nk);
  noise(v, (size_t)prefix * nk);
  noise(q, (size_t)prefix * nq);
  rc |= check(nntr_hvx_attn_m1_register(session, 1, heads, gqa, hd,
                                        (rows + 64) | TQ_FLAG),
              "long hybrid register");
  if (rc)
    goto done;
  uint32_t stats[12] = {0};
  rc |= check(reference_q8_prefix(reference, 0, prefix, heads, hd, 0, k, v),
              "long hybrid seed");
  rc |=
    check(nntr_hvx_attn_tq_step(session, HTP_ATTN_TQ_PREFILL_Q8, 0, prefix, 0,
                                scale, q, prefix * nq, k, prefix * nk, v,
                                prefix * nk, actual, prefix * nq, stats, 12),
          "long hybrid prefill");
  for (uint32_t pos = prefix; pos < rows && !rc; pos += chunk) {
    const uint32_t count = rows - pos < chunk ? rows - pos : chunk;
    noise(k, (size_t)count * nk);
    noise(v, (size_t)count * nk);
    rc |= check(hvx_tq_append(reference, 0, pos, count, 0, k, v),
                "long hybrid reference append");
    rc |= check(nntr_hvx_attn_m1_kv_append(session, 0, pos, count, k,
                                           count * nk, v, count * nk),
                "long hybrid DSP append");
  }
  for (uint32_t i = 0; i < 8 && !rc; ++i) {
    noise(q, nq);
    noise(k, nk);
    noise(v, nk);
    rc |=
      check(hvx_tq_step(reference, 0, rows + i, 1, 0, scale, q, k, v, expected),
            "long hybrid reference decode");
    rc |= check(nntr_hvx_attn_tq_step(session, 0, rows + i, 1, 0, scale, q, nq,
                                      k, nk, v, nk, actual, nq, stats, 12),
                "long hybrid DSP decode");
    // Same hybrid tolerance as the Q12 path; no relaxation for byte dots.
    rc |= compare(actual, expected, nq, 0.002f, "long hybrid byte dots");
  }
  rc |= check(nntr_hvx_attn_m1_release(session), "long hybrid release");
done:
  hvx_tq_free(reference);
  free(k);
  free(v);
  free(q);
  free(actual);
  free(expected);
  return rc;
}

static int fresh_prefill_check(remote_handle64 session, uint32_t hd,
                               uint32_t heads, uint32_t gqa, uint32_t window,
                               int shared_v) {
  const uint32_t rows = 33, nk = heads * hd, nq = nk * gqa;
  float *q = malloc((size_t)rows * nq * 4), *k = malloc((size_t)rows * nk * 4),
        *v = malloc((size_t)rows * nk * 4);
  float *actual = malloc((size_t)rows * nq * 4),
        *expected = malloc((size_t)rows * nq * 4);
  int rc = 0, error = 0;
  hvx_tq_cache *reference = hvx_tq_create(1, heads, gqa, hd, 256, NULL, &error);
  if (!q || !k || !v || !actual || !expected || !reference) {
    rc = 1;
    goto done;
  }
  noise(q, (size_t)rows * nq);
  noise(k, (size_t)rows * nk);
  noise(v, (size_t)rows * nk);
  if (shared_v)
    memcpy(v, k, (size_t)rows * nk * sizeof(float));
  // FP16-rounded operands make the independent FP64 attention reference
  // match the existing Q8 prefill kernel's conversion contract.
  for (size_t i = 0; i < (size_t)rows * nk; ++i) {
    k[i] = (float)(_Float16)k[i];
    v[i] = (float)(_Float16)v[i];
  }
  rc |=
    check(nntr_hvx_attn_m1_register(session, 1, heads, gqa, hd, 256 | TQ_FLAG),
          "fresh register");
  if (rc)
    goto done;
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t h = 0; h < heads * gqa; ++h) {
      double weights[33], maximum = -INFINITY, den = 0;
      for (uint32_t t = window && row + 1 > window ? row + 1 - window : 0;
           t <= row; ++t) {
        double dot = 0;
        for (uint32_t d = 0; d < hd; ++d)
          dot += (double)q[(size_t)row * nq + h * hd + d] *
                 k[(size_t)t * nk + (h / gqa) * hd + d];
        weights[t] = dot / sqrt((double)hd);
        if (weights[t] > maximum)
          maximum = weights[t];
      }
      for (uint32_t t = window && row + 1 > window ? row + 1 - window : 0;
           t <= row; ++t) {
        weights[t] = exp(weights[t] - maximum);
        den += weights[t];
      }
      for (uint32_t d = 0; d < hd; ++d) {
        double sum = 0;
        for (uint32_t t = window && row + 1 > window ? row + 1 - window : 0;
             t <= row; ++t)
          sum += weights[t] * v[(size_t)t * nk + (h / gqa) * hd + d];
        expected[(size_t)row * nq + h * hd + d] = sum / den;
      }
    }
  float *baseline = malloc((size_t)rows * nq * 4);
  float *joined = malloc(((size_t)rows * nq + 2u * rows * nk) * 4u);
  if (!baseline || !joined) {
    rc = 1;
    goto release;
  }
  uint32_t active_rows = rows;
  uint32_t heap_after_first = 0;
  for (uint32_t run = 0; run < 12 && !rc; ++run) {
    active_rows = rows - (run % 4 == 2 ? 2u : run % 4 == 3 ? 1u : 0u);
    rc |= baseline_q8(session, heads, gqa, hd, active_rows, window, q, k, v,
                      baseline);
    const int use_joined = shared_v || run % 3;
    const size_t qn = (size_t)active_rows * nq, kn = (size_t)active_rows * nk;
    if (use_joined) {
      memcpy(joined, q, qn * 4);
      memcpy(joined + qn, k, kn * 4);
      if (!shared_v)
        memcpy(joined + qn + kn, v, kn * 4);
    }
    uint32_t stats[140] = {0}, info[8] = {0};
    rc |= check(
      nntr_hvx_attn_tq_step(
        session,
        HTP_ATTN_TQ_PREFILL_Q8 | (run % 2 ? HTP_ATTN_TQ_OUT_U16 : 0u) |
          (use_joined ? HTP_ATTN_TQ_JOINED_QKV : 0u) |
          (shared_v ? HTP_ATTN_TQ_SHARED_V : 0u),
        0, active_rows, window, 1 / sqrtf((float)hd), use_joined ? joined : q,
        use_joined ? qn + (shared_v ? kn : 2u * kn) : qn, use_joined ? NULL : k,
        use_joined ? 0 : kn, use_joined ? NULL : v, use_joined ? 0 : kn, actual,
        active_rows * nq / (run % 2 ? 2u : 1u), stats,
        run % 2 ? 12 + 2 * heads * gqa : 11),
      "fresh Q8 HMX with retained Q8 prefix");
    rc |= stats[9] != 1 || stats[10] > 96u * 1024u * 1024u;
    const uint32_t prefix_stride = (rows + 63u) / 32u * 32u;
    const size_t prefix_bytes =
      (size_t)heads * (2u * prefix_stride * hd + 4u * (1u + hd));
    if (stats[1] != prefix_bytes) {
      printf("FAIL Q8-only prefill payload got=%u expected=%zu\n", stats[1],
             prefix_bytes);
      rc = 1;
    }
    if (!rc && run % 2) {
      rc |= stats[11] != 1;
      for (size_t i = (size_t)active_rows * nq; i-- > 0;) {
        const uint32_t h = (i / hd) % (heads * gqa);
        float scale, zp;
        memcpy(&scale, stats + 12 + 2 * h, 4);
        memcpy(&zp, stats + 13 + 2 * h, 4);
        actual[i] = scale * ((int32_t)((uint16_t *)actual)[i] - (int32_t)zp);
      }
    }
    if (!rc)
      rc |= compare(actual, expected, (size_t)active_rows * nq, 0.008f,
                    "fresh Q8 vs FP64 attention");
    if (!rc)
      rc |= compare(actual, baseline, (size_t)active_rows * nq, 1e-6f,
                    "fresh Q8 matches baseline Q8");
    rc |= check(nntr_hvx_session_info(session, info, 8), "fresh heap check");
    if (!run)
      heap_after_first = info[4];
    else if (info[4] > heap_after_first + 64u) {
      printf("FAIL fresh workspace retained heap=%u first=%u KiB\n", info[4],
             heap_after_first);
      rc = 1;
    }
    if (run + 1 < 12) {
      const float factor = shared_v ? -1.0f : (run % 2 ? -2.0f : -0.5f);
      for (size_t i = 0; i < (size_t)rows * nq; ++i) {
        q[i] = -q[i];
        expected[i] *= factor;
      }
      for (size_t i = 0; i < (size_t)rows * nk; ++i) {
        k[i] = -k[i];
        v[i] *= factor;
      }
    }
  }
  if (!rc) {
    // A compact output buffer must never receive a full-FP32 fallback.
    float *guard = malloc((size_t)rows * nq * 4);
    uint32_t stats[140] = {0};
    if (!guard)
      rc = 1;
    else {
      memset(guard, 0x5a, (size_t)rows * nq * 4);
      int declined = nntr_hvx_attn_tq_step(
        session, HTP_ATTN_TQ_OUT_U16, 0, 16, window, 1 / sqrtf((float)hd), q,
        16 * nq, k, 16 * nk, v, 16 * nk, guard, 8 * nq, stats,
        12 + 2 * heads * gqa);
      if (declined != AEE_EUNSUPPORTED && (uint32_t)declined != 0x80000414u)
        printf("FAIL compact fallback status=%d expected=%d\n", declined,
               AEE_EUNSUPPORTED);
      rc |= declined != AEE_EUNSUPPORTED && (uint32_t)declined != 0x80000414u;
      for (size_t i = (size_t)8u * nq * sizeof(float);
           i < (size_t)rows * nq * 4; ++i)
        if (((uint8_t *)guard)[i] != 0x5a) {
          printf("FAIL compact output guard offset=%zu\n", i);
          rc = 1;
          break;
        }
      free(guard);
    }
  }
  if (!rc) {
    rc |= check(
      reference_q8_prefix(reference, 0, active_rows, heads, hd, window, k, v),
      "fresh reference packed seed");
    noise(q, nq);
    noise(k, nk);
    noise(v, nk);
    rc |= check(hvx_tq_step(reference, 0, active_rows, 1, window,
                            1 / sqrtf((float)hd), q, k, v, expected),
                "single-prefix inline reference");
    uint32_t first_stats[12] = {0};
    rc |= check(nntr_hvx_attn_tq_step(session, 0, active_rows, 1, window,
                                      1 / sqrtf((float)hd), q, nq, k, nk, v, nk,
                                      actual, nq, first_stats, 12),
                "single-prefix inline decode");
    if (!rc)
      rc |= compare(actual, expected, nq, 0.002f,
                    "single-prefix Q8 and TQ inline head");
    ++active_rows;
    const uint32_t appended = 17, next = active_rows + appended;
    noise(q, (size_t)appended * nq);
    noise(k, (size_t)appended * nk);
    noise(v, (size_t)appended * nk);
    uint32_t stats[12] = {0};
    rc |= check(reference_q8_prefix(reference, active_rows, appended, heads, hd,
                                    window, k, v),
                "reference Q8 continuation");
    rc |= check(hvx_tq_step(reference, 0, active_rows, appended, window,
                            1 / sqrtf((float)hd), q, NULL, NULL, expected),
                "fresh reference appended prefill");
    rc |= check(nntr_hvx_attn_tq_step(session, HTP_ATTN_TQ_PREFILL_Q8,
                                      active_rows, appended, window,
                                      1 / sqrtf((float)hd), q, appended * nq, k,
                                      appended * nk, v, appended * nk, actual,
                                      appended * nq, stats, 12),
                "continuation retains Q8 history");
    rc |= stats[9] != 1;
    if (!rc)
      rc |= compare(actual, expected, (size_t)appended * nq, 0.002f,
                    "Q8 prefix with Q8 continuation");
    noise(q, nq);
    noise(k, nk);
    noise(v, nk);
    rc |= check(hvx_tq_step(reference, 0, next, 1, window, 1 / sqrtf((float)hd),
                            q, k, v, expected),
                "fresh reference decode");
    if (!window)
      rc |= check(nntr_hvx_attn_m1_forward(session, 0, next,
                                           1 / sqrtf((float)hd), q, nq, k, nk,
                                           v, nk, actual, nq, NULL, 0),
                  "fresh native packed decode");
    else
      rc |= check(nntr_hvx_attn_tq_step(session, 0, next, 1, window,
                                        1 / sqrtf((float)hd), q, nq, k, nk, v,
                                        nk, actual, nq, stats, 2),
                  "fresh sliding packed decode");
    if (!rc)
      rc |= compare(actual, expected, nq, 0.002f, "Q8 prefix with TQ decode");
  }
  // Cross several TQ page boundaries and wrap the sliding cache after
  // the immutable Q8 prefix has fallen out of the visible window.
  for (uint32_t pos = active_rows + 18u; pos < active_rows + 90u && !rc;
       ++pos) {
    if (pos == active_rows + 40u) {
      noise(q, (size_t)3 * nq);
      noise(k, (size_t)3 * nk);
      noise(v, (size_t)3 * nk);
      uint32_t chunk_stats[12] = {0};
      rc |=
        check(reference_q8_prefix(reference, pos, 3, heads, hd, window, k, v),
              "Q8 prompt after decode reference");
      rc |= check(hvx_tq_step(reference, 0, pos, 3, window,
                              1 / sqrtf((float)hd), q, NULL, NULL, expected),
                  "reference new prompt attention");
      rc |= check(nntr_hvx_attn_tq_step(session, HTP_ATTN_TQ_PREFILL_Q8, pos, 3,
                                        window, 1 / sqrtf((float)hd), q, 3 * nq,
                                        k, 3 * nk, v, 3 * nk, actual, 3 * nq,
                                        chunk_stats, 12),
                  "Q8 prompt after TQ decode");
      if (!rc) {
        rc |= compare(actual, expected, (size_t)3 * nq, 0.002f,
                      "interleaved Q8 prompts and TQ decode");
      }
      pos += 3;
    }
    noise(q, nq);
    noise(k, nk);
    noise(v, nk);
    uint32_t stats[12] = {0};
    rc |= check(hvx_tq_step(reference, 0, pos, 1, window, 1 / sqrtf((float)hd),
                            q, k, v, expected),
                "hybrid reference decode sequence");
    rc |= check(nntr_hvx_attn_tq_step(session, 0, pos, 1, window,
                                      1 / sqrtf((float)hd), q, nq, k, nk, v, nk,
                                      actual, nq, stats, 12),
                "hybrid DSP decode sequence");
    if (stats[10] != 0) {
      printf("FAIL decode retained prefill workspace=%u\n", stats[10]);
      rc = 1;
    }
    if (!rc)
      rc |= compare(actual, expected, nq, 0.002f,
                    "hybrid pages and window sequence");
  }
  free(baseline);
  free(joined);
release:
  rc |= check(nntr_hvx_attn_m1_release(session), "fresh release");
done:
  printf("FRESH_RESULT hd=%u window=%u shared=%d rc=%d\n", hd, window, shared_v,
         rc);
  hvx_tq_free(reference);
  free(q);
  free(k);
  free(v);
  free(actual);
  free(expected);
  return rc;
}

static int raw_fp16_check(remote_handle64 session) {
  int rc = 0;
  for (int run = 0; run < 3 && !rc; ++run) {
    float q[33 * 128], out[33 * 128], expected[33 * 128];
    uint16_t k[33 * 128], v[33 * 128];
    noise(q, 33 * 128);
    for (int i = 0; i < 33 * 128; ++i) {
      _Float16 kh = random_value(), vh = random_value();
      memcpy(k + i, &kh, 2);
      memcpy(v + i, &vh, 2);
    }
    uint32_t stats[32] = {0};
    rc |= check(nntr_hvx_attn_f16_prefill(
                  session, 33, 0, 33, 2, 2, 64, 0, 0, 0, 0, q, 33 * 128, k,
                  33 * 128, v, 33 * 128, NULL, 0, out, 33 * 128, stats, 32),
                "raw FP16 HMX prefill");
    for (int row = 0; row < 33; ++row)
      for (int h = 0; h < 2; ++h) {
        double logits[33], maximum = -INFINITY, den = 0;
        for (int t = 0; t <= row; ++t) {
          double dot = 0;
          for (int d = 0; d < 64; ++d) {
            _Float16 x;
            memcpy(&x, k + t * 128 + h * 64 + d, 2);
            dot += q[row * 128 + h * 64 + d] * (float)x;
          }
          logits[t] = dot / 8;
          if (logits[t] > maximum)
            maximum = logits[t];
        }
        for (int t = 0; t <= row; ++t) {
          logits[t] = exp(logits[t] - maximum);
          den += logits[t];
        }
        for (int d = 0; d < 64; ++d) {
          double sum = 0;
          for (int t = 0; t <= row; ++t) {
            _Float16 x;
            memcpy(&x, v + t * 128 + h * 64 + d, 2);
            sum += logits[t] * (float)x;
          }
          expected[row * 128 + h * 64 + d] = sum / den;
        }
      }
    if (!rc)
      rc |=
        compare(out, expected, 33 * 128, 0.002f, "raw FP16 HMX vs FP64 scalar");
  }
  return rc;
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  struct remote_rpc_control_unsigned_module u = {CDSP_DOMAIN_ID, 1};
  if (check(
        remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &u, sizeof(u)),
        "unsigned PD"))
    return 1;
  struct remote_rpc_thread_params t = {CDSP_DOMAIN_ID, -1, 128 * 1024};
  if (check(remote_session_control(FASTRPC_THREAD_PARAMS, &t, sizeof(t)),
            "stack"))
    return 1;
  remote_handle64 session;
  if (check(nntr_hvx_open(nntr_hvx_URI "&_dom=cdsp", &session), "open"))
    return 1;
  uint32_t info[8] = {0};
  int rc =
    check(nntr_hvx_session_info(session, info, 8), "session capabilities");
  printf("CAPS hmx_locked=%u vtcm_bytes=%u hvx_units=%u hmx_fp16_rate=%u\n",
         info[0], info[1], info[6], info[7]);
  if (argc > 1 && !strcmp(argv[1], "bench")) {
    rc |= benchmark(session, 1024);
    if (!rc)
      rc |= benchmark(session, 8192);
    if (!rc)
      rc |= benchmark(session, 32768);
  } else {
    const uint32_t groups[] = {1, 2, 3, 8};
    for (uint32_t hd = 64; hd <= 512 && !rc; hd *= 2)
      for (uint32_t gi = 0; gi < 4 && !rc; ++gi) {
        rc |= shape_check(session, hd, 0, groups[gi], 33);
        if (!rc)
          rc |= shape_check(session, hd, 17, groups[gi], 33);
      }
  }
  if (!rc && argc == 1)
    rc |= graph_check(session);
  if (!rc && argc == 1)
    rc |= shape_check(session, 128, 512, 2, 33);
  if (!rc && argc == 1)
    rc |= shape_check(session, 128, 0, 2, 128);
  if (!rc && argc == 1)
    rc |= shape_check(session, 128, 17, 2, 128);
  if (!rc && argc == 1 && info[0])
    rc |= fresh_prefill_check(session, 64, 2, 3, 0, 0);
  if (!rc && argc == 1 && info[0])
    rc |= fresh_prefill_check(session, 128, 2, 2, 17, 0);
  if (!rc && argc == 1 && info[0])
    rc |= fresh_prefill_check(session, 256, 8, 2, 0, 0);
  if (!rc && argc == 1 && info[0])
    rc |= fresh_prefill_check(session, 512, 2, 8, 0, 0);
  if (!rc && argc == 1 && info[0])
    rc |= fresh_prefill_check(session, 256, 8, 2, 17, 0);
  if (!rc && argc == 1 && info[0])
    rc |= fresh_prefill_check(session, 512, 2, 8, 17, 0);
  if (!rc && argc == 1 && info[0])
    rc |= fresh_prefill_check(session, 512, 2, 8, 0, 1);
  if (!rc && argc == 1 && info[7])
    rc |= raw_fp16_check(session);
  if (!rc && argc == 1 && info[0])
    rc |= long_hybrid_check(session, 256, 8, 2);
  if (!rc && argc == 1 && info[0])
    rc |= long_hybrid_check(session, 512, 2, 8);
  rc |= check(nntr_hvx_close(session), "close");
  printf("TURBOQUANT_DSP %s\n", rc ? "FAIL" : "PASS");
  return rc ? 1 : 0;
}
