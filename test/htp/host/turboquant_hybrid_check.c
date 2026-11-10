// SPDX-License-Identifier: Apache-2.0
/** @file turboquant_hybrid_check.c
 * @brief Hybrid Q8 prefix ownership, memory and shared softmax regression.
 */
#include "hvx_turboquant.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void prefix(hvx_tq_cache *cache, unsigned from, unsigned rows,
                   unsigned window) {
  const unsigned stride = 64, dim = 64;
  uint8_t *k = malloc(stride * dim), *v = malloc(stride * dim);
  float *sk = malloc(4), *sv = malloc(dim * 4);
  assert(k && v && sk && sv);
  memset(k, 128, stride * dim);
  memset(v, 128, stride * dim);
  *sk = 1;
  for (unsigned d = 0; d < dim; ++d)
    sv[d] = 1;
  for (unsigned r = 0; r < rows; ++r)
    memset(v + r * dim, 128 + (from ? 7 : r ? 3 : 1), dim);
  assert(hvx_tq_adopt_q8_prefix(cache, 0, from, rows, stride, window, k, v, sk,
                                sv) == 0);
}

int main(void) {
  int error = 0;
  hvx_tq_cache *cache = hvx_tq_create(1, 1, 1, 64, 256, NULL, &error);
  assert(cache && error == 0);
  for (unsigned run = 0; run < 4; ++run) {
    const unsigned window = run % 2 ? 17 : 0;
    prefix(cache, 0, 2, window);
    // No packed pages or FP32 history are retained by prefill.
    const size_t one_prefix = 2 * 64 * 64 + 4 * (1 + 64);
    assert(hvx_tq_payload_bytes(cache) == one_prefix);
    prefix(cache, 2, 1, window);
    assert(hvx_tq_payload_bytes(cache) == 2 * one_prefix);
    float q[64] = {0}, k[64] = {0}, v[64], out[64];
    for (unsigned d = 0; d < 64; ++d)
      v[d] = 5;
    assert(hvx_tq_lengths(cache)[0] == 3);
    for (unsigned pos = 3; pos < 120; ++pos) {
      if (pos == 35) {
        prefix(cache, pos, 2, window);
        pos += 2;
      }
      assert(hvx_tq_step(cache, 0, pos, 1, window, 0.125f, q, k, v, out) == 0);
      const unsigned start = window && pos + 1 > window ? pos + 1 - window : 0;
      double sum = 0;
      for (unsigned r = start; r <= pos; ++r)
        sum += r == 0 ? 1 : r == 1 ? 3 : r == 2 || r == 35 || r == 36 ? 7 : 5;
      const double expected = sum / (pos + 1 - start);
      for (unsigned d = 0; d < 64; ++d)
        assert(fabs(out[d] - expected) < 1e-5);
    }
    // A decode write cannot silently replace part of the immutable prefix.
    assert(hvx_tq_append(cache, 0, 1, 1, window, k, v) != 0);
    hvx_tq_reset(cache);
    assert(hvx_tq_payload_bytes(cache) == 0);
    assert(hvx_tq_lengths(cache)[0] == 0);
  }
  hvx_tq_free(cache);
  puts("Hybrid Q8 prefix: ownership, payload, shared softmax, chunks, sliding, "
       "reset PASS");
  return 0;
}
