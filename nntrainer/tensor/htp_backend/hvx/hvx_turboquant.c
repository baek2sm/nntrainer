// SPDX-License-Identifier: Apache-2.0
/** @file hvx_turboquant.c
 * @brief Fused packed four-bit attention; pages persist across MoE decode.
 * @author NNTrainer contributors
 * @bug Snapshot/session switching is not yet supported.
 */
#include "hvx_turboquant.h"
#include "../../../kv_cache/turboquant_codec.h"
#include <AEEStdErr.h>
#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#ifndef __hexagon__
#include <malloc.h>
#endif
#if defined(__hexagon__)
#include <HAP_perf.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#if __HEXAGON_ARCH__ < 79
#undef Q6_Vsf_vadd_VsfVsf
#undef Q6_Vsf_vsub_VsfVsf
#undef Q6_Vsf_vmpy_VsfVsf
#define Q6_Vsf_vadd_VsfVsf(a, b)                                               \
  Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf((a), (b)))
#define Q6_Vsf_vsub_VsfVsf(a, b)                                               \
  Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf((a), (b)))
#define Q6_Vsf_vmpy_VsfVsf(a, b)                                               \
  Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf((a), (b)))
#endif
#include "hexkl_attn_f16.h"
#include "hvx_attn_m1_f32.h"
#include "hvx_attn_math_f32.h"
#include "hvx_tile_f16.h"
#endif
#define TQ_TILE 32u
#define TQ_MAX_HD 512u
#define TQ_LOG2E 1.4426950408889634f
#define TQ_ALIGN __attribute__((aligned(128)))

typedef struct tq_page_block {
  struct tq_page_block *next;
} tq_page_block;
typedef struct tq_q8_prefix {
  struct tq_q8_prefix *next;
  uint32_t from, count, stride;
  uint8_t *k, *v;
  float *sk, *sv;
  void *native;
} tq_q8_prefix;
struct tq_append_task;
struct hvx_tq_cache {
  struct tq_append_task *pending;
  void (*prefill_cleanup)(void *);
  void *prefill_cleanup_ctx;
  uint32_t layers, heads, gqa, dim, max_seq, page_slots;
  uint32_t *lengths, *high_water, *capacity, *windows;
  uint8_t **pages;
  tq_page_block **blocks;
  tq_q8_prefix **prefix;
  uint32_t *prefix_end, *first_packed;
  size_t page_bytes;
  nntr_tq_codec codec;
  hvx_worker_pool *pool;
  float *prefix_state, *prefix_work;
  float *decode_input;
  uint16_t *native_exp;
  size_t prefix_state_capacity, prefix_work_capacity;
  float *partials;
  size_t partial_capacity;
  float *rotated_q;
  size_t rotated_capacity;
  uint32_t hmx_layer, hmx_bad_range;
  uint32_t *profile_us;
  float centroid_scale, centroid_i8_scale;
  uint8_t lut_i8[128] TQ_ALIGN;
  int16_t lut_q12[64] TQ_ALIGN;
  uint8_t lut_q12_low[128] TQ_ALIGN, lut_q12_high[128] TQ_ALIGN;
  uint16_t lut_low[64] TQ_ALIGN;
  uint16_t lut_high[64] TQ_ALIGN;
};
static size_t page_index(const hvx_tq_cache *c, uint32_t l, uint32_t h,
                         uint32_t row) {
  return ((size_t)l * c->heads + h) * c->page_slots +
         (row % c->capacity[l]) / TQ_TILE;
}
static uint8_t *page_at(const hvx_tq_cache *c, uint32_t l, uint32_t h,
                        uint32_t row) {
  return c->pages[page_index(c, l, h, row)];
}
static float *page_scales(const hvx_tq_cache *c, uint8_t *p) {
  return (float *)(p + c->dim * TQ_TILE);
}
static int finite_row(const float *x, size_t count) {
  size_t i = 0;
#if defined(__hexagon__)
  HVX_Vector max_exp = Q6_V_vzero();
  for (; i + 32 <= count; i += 32)
    max_exp =
      Q6_Vw_vmax_VwVw(max_exp, Q6_V_vand_VV(*(const HVX_UVector *)(x + i),
                                            Q6_V_vsplat_R(0x7f800000u)));
  for (uint32_t shift = 4; shift <= 64; shift *= 2)
    max_exp = Q6_Vw_vmax_VwVw(max_exp, Q6_V_vror_VR(max_exp, shift));
  if ((uint32_t)Q6_R_vextract_VR(max_exp, 0) == 0x7f800000u)
    return 0;
#endif
  for (; i < count; ++i) {
    uint32_t u;
    memcpy(&u, x + i, 4);
    if ((u & 0x7f800000u) == 0x7f800000u)
      return 0;
  }
  return 1;
}

hvx_tq_cache *hvx_tq_create(uint32_t layers, uint32_t heads, uint32_t gqa,
                            uint32_t dim, uint32_t max_seq,
                            hvx_worker_pool *pool, int *error) {
  int rc = AEE_EINVALIDFORMAT;
  hvx_tq_cache *c = NULL;
  const uint64_t slots = (uint64_t)layers * heads * ((max_seq + 31ull) / 32);
  if (!layers || !heads || !gqa || gqa > 8 || (uint64_t)heads * gqa > 64 ||
      dim < 64 || dim > TQ_MAX_HD || (dim & (dim - 1)) || !max_seq ||
      slots > 0x1fffffffu || hvx_worker_pool_workers(pool) >= 64)
    goto done;
  c = memalign(128, sizeof(*c));
  if (!c) {
    rc = AEE_ENOMEMORY;
    goto done;
  }
  memset(c, 0, sizeof(*c));
  c->layers = layers;
  c->heads = heads;
  c->gqa = gqa;
  c->dim = dim;
  c->max_seq = max_seq;
  c->page_slots = (max_seq + 31u) / 32u;
  c->pool = pool;
  // K [D/2][32] nibbles, V [row_quad][D][2], three FP32 metadata planes,
  // then an explicit guard vector for bounded unaligned HVX reads.
  c->page_bytes = (size_t)dim * 32u + 3u * 128u + 128u;
  c->pages = calloc((size_t)slots, sizeof(*c->pages));
  c->blocks = calloc(layers, sizeof(*c->blocks));
  c->prefix = calloc(layers, sizeof(*c->prefix));
  c->prefix_end = calloc(layers, sizeof(*c->prefix_end));
  c->first_packed = malloc(layers * sizeof(*c->first_packed));
  if (c->first_packed)
    for (uint32_t l = 0; l < layers; ++l)
      c->first_packed[l] = max_seq;
  c->lengths = calloc(layers, 4);
  c->high_water = calloc(layers, 4);
  c->capacity = calloc(layers, 4);
  c->windows = calloc(layers, 4);
  if (!c->prefix || !c->prefix_end || !c->first_packed || !c->blocks ||
      !c->pages || !c->lengths || !c->high_water || !c->capacity ||
      !c->windows) {
    rc = AEE_ENOMEMORY;
    hvx_tq_free(c);
    c = NULL;
    goto done;
  }
  nntr_tq_init(&c->codec, dim);
  c->centroid_scale = c->codec.centroids[15] / 4095.0f;
  c->centroid_i8_scale = c->codec.centroids[15] / 127.0f;
  for (uint32_t i = 0; i < 16; ++i) {
    const float code = c->codec.centroids[i] / c->centroid_scale;
    c->lut_q12[2 * i] = (int16_t)(code + (code < 0 ? -0.5f : 0.5f));
    for (uint32_t bank = 0; bank < 4; ++bank) {
      const uint16_t bits = (uint16_t)c->lut_q12[2 * i];
      c->lut_q12_low[bank * 32u + 2 * i] = (uint8_t)bits;
      c->lut_q12_high[bank * 32u + 2 * i] = (uint8_t)(bits >> 8);
    }
    const float code8 = c->codec.centroids[i] / c->centroid_i8_scale;
    const int32_t q8 = (int32_t)(code8 + (code8 < 0 ? -0.5f : 0.5f));
    for (uint32_t bank = 0; bank < 4; ++bank)
      c->lut_i8[bank * 32u + 2 * i] = (uint8_t)(q8 + 128);
    uint32_t u;
    memcpy(&u, c->codec.centroids + i, 4);
    c->lut_low[2 * i] = (uint16_t)u;
    c->lut_high[2 * i] = (uint16_t)(u >> 16);
  }
  for (uint32_t l = 0; l < layers; ++l)
    c->capacity[l] = c->page_slots * 32u;
  rc = AEE_SUCCESS;
done:
  if (error)
    *error = rc;
  return c;
}
void hvx_tq_prefill_cleanup(hvx_tq_cache *c, void (*cleanup)(void *),
                            void *ctx) {
  if (c) {
    c->prefill_cleanup = cleanup;
    c->prefill_cleanup_ctx = ctx;
  }
}
static void release_prefill(hvx_tq_cache *c) {
  if (c && c->prefill_cleanup) {
    void (*cleanup)(void *) = c->prefill_cleanup;
    c->prefill_cleanup = NULL;
    cleanup(c->prefill_cleanup_ctx);
  }
}
static void free_page_layer(hvx_tq_cache *c, uint32_t layer) {
  if (c->prefix) {
    tq_q8_prefix *p = c->prefix[layer];
    while (p) {
      tq_q8_prefix *next = p->next;
#if defined(__hexagon__)
      hvx_attn_m1_free(p->native);
#endif
      free(p->k);
      free(p->v);
      free(p->sk);
      free(p->sv);
      free(p);
      p = next;
    }
    c->prefix[layer] = NULL;
  }
  if (c->prefix_end)
    c->prefix_end[layer] = 0;
  if (c->first_packed)
    c->first_packed[layer] = c->max_seq;
  if (c->blocks) {
    tq_page_block *p = c->blocks[layer];
    while (p) {
      tq_page_block *next = p->next;
      free(p);
      p = next;
    }
    c->blocks[layer] = NULL;
  }
  if (c->pages)
    memset(c->pages + (size_t)layer * c->heads * c->page_slots, 0,
           (size_t)c->heads * c->page_slots * sizeof(*c->pages));
}
void hvx_tq_reset(hvx_tq_cache *c) {
  (void)hvx_tq_append_wait(c);
  release_prefill(c);
  if (!c)
    return;
  for (uint32_t layer = 0; layer < c->layers; ++layer)
    free_page_layer(c, layer);
  if (c->lengths)
    memset(c->lengths, 0, c->layers * 4u);
  if (c->high_water)
    memset(c->high_water, 0, c->layers * 4u);
}
void hvx_tq_free(hvx_tq_cache *c) {
  if (!c)
    return;
  if (c->pages)
    hvx_tq_reset(c);
  free(c->rotated_q);
  free(c->prefix_state);
  free(c->prefix_work);
  free(c->decode_input);
  free(c->native_exp);
  free(c->partials);
  free(c->pages);
  free(c->blocks);
  free(c->prefix);
  free(c->prefix_end);
  free(c->first_packed);
  free(c->lengths);
  free(c->high_water);
  free(c->capacity);
  free(c->windows);
  free(c);
}
uint32_t *hvx_tq_lengths(hvx_tq_cache *c) { return c ? c->lengths : NULL; }
size_t hvx_tq_payload_bytes(const hvx_tq_cache *c) {
  size_t bytes = 0;
  if (c)
    for (size_t i = 0; i < (size_t)c->layers * c->heads * c->page_slots; ++i)
      if (c->pages[i])
        bytes += c->page_bytes;
  if (c && c->prefix)
    for (uint32_t l = 0; l < c->layers; ++l)
      for (const tq_q8_prefix *p = c->prefix[l]; p; p = p->next)
#if defined(__hexagon__)
        bytes += p->native ? hvx_attn_m1_payload_bytes(p->native) +
                               (size_t)c->heads * 4u * (1u + c->dim)
                           : (size_t)c->heads *
                               (2ull * p->stride * c->dim + 4u * (1u + c->dim));
#else
        bytes +=
          (size_t)c->heads * (2ull * p->stride * c->dim + 4u * (1u + c->dim));
#endif
  return bytes;
}
int hvx_tq_adopt_q8_prefix(hvx_tq_cache *c, uint32_t layer, uint32_t from,
                           uint32_t count, uint32_t stride, uint32_t window,
                           uint8_t *k, uint8_t *v, float *sk, float *sv) {
  if (!c || layer >= c->layers || !count || from > c->max_seq ||
      count > c->max_seq - from || (stride % 32u) ||
      (uint64_t)stride < (uint64_t)count + 32u || !k || !v || !sk || !sv)
    return AEE_EINVALIDFORMAT;
  if (!finite_row(sk, c->heads) || !finite_row(sv, (size_t)c->heads * c->dim))
    return AEE_EINVALIDFORMAT;
  for (uint32_t h = 0; h < c->heads; ++h) {
    if (!(sk[h] > 0))
      return AEE_EINVALIDFORMAT;
    for (uint32_t d = 0; d < c->dim; ++d)
      if (!(sv[h * c->dim + d] > 0))
        return AEE_EINVALIDFORMAT;
  }
  int rc = hvx_tq_append_wait(c);
  if (rc)
    return rc;
  if (from && (from != c->lengths[layer] || window != c->windows[layer]))
    return AEE_EBADSTATE;
  tq_q8_prefix *p = calloc(1, sizeof(*p));
  if (!p)
    return AEE_ENOMEMORY;
  if (!from) {
    free_page_layer(c, layer);
    uint64_t cap =
      window && window < c->max_seq ? (uint64_t)window + 32u : c->max_seq;
    if (cap > c->max_seq)
      cap = c->max_seq;
    c->capacity[layer] = (uint32_t)((cap + 31u) / 32u) * 32u;
    c->windows[layer] = window;
  }
  *p =
    (tq_q8_prefix){c->prefix[layer], from, count, stride, k, v, sk, sv, NULL};
  c->prefix[layer] = p;
  c->lengths[layer] = c->high_water[layer] = c->prefix_end[layer] =
    from + count;
  return AEE_SUCCESS;
}
static __attribute__((unused)) const tq_q8_prefix *
prefix_row(const hvx_tq_cache *c, uint32_t layer, uint32_t row) {
  for (const tq_q8_prefix *p = c->prefix[layer]; p; p = p->next)
    if (row >= p->from && row - p->from < p->count)
      return p;
  return NULL;
}
// Retained Q8 chunks can bridge an interval older than the TQ ring,
// but a missing TQ row must never be treated as a retained prefix row.
static int prefix_covers(const hvx_tq_cache *c, uint32_t layer, uint32_t begin,
                         uint32_t end) {
  for (const tq_q8_prefix *p = c->prefix[layer]; p && begin < end;
       p = p->next) {
    if (p->from >= end)
      continue;
    if ((uint64_t)p->from + p->count < end)
      return 0;
    end = p->from;
  }
  return begin >= end;
}
typedef struct {
  hvx_tq_cache *c;
  uint32_t layer, from, count, first_tile, tiles, tile_rows;
  const float *k, *v;
  _Atomic uint32_t error;
  uint32_t *profile;
#if defined(__hexagon__)
  uint32_t key_us[64], value_us[64], store_us[64];
#endif
} append_job;
#if defined(__hexagon__)
static HVX_Vector packed_codes64(const uint8_t *p) {
  const HVX_Vector x = *(const HVX_UVector *)p,
                   mask = Q6_V_vsplat_R(0x0f0f0f0f);
  const HVX_Vector lo = Q6_V_vand_VV(x, mask),
                   hi = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(x, 4), mask);
  const HVX_Vector both = Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -1));
  return Q6_V_lo_W(Q6_Wuh_vunpack_Vub(both));
}
#endif
static void append_worker(uint32_t threads, uint32_t lane, void *data) {
  append_job *j = data;
  hvx_tq_cache *c = j->c;
  // Guard each row for bounded HVX reads. Four rows produce one complete
  // V quad and aligned four-byte K writes, avoiding per-nibble RMW stores.
  uint8_t key[4][TQ_MAX_HD / 2 + 128] TQ_ALIGN;
  uint8_t value[4][TQ_MAX_HD / 2 + 128] TQ_ALIGN;
  float scratch[TQ_MAX_HD] TQ_ALIGN;
  memset(key, 0, sizeof(key));
  memset(value, 0, sizeof(value));
  for (uint32_t u = lane; u < c->heads * j->tiles; u += threads) {
    const uint32_t h = u / j->tiles, t = j->first_tile + u % j->tiles;
    uint32_t row = t * j->tile_rows, hi = (t + 1u) * j->tile_rows;
    if (row < j->from)
      row = j->from;
    if (hi > j->from + j->count)
      hi = j->from + j->count;
    while (row < hi) {
      const uint32_t nr = !(row % 4u) && hi - row >= 4u ? 4u : 1u;
      uint8_t *p = page_at(c, j->layer, h, row);
      for (uint32_t i = 0; i < nr; ++i) {
        const size_t src =
          ((size_t)(row + i - j->from) * c->heads + h) * c->dim;
#if defined(__hexagon__)
        uint64_t tick = j->profile ? HAP_perf_get_time_us() : 0;
#endif
        const int key_rc =
          nntr_tq_encode_key(&c->codec, j->k + src, key[i], scratch);
#if defined(__hexagon__)
        if (j->profile && lane < 64) {
          j->key_us[lane] += HAP_perf_get_time_us() - tick;
          tick = HAP_perf_get_time_us();
        }
#endif
        const int value_rc =
          nntr_tq_encode_value(&c->codec, j->v + src, value[i]);
#if defined(__hexagon__)
        if (j->profile && lane < 64)
          j->value_us[lane] += HAP_perf_get_time_us() - tick;
#endif
        if (key_rc || value_rc) {
          atomic_store_explicit(&j->error, 1, memory_order_relaxed);
          return;
        }
        float *meta = page_scales(c, p);
        const uint32_t r = (row + i) % 32u;
        memcpy(meta + r, key[i] + c->dim / 2 + 4, 4);
        memcpy(meta + 32u + r, value[i] + c->dim / 2, 4);
        memcpy(meta + 64u + r, value[i] + c->dim / 2 + 4, 4);
      }
#if defined(__hexagon__)
      const uint64_t store_tick = j->profile ? HAP_perf_get_time_us() : 0;
#endif
      const uint32_t r = row % 32u;
      if (nr == 4) {
        for (uint32_t d = 0; d < c->dim / 2; ++d) {
#if defined(__hexagon__)
          const uint32_t word = key[0][d] | ((uint32_t)key[1][d] << 8) |
                                ((uint32_t)key[2][d] << 16) |
                                ((uint32_t)key[3][d] << 24);
          *(uint32_t *)(p + (size_t)d * 32u + r) = word;
#else
          for (uint32_t i = 0; i < 4; ++i)
            p[(size_t)d * 32u + r + i] = key[i][d];
#endif
        }
        uint8_t *vp = p + (size_t)c->dim * 16u + (size_t)(r / 4u) * c->dim * 2u;
#if defined(__hexagon__)
        for (uint32_t d = 0; d < c->dim; d += 64) {
          HVX_Vector codes = packed_codes64(value[0] + d / 2);
          codes = Q6_V_vor_VV(
            codes, Q6_Vh_vasl_VhR(packed_codes64(value[1] + d / 2), 4));
          codes = Q6_V_vor_VV(
            codes, Q6_Vh_vasl_VhR(packed_codes64(value[2] + d / 2), 8));
          codes = Q6_V_vor_VV(
            codes, Q6_Vh_vasl_VhR(packed_codes64(value[3] + d / 2), 12));
          *(HVX_Vector *)(vp + 2u * d) = codes;
        }
#else
        for (uint32_t d = 0; d < c->dim; ++d) {
          uint16_t codes = 0;
          for (uint32_t i = 0; i < 4; ++i)
            codes |= (uint16_t)(((value[i][d / 2] >> (4u * (d % 2u))) & 15u)
                                << (4u * i));
          vp[2 * d] = (uint8_t)codes;
          vp[2 * d + 1] = (uint8_t)(codes >> 8);
        }
#endif
      } else {
        for (uint32_t d = 0; d < c->dim / 2; ++d)
          p[(size_t)d * 32u + r] = key[0][d];
        uint8_t *vp = p + (size_t)c->dim * 16u + (size_t)(r / 4u) * c->dim * 2u;
        for (uint32_t d = 0; d < c->dim; ++d) {
          uint8_t *slot = vp + 2u * d + (r % 4u) / 2u;
          const uint8_t code = (value[0][d / 2] >> (4u * (d % 2u))) & 15u;
          const uint32_t shift = 4u * (r % 2u);
          *slot = (uint8_t)((*slot & ~(15u << shift)) | (code << shift));
        }
      }
#if defined(__hexagon__)
      if (j->profile && lane < 64)
        j->store_us[lane] += HAP_perf_get_time_us() - store_tick;
#endif
      row += nr;
    }
  }
}

#if defined(__hexagon__)
typedef struct {
  uint8_t *base;
  size_t vectors;
} zero_job;
static void zero_worker(uint32_t threads, uint32_t lane, void *data) {
  zero_job *j = data;
  const size_t lo = j->vectors * lane / threads,
               hi = j->vectors * (lane + 1u) / threads;
  for (size_t i = lo; i < hi; ++i)
    ((HVX_Vector *)j->base)[i] = Q6_V_vzero();
}
#endif
static int append_prepare(hvx_tq_cache *c, uint32_t layer, uint32_t from,
                          uint32_t count, uint32_t window, const float *keys,
                          const float *values, append_job *j) {
  if (!c || c->pending || layer >= c->layers || !count || from > c->max_seq ||
      count > c->max_seq - from || !keys || !values)
    return AEE_EINVALIDFORMAT;
  if (from > c->lengths[layer] || (from && from < c->prefix_end[layer]))
    return AEE_EBADSTATE;
  if (!finite_row(keys, (size_t)count * c->heads * c->dim) ||
      !finite_row(values, (size_t)count * c->heads * c->dim))
    return AEE_EINVALIDFORMAT;
  if (!from) {
    free_page_layer(c, layer);
    uint64_t cap = window && window < c->max_seq
                     ? (uint64_t)window + (count > 32u ? count : 32u)
                     : c->max_seq;
    if (cap > c->max_seq)
      cap = c->max_seq;
    c->capacity[layer] = (uint32_t)((cap + 31u) / 32u) * 32u;
    c->windows[layer] = window;
    c->high_water[layer] = 0;
    c->lengths[layer] = 0;
  }
  if (window != c->windows[layer])
    return AEE_EINVALIDFORMAT;
  const uint32_t earliest = c->high_water[layer] > c->capacity[layer]
                              ? c->high_water[layer] - c->capacity[layer]
                              : 0;
  const uint32_t needed = window && from + 1u > window ? from + 1u - window : 0;
  if ((needed < earliest && !prefix_covers(c, layer, needed, earliest)) ||
      (window && (uint64_t)count + window > c->capacity[layer] &&
       c->capacity[layer] < c->max_seq))
    return AEE_EBADSTATE;
  const uint32_t first = from / 32u, last = (from + count - 1u) / 32u;
  // Reserve each missing physical page once (a ring interval may visit
  // its first page twice). One aligned allocation backs the whole append
  // batch; page pointers remain stable and individual visibility is intact.
  const uint8_t *reserved = (const uint8_t *)(uintptr_t)1u;
  size_t missing = 0;
  for (uint32_t h = 0; h < c->heads; ++h)
    for (uint32_t t = first; t <= last; ++t) {
      const size_t at = page_index(c, layer, h, t * 32u);
      if (!c->pages[at]) {
        c->pages[at] = (uint8_t *)reserved;
        ++missing;
      }
    }
  if (missing) {
    tq_page_block *block = missing <= (SIZE_MAX - 128u) / c->page_bytes
                             ? memalign(128, 128u + missing * c->page_bytes)
                             : NULL;
    if (!block) {
      for (uint32_t h = 0; h < c->heads; ++h)
        for (uint32_t t = first; t <= last; ++t) {
          const size_t at = page_index(c, layer, h, t * 32u);
          if (c->pages[at] == reserved)
            c->pages[at] = NULL;
        }
      return AEE_ENOMEMORY;
    }
    block->next = c->blocks[layer];
    c->blocks[layer] = block;
    uint8_t *next = (uint8_t *)block + 128u;
#if defined(__hexagon__)
    zero_job zero = {next, missing * c->page_bytes / 128u};
    hvx_worker_pool_run(
      c->pool, zero_worker, &zero,
      (uint32_t)(zero.vectors > UINT32_MAX ? UINT32_MAX : zero.vectors));
#else
    memset(next, 0, missing * c->page_bytes);
#endif
    for (uint32_t h = 0; h < c->heads; ++h)
      for (uint32_t t = first; t <= last; ++t) {
        const size_t at = page_index(c, layer, h, t * 32u);
        if (c->pages[at] == reserved) {
          c->pages[at] = next;
          next += c->page_bytes;
        }
      }
  }
  memset(j, 0, sizeof(*j));
  atomic_init(&j->error, 0);
  j->c = c;
  j->layer = layer;
  j->from = from;
  j->count = count;
  j->first_tile = first;
  j->tiles = last - first + 1;
  j->tile_rows = 32;
  j->k = keys;
  j->v = values;
  j->profile = c->profile_us;
  return AEE_SUCCESS;
}
static int append_finish(hvx_tq_cache *c, append_job *j) {
  if (atomic_load_explicit(&j->error, memory_order_relaxed)) {
    c->lengths[j->layer] = 0;
    return AEE_EINVALIDFORMAT;
  }
#if defined(__hexagon__)
  if (j->profile)
    for (uint32_t i = 0; i < 64; ++i) {
      j->profile[4] += j->key_us[i];
      j->profile[5] += j->value_us[i];
      j->profile[6] += j->store_us[i];
    }
#endif
  if (j->from < c->first_packed[j->layer])
    c->first_packed[j->layer] = j->from;
  c->lengths[j->layer] = j->from + j->count;
  if (c->lengths[j->layer] > c->high_water[j->layer])
    c->high_water[j->layer] = c->lengths[j->layer];
  return AEE_SUCCESS;
}
int hvx_tq_append(hvx_tq_cache *c, uint32_t layer, uint32_t from,
                  uint32_t count, uint32_t window, const float *keys,
                  const float *values) {
  const int pending_rc = hvx_tq_append_wait(c);
  if (pending_rc)
    return pending_rc;
  append_job job;
  const int rc =
    append_prepare(c, layer, from, count, window, keys, values, &job);
  if (rc)
    return rc;
  hvx_worker_pool_run(c->pool, append_worker, &job, c->heads * job.tiles);
  return append_finish(c, &job);
}
#if defined(__hexagon__)
struct tq_append_task {
  append_job append;
  hvx_bg_job bg;
};
#endif
int hvx_tq_append_begin(hvx_tq_cache *c, uint32_t layer, uint32_t from,
                        uint32_t count, uint32_t window, const float *keys,
                        const float *values) {
#if defined(__hexagon__)
  if (!c || c->pending)
    return AEE_EBADSTATE;
  struct tq_append_task *task = calloc(1, sizeof(*task));
  if (!task)
    return AEE_ENOMEMORY;
  int rc =
    append_prepare(c, layer, from, count, window, keys, values, &task->append);
  if (rc) {
    free(task);
    return rc;
  }
  append_job *j = &task->append;
  j->profile = NULL;
  // One four-row unit owns a complete V quad. Foreground HMX work can
  // preempt between quads, without any concurrent nibble RMW stores.
  j->tile_rows = 4;
  j->first_tile = from / 4u;
  j->tiles = (from + count + 3u) / 4u - j->first_tile;
  task->bg.func = append_worker;
  task->bg.ctx = j;
  task->bg.n_units = c->heads * j->tiles;
  task->bg.done = calloc(task->bg.n_units, 1);
  if (!task->bg.done) {
    free(task);
    return AEE_ENOMEMORY;
  }
  c->pending = task;
  // Publish the logical append for graph preflight. Every data consumer
  // joins pending work before touching pages; the input owner retains it.
  c->lengths[layer] = from + count;
  hvx_worker_pool_submit_bg(c->pool, &task->bg);
  return AEE_SUCCESS;
#else
  return hvx_tq_append(c, layer, from, count, window, keys, values);
#endif
}
int hvx_tq_append_wait(hvx_tq_cache *c) {
  if (!c)
    return AEE_EBADSTATE;
#if defined(__hexagon__)
  struct tq_append_task *task = c->pending;
  if (!task)
    return AEE_SUCCESS;
  hvx_worker_pool_wait_bg(c->pool, &task->bg, UINT32_MAX);
  int rc = append_finish(c, &task->append);
  c->pending = NULL;
  free(task->bg.done);
  free(task);
  return rc;
#else
  return AEE_SUCCESS;
#endif
}

#if defined(__hexagon__)
static HVX_Vector splat(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return Q6_V_vsplat_R(u);
}
static int prefix_workspace(hvx_tq_cache *c, uint32_t count) {
  if (!c->native_exp) {
    c->native_exp = hvx_attn_m1_create_exp_table();
    if (!c->native_exp)
      return AEE_ENOMEMORY;
  }
  const size_t heads = c->heads * c->gqa;
  const size_t work_bytes = heads * (c->dim + 2u) * 4u;
  const size_t state_bytes = (size_t)count * work_bytes;
  if (state_bytes > 96u * 1024u * 1024u)
    return AEE_ENOMEMORY;
  if (work_bytes > c->prefix_work_capacity) {
    float *next = memalign(128, work_bytes);
    if (!next)
      return AEE_ENOMEMORY;
    free(c->prefix_work);
    c->prefix_work = next;
    c->prefix_work_capacity = work_bytes;
  }
  if (state_bytes > c->prefix_state_capacity) {
    float *next = memalign(128, state_bytes);
    if (!next)
      return AEE_ENOMEMORY;
    free(c->prefix_state);
    c->prefix_state = next;
    c->prefix_state_capacity = state_bytes;
  }
  memset(c->prefix_state, 0, state_bytes);
  return AEE_SUCCESS;
}
static int prefix_prepare(hvx_tq_cache *c, tq_q8_prefix *p) {
  if (p->native)
    return AEE_SUCCESS;
  int error = 0;
  p->native = hvx_attn_m1_q8_prefix_create(c->heads, c->gqa, c->dim, p->count,
                                           p->stride, p->k, p->v, p->sk, p->sv,
                                           c->native_exp, c->pool, &error);
  if (!p->native)
    return error;
  free(p->k);
  free(p->v);
  p->k = p->v = NULL;
  return AEE_SUCCESS;
}
static int prefix_native(hvx_tq_cache *c, uint32_t layer, uint32_t from,
                         uint32_t count, uint32_t window, float scale,
                         const float *q) {
  int rc = prefix_workspace(c, count);
  if (rc)
    return rc;
  const size_t heads = c->heads * c->gqa;

  for (tq_q8_prefix *p = c->prefix[layer]; p; p = p->next) {
    const uint32_t last = from + count;
    const uint32_t first =
      window && from + 1u > window ? from + 1u - window : 0;
    if (p->from >= last || p->from + p->count <= first)
      continue;
    rc = prefix_prepare(c, p);
    if (rc)
      return rc;
    for (uint32_t qi = 0; qi < count; ++qi) {
      const uint32_t end = from + qi + 1u;
      uint32_t lo = window && end > window ? end - window : 0;
      uint32_t hi = end < p->from + p->count ? end : p->from + p->count;
      if (lo < p->from)
        lo = p->from;
      if (lo >= hi)
        continue;
      float *stats = c->prefix_work + heads * c->dim;
      int rc = hvx_attn_m1_forward_cached(
        p->native, hi - p->from - 1u, hi - lo, scale,
        q + (size_t)qi * heads * c->dim, c->prefix_work, stats);
      if (rc)
        return rc;
      for (size_t h = 0; h < heads; ++h) {
        if (!(stats[2 * h + 1] > 0))
          continue;
        const float maximum = stats[2 * h] * TQ_LOG2E;
        float *dst = c->prefix_state + ((size_t)qi * heads + h) * (c->dim + 2u);
        const float *src = c->prefix_work + h * c->dim;
        const float next =
          dst[c->dim + 1] > 0 ? fmaxf(dst[c->dim], maximum) : maximum;
        const float wa = dst[c->dim + 1] > 0 ? exp2f(dst[c->dim] - next) : 0;
        const float wb = exp2f(maximum - next) * stats[2 * h + 1];
        for (uint32_t d = 0; d < c->dim; d += 32)
          *(HVX_UVector *)(dst + d) = Q6_Vsf_vadd_VsfVsf(
            Q6_Vsf_vmpy_VsfVsf(*(const HVX_UVector *)(dst + d), splat(wa)),
            Q6_Vsf_vmpy_VsfVsf(*(const HVX_UVector *)(src + d), splat(wb)));
        dst[c->dim] = next;
        dst[c->dim + 1] = dst[c->dim + 1] * wa + wb;
      }
    }
  }
  return AEE_SUCCESS;
}
static __attribute__((unused)) HVX_Vector
lut32(HVX_Vector bytes, HVX_Vector low, HVX_Vector high) {
  const HVX_Vector idx = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(bytes));
  const HVX_Vector a = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR(idx, low, 0));
  const HVX_Vector b = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR(idx, high, 0));
  return Q6_V_vor_VV(Q6_V_lo_W(Q6_Wuw_vunpack_Vuh(a)),
                     Q6_Vw_vasl_VwR(Q6_V_lo_W(Q6_Wuw_vunpack_Vuh(b)), 16));
}
static __attribute__((unused)) HVX_Vector value32(const uint8_t *p) {
  const HVX_Vector x = *(const HVX_UVector *)p,
                   mask = Q6_V_vsplat_R(0x0f0f0f0f);
  const HVX_Vector lo = Q6_V_vand_VV(x, mask),
                   hi = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(x, 4), mask);
  const HVX_Vector both = Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -1));
  return Q6_V_lo_W(Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(Q6_Wuh_vunpack_Vub(both))));
}
#endif

typedef struct {
  hvx_tq_cache *c;
  uint32_t layer, from, count, window, group, groups, splits, units;
  float scale;
  const float *q, *prefix_state, *current_k, *current_v;
  float *partials, *output;
} attention_job;
static void attention_unit(attention_job *j, uint32_t unit) {
  hvx_tq_cache *c = j->c;
  const uint32_t hd = c->dim;
  const uint32_t split = unit % j->splits, chain = unit / j->splits;
  const uint32_t group = chain % (c->heads * j->groups),
                 qi = chain / (c->heads * j->groups);
  const uint32_t h = group / j->groups, g0 = (group % j->groups) * j->group;
  const uint32_t ng = c->gqa - g0 < j->group ? c->gqa - g0 : j->group;
  const uint32_t end = j->from + qi + 1u,
                 start = j->window && end > j->window ? end - j->window : 0;
  const uint32_t first = j->prefix_state && c->first_packed[j->layer] > start
                           ? c->first_packed[j->layer]
                           : start;
  const uint32_t history_end = j->current_k ? end - 1u : end;
  const uint32_t t1 = (history_end + 31u) / 32u;
  const uint32_t t0 = first < history_end ? first / 32u : t1, tiles = t1 - t0;
  const uint32_t begin = t0 + (uint32_t)((uint64_t)tiles * split / j->splits);
  const uint32_t finish =
    t0 + (uint32_t)((uint64_t)tiles * (split + 1u) / j->splits);
  float q[2][TQ_MAX_HD] TQ_ALIGN;
  float *part =
    j->partials ? j->partials + (size_t)unit * j->group * (hd + 2u) : NULL;
  for (uint32_t g = 0; g < ng; ++g) {
    memcpy(q[g],
           j->q + ((size_t)qi * c->heads * c->gqa + h * c->gqa + g0 + g) * hd,
           hd * 4u);
    nntr_tq_rotate(q[g], hd);
  }
#if defined(__hexagon__)
  const HVX_Vector zero = Q6_V_vzero(), neg = Q6_V_vsplat_R(0xff800000u);
  // Byte dots pay off only for long global history on the measured model
  // shape. Keep short/sliding and packed-only attention on Q12 to avoid
  // extra score approximation where it has no demonstrated benefit.
  // Persistent 4-bit codes are unchanged.
  const int score_i8 = c->prefix[j->layer] && !j->window &&
                       ((hd == 256 && c->heads == 8 && c->gqa == 2) ||
                        (hd == 512 && c->heads == 2 && c->gqa == 8)) &&
                       history_end > first && history_end - first >= 4096;
  float qscale[2];
  int8_t quantized8[2][TQ_MAX_HD] TQ_ALIGN;
  int16_t quantized[2][TQ_MAX_HD] TQ_ALIGN;
  int32_t qsum[2];
  const HVX_Vector table_i8 = *(const HVX_Vector *)c->lut_i8;
  const HVX_Vector table_low = *(const HVX_Vector *)c->lut_q12_low;
  const HVX_Vector table_high = *(const HVX_Vector *)c->lut_q12_high;
  if (score_i8) {

    for (uint32_t g = 0; g < ng; ++g) {
      HVX_Vector vmax = zero;
      for (uint32_t d = 0; d < hd; d += 32)
        vmax =
          Q6_Vsf_vmax_VsfVsf(vmax, Q6_V_vand_VV(*(const HVX_Vector *)(q[g] + d),
                                                Q6_V_vsplat_R(0x7fffffff)));
      const float maximum = hvx_attn_lane0_f32(hvx_attn_max32_sf(vmax));
      const float inv = maximum > 1e-30f ? 127.0f / maximum : 1;
      qscale[g] = (1.0f / inv) * c->centroid_i8_scale;
      HVX_Vector sum = zero;
      for (uint32_t d = 0; d < hd; d += 128) {
        HVX_Vector w[4];
        const uint32_t vectors = hd - d < 128 ? 2u : 4u;
        for (uint32_t n = 0; n < vectors; ++n) {
          w[n] = hvx_sf_to_w_rne(Q6_Vsf_vmpy_VsfVsf(
            *(const HVX_Vector *)(q[g] + d + n * 32u), splat(inv)));
          sum = Q6_Vw_vadd_VwVw(sum, w[n]);
        }
        const HVX_Vector a = Q6_Vh_vpack_VwVw_sat(w[1], w[0]);
        const HVX_Vector b =
          vectors == 4 ? Q6_Vh_vpack_VwVw_sat(w[3], w[2]) : a;
        const HVX_Vector bytes = Q6_Vb_vpack_VhVh_sat(b, a);
        memcpy(quantized8[g] + d, &bytes, vectors * 32u);
      }
      for (uint32_t shift = 4; shift <= 64; shift *= 2)
        sum = Q6_Vw_vadd_VwVw(sum, Q6_V_vror_VR(sum, shift));
      qsum[g] = Q6_R_vextract_VR(sum, 0);
    }

  } else {

    for (uint32_t g = 0; g < ng; ++g) {
      HVX_Vector vmax = zero, vsum = zero;
      const HVX_Vector mask = Q6_V_vsplat_R(0x7fffffff);
      for (uint32_t d = 0; d < hd; d += 32) {
        const HVX_Vector value =
          Q6_V_vand_VV(*(const HVX_Vector *)(q[g] + d), mask);
        vmax = Q6_Vsf_vmax_VsfVsf(vmax, value);
        vsum = Q6_Vsf_vadd_VsfVsf(vsum, value);
      }
      const float maximum = hvx_attn_lane0_f32(hvx_attn_max32_sf(vmax));
      const float sum = hvx_attn_lane0_f32(hvx_attn_sum32_sf(vsum));
      // FP32 bounds retain headroom below the saturating int32 accumulator.
      // Very small logits may quantize to zero without overflowing 1/max.
      float inv = maximum > 1e-30f ? 32767.0f / maximum : 1;
      if (maximum > 1e30f)
        inv = (1.0e9f / (4095.0f * hd)) / maximum;
      else if (sum > 0) {
        const float bound =
          1.0e9f / (4095.0f * sum + 4095.0f * hd * maximum / 65534.0f);
        if (inv > bound)
          inv = bound;
      }
      qscale[g] = (1.0f / inv) * c->centroid_scale;
      const HVX_Vector vi = splat(inv);
      for (uint32_t d = 0; d < hd; d += 64) {
        const HVX_Vector lo = hvx_sf_to_w_rne(
          Q6_Vsf_vmpy_VsfVsf(*(const HVX_Vector *)(q[g] + d), vi));
        const HVX_Vector hi = hvx_sf_to_w_rne(
          Q6_Vsf_vmpy_VsfVsf(*(const HVX_Vector *)(q[g] + d + 32), vi));
        *(HVX_Vector *)(quantized[g] + d) = Q6_Vh_vpack_VwVw_sat(hi, lo);
      }
    }
  }

  HVX_Vector acc[2][TQ_MAX_HD / 32], m[2], den[2];
  for (uint32_t g = 0; g < ng; ++g) {
    m[g] = neg;
    den[g] = zero;
    for (uint32_t d = 0; d < hd / 32; ++d)
      acc[g][d] = zero;
    if (j->prefix_state && split == 0) {
      const float *p =
        j->prefix_state +
        ((size_t)qi * c->heads * c->gqa + h * c->gqa + g0 + g) * (hd + 2u);
      if (p[hd + 1] > 0) {
        m[g] = splat(p[hd]);
        den[g] = splat(p[hd + 1]);
        for (uint32_t d = 0; d < hd / 32; ++d)
          acc[g][d] = *(const HVX_UVector *)(p + d * 32u);
      }
    }
  }
  int32_t indices[32] TQ_ALIGN;
  for (uint32_t r = 0; r < 32; ++r)
    indices[r] = r;
  for (uint32_t t = begin; t < finish && t < begin + 2u; ++t) {
    const void *next = page_at(c, j->layer, h, t * 32u);
    if (next)
      Q6_l2fetch_AP((void *)next,
                    (128ull << 32) | (128ull << 16) | (c->page_bytes / 128u));
  }
  for (uint32_t tile = begin; tile < finish; ++tile) {
    if (tile + 2u < finish) {
      const void *next = page_at(c, j->layer, h, (tile + 2u) * 32u);
      if (next)
        Q6_l2fetch_AP((void *)next,
                      (128ull << 32) | (128ull << 16) | (c->page_bytes / 128u));
    }
    const uint32_t row0 = tile * 32u;
    uint32_t q8_mask = 0;
    for (const tq_q8_prefix *p8 = c->prefix[j->layer]; p8; p8 = p8->next) {
      uint32_t lo = row0 > p8->from ? row0 : p8->from;
      uint32_t hi =
        row0 + 32u < p8->from + p8->count ? row0 + 32u : p8->from + p8->count;
      if (lo < hi)
        q8_mask |= (uint32_t)(((1ull << (hi - lo)) - 1ull) << (lo - row0));
    }
    const uint32_t visible_lo = start > row0 ? start - row0 : 0;
    const uint32_t visible_hi =
      history_end < row0 + 32u ? history_end - row0 : 32u;
    const uint32_t visible_bits =
      (uint32_t)(((1ull << (visible_hi - visible_lo)) - 1ull) << visible_lo);
    if (!(visible_bits & ~q8_mask))
      continue;
    const uint8_t *p = page_at(c, j->layer, h, row0);
    const float *meta = page_scales(c, (uint8_t *)p);
    HVX_Vector score[2] = {zero, zero};
    if (score_i8) {

      for (uint32_t base = 0; base < hd / 2; base += 4) {
        const HVX_Vector codes = *(const HVX_Vector *)(p + (size_t)base * 32u);
        const HVX_Vector mask = Q6_V_vsplat_R(0x0f0f0f0f);
        const HVX_Vector lo =
          Q6_Vb_vlut32_VbVbI(Q6_V_vand_VV(codes, mask), table_i8, 0);
        const HVX_Vector hi = Q6_Vb_vlut32_VbVbI(
          Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(codes, 4), mask), table_i8, 0);
        const HVX_VectorPair pair = Q6_W_vshuff_VVR(hi, lo, -1);
        const HVX_Vector a = Q6_V_lo_W(pair), b = Q6_V_hi_W(pair);
        const HVX_Vector ka =
          Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vror_VR(a, 64), a, -2));
        const HVX_Vector kb =
          Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vror_VR(b, 64), b, -2));
        for (uint32_t g = 0; g < ng; ++g) {
          uint32_t qa, qb;
          memcpy(&qa, quantized8[g] + 2 * base, 4);
          memcpy(&qb, quantized8[g] + 2 * base + 4, 4);
          score[g] = Q6_Vw_vrmpyacc_VwVubRb(score[g], ka, qa);
          score[g] = Q6_Vw_vrmpyacc_VwVubRb(score[g], kb, qb);
        }
      }
      for (uint32_t g = 0; g < ng; ++g)
        score[g] = Q6_Vw_vsub_VwVw(score[g], Q6_V_vsplat_R(128 * qsum[g]));

    } else {

      for (uint32_t base = 0; base < hd / 2; base += 4) {
        const HVX_Vector block = *(const HVX_Vector *)(p + (size_t)base * 32u);
        const HVX_Vector mask = Q6_V_vsplat_R(0x0f0f0f0f);
        const HVX_Vector lo = Q6_V_vand_VV(block, mask);
        const HVX_Vector hi = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(block, 4), mask);
        // Four dimension pairs share one 128-byte load. Byte-plane lookups
        // reconstruct the same signed Q12 centroids without widening each
        // 32-row plane separately. vlut32 uses table byte 2*index on v81.
        const HVX_VectorPair a =
          Q6_W_vshuff_VVR(Q6_Vb_vlut32_VbVbI(lo, table_high, 0),
                          Q6_Vb_vlut32_VbVbI(lo, table_low, 0), -1);
        const HVX_VectorPair b =
          Q6_W_vshuff_VVR(Q6_Vb_vlut32_VbVbI(hi, table_high, 0),
                          Q6_Vb_vlut32_VbVbI(hi, table_low, 0), -1);
        const HVX_VectorPair ab0 =
          Q6_W_vshuff_VVR(Q6_V_lo_W(b), Q6_V_lo_W(a), -2);
        const HVX_VectorPair ab1 =
          Q6_W_vshuff_VVR(Q6_V_hi_W(b), Q6_V_hi_W(a), -2);
        const HVX_Vector pairs[4] = {Q6_V_lo_W(ab0), Q6_V_hi_W(ab0),
                                     Q6_V_lo_W(ab1), Q6_V_hi_W(ab1)};
#pragma clang loop unroll(full)
        for (uint32_t pair_index = 0; pair_index < 4; ++pair_index)
          for (uint32_t g = 0; g < ng; ++g) {
            uint32_t qw;
            memcpy(&qw, quantized[g] + 2 * (base + pair_index), 4);
            score[g] =
              Q6_Vw_vdmpyacc_VwVhRh_sat(score[g], pairs[pair_index], qw);
          }
      }
    }

    const HVX_Vector row =
      Q6_Vw_vadd_VwVw(*(HVX_Vector *)indices, Q6_V_vsplat_R(row0));
    HVX_VectorPred visible =
      Q6_Q_and_QQ(Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R(history_end), row),
                  Q6_Q_not_Q(Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R(start), row)));
    if (q8_mask) {
      int32_t tq_rows[32] TQ_ALIGN;
      for (uint32_t r = 0; r < 32; ++r)
        tq_rows[r] = (q8_mask >> r) & 1u;
      visible = Q6_Q_and_QQ(
        visible, Q6_Q_vcmp_eq_VwVw(*(const HVX_Vector *)tq_rows, zero));
    }
    float weights[2][32] TQ_ALIGN;
    for (uint32_t g = 0; g < ng; ++g) {
      HVX_Vector s = Q6_Vsf_vmpy_VsfVsf(
        Q6_Vsf_equals_Vw(score[g]),
        Q6_Vsf_vmpy_VsfVsf(*(const HVX_Vector *)meta,
                           splat(j->scale * TQ_LOG2E * qscale[g])));
      s = Q6_V_vmux_QVV(visible, s, neg);
      const HVX_Vector next = Q6_Vsf_vmax_VsfVsf(m[g], hvx_attn_max32_sf(s));
      const HVX_Vector alpha =
        hvx_attn_lane0_f32(den[g]) > 0
          ? hvx_attn_exp2_f32(Q6_Vsf_vsub_VsfVsf(m[g], next))
          : zero;
      HVX_Vector prob = hvx_attn_exp2_f32(Q6_Vsf_vsub_VsfVsf(s, next));
      prob = Q6_V_vmux_QVV(visible, prob, zero);
      den[g] = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(den[g], alpha),
                                  hvx_attn_sum32_sf(prob));
      const HVX_Vector offset = hvx_attn_sum32_sf(
        Q6_Vsf_vmpy_VsfVsf(prob, *(const HVX_Vector *)(meta + 64)));
      *(HVX_Vector *)weights[g] =
        Q6_Vsf_vmpy_VsfVsf(prob, *(const HVX_Vector *)(meta + 32));
      for (uint32_t d = 0; d < hd / 32; ++d)
        acc[g][d] =
          Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(acc[g][d], alpha), offset);
      m[g] = next;
    }
    uint32_t lo = start > row0 ? start - row0 : 0,
             hi = history_end - row0 < 32 ? history_end - row0 : 32;
    uint8_t pl[2][32] TQ_ALIGN, ph[2][32] TQ_ALIGN;
    float ps[2];
    for (uint32_t g = 0; g < ng; ++g) {
      const HVX_Vector values = *(const HVX_Vector *)weights[g];
      const float maximum = hvx_attn_lane0_f32(hvx_attn_max32_sf(values));
      ps[g] = maximum > 0 ? maximum / 32767.0f : 1;
      if (ps[g] >= 1e-30f) {
        HVX_Vector w =
          hvx_sf_to_w_rne(Q6_Vsf_vmpy_VsfVsf(values, splat(1.0f / ps[g])));
        w = Q6_Vw_vmax_VwVw(w, zero);
        w = Q6_Vw_vmin_VwVw(w, Q6_V_vsplat_R(32767));
        const HVX_Vector half = Q6_Vh_vpack_VwVw_sat(w, w);
        const HVX_Vector low = Q6_Vb_vpacke_VhVh(half, half);
        const HVX_Vector high_half = Q6_Vuh_vlsr_VuhR(half, 8);
        const HVX_Vector high = Q6_Vb_vpacke_VhVh(high_half, high_half);
        memcpy(pl[g], &low, 32);
        memcpy(ph[g], &high, 32);
      } else {
        // Rare tiny V scales retain finite division instead of 1/scale
        // overflow.
        memset(pl[g], 0, 32);
        memset(ph[g], 0, 32);
        for (uint32_t r = lo; r < hi; ++r) {
          uint32_t w = (uint32_t)(weights[g][r] / ps[g] + 0.5f);
          if (w > 32767)
            w = 32767;
          pl[g][r] = (uint8_t)w;
          ph[g][r] = (uint8_t)(w >> 8);
        }
      }
    }
    for (uint32_t d = 0; d < hd; d += 64) {
      HVX_Vector a0[2] = {zero, zero}, a1[2] = {zero, zero},
                 b0[2] = {zero, zero}, b1[2] = {zero, zero};
      for (uint32_t quad = 0; quad < 8; ++quad) {
        const uint8_t *vp =
          p + (size_t)hd * 16u + (size_t)quad * hd * 2u + d * 2u;
        const HVX_Vector packed = *(const HVX_Vector *)vp;
        const HVX_Vector mask = Q6_V_vsplat_R(0x0f0f0f0f);
        const HVX_Vector low = Q6_V_vand_VV(packed, mask);
        const HVX_Vector high = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(packed, 4), mask);
        const HVX_VectorPair expanded = Q6_W_vshuff_VVR(high, low, -1);
        for (uint32_t g = 0; g < ng; ++g) {
          uint32_t wl, wh;
          memcpy(&wl, pl[g] + 4 * quad, 4);
          memcpy(&wh, ph[g] + 4 * quad, 4);
          a0[g] = Q6_Vuw_vrmpyacc_VuwVubRub(a0[g], Q6_V_lo_W(expanded), wl);
          a1[g] = Q6_Vuw_vrmpyacc_VuwVubRub(a1[g], Q6_V_lo_W(expanded), wh);
          b0[g] = Q6_Vuw_vrmpyacc_VuwVubRub(b0[g], Q6_V_hi_W(expanded), wl);
          b1[g] = Q6_Vuw_vrmpyacc_VuwVubRub(b1[g], Q6_V_hi_W(expanded), wh);
        }
      }
      for (uint32_t g = 0; g < ng; ++g) {
        const HVX_Vector a = Q6_Vw_vadd_VwVw(a0[g], Q6_Vw_vasl_VwR(a1[g], 8));
        const HVX_Vector b = Q6_Vw_vadd_VwVw(b0[g], Q6_Vw_vasl_VwR(b1[g], 8));
        acc[g][d / 32] = Q6_Vsf_vadd_VsfVsf(
          acc[g][d / 32],
          Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_equals_Vw(a), splat(ps[g])));
        acc[g][d / 32 + 1] = Q6_Vsf_vadd_VsfVsf(
          acc[g][d / 32 + 1],
          Q6_Vsf_vmpy_VsfVsf(Q6_Vsf_equals_Vw(b), splat(ps[g])));
      }
    }
  }
  if (j->current_k && split == 0)
    for (uint32_t g = 0; g < ng; ++g) {
      const float *original =
        j->q + ((size_t)qi * c->heads * c->gqa + h * c->gqa + g0 + g) * hd;
      HVX_Vector dot = zero;
      for (uint32_t d = 0; d < hd; d += 32)
        dot = Q6_Vsf_vadd_VsfVsf(
          dot, Q6_Vsf_vmpy_VsfVsf(
                 *(const HVX_UVector *)(original + d),
                 *(const HVX_UVector *)(j->current_k + (size_t)h * hd + d)));
      const HVX_Vector score =
        Q6_Vsf_vmpy_VsfVsf(hvx_attn_sum32_sf(dot), splat(j->scale * TQ_LOG2E));
      const HVX_Vector next = Q6_Vsf_vmax_VsfVsf(m[g], score);
      const HVX_Vector alpha =
        hvx_attn_lane0_f32(den[g]) > 0
          ? hvx_attn_exp2_f32(Q6_Vsf_vsub_VsfVsf(m[g], next))
          : zero;
      const HVX_Vector prob =
        hvx_attn_exp2_f32(Q6_Vsf_vsub_VsfVsf(score, next));
      for (uint32_t d = 0; d < hd / 32; ++d)
        acc[g][d] = Q6_Vsf_vadd_VsfVsf(
          Q6_Vsf_vmpy_VsfVsf(acc[g][d], alpha),
          Q6_Vsf_vmpy_VsfVsf(
            *(const HVX_UVector *)(j->current_v + (size_t)h * hd + d * 32u),
            prob));
      den[g] = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vmpy_VsfVsf(den[g], alpha), prob);
      m[g] = next;
    }
  for (uint32_t g = 0; g < ng; ++g) {
    float *o = j->splits == 1
                 ? j->output +
                     ((size_t)qi * c->heads * c->gqa + h * c->gqa + g0 + g) * hd
                 : part + g * (hd + 2u);
    const HVX_Vector inv =
      j->splits == 1 ? hvx_attn_recip_pos_f32(den[g]) : splat(1);
    for (uint32_t d = 0; d < hd / 32; ++d)
      *(HVX_UVector *)(o + d * 32u) = Q6_Vsf_vmpy_VsfVsf(acc[g][d], inv);
    if (j->splits > 1) {
      o[hd] = hvx_attn_lane0_f32(m[g]);
      o[hd + 1] = hvx_attn_lane0_f32(den[g]);
    }
  }
#else
  for (uint32_t g = 0; g < ng; ++g) {
    float *o = j->splits == 1
                 ? j->output +
                     ((size_t)qi * c->heads * c->gqa + h * c->gqa + g0 + g) * hd
                 : part + g * (hd + 2u);
    memset(o, 0, hd * 4u);
    float m = -INFINITY, den = 0;
    for (uint32_t r = begin * 32u; r < finish * 32u && r < history_end; ++r) {
      if (r < start)
        continue;
      const tq_q8_prefix *p8 = prefix_row(c, j->layer, r);
      if (p8) {
        const size_t at = ((size_t)h * p8->stride + r - p8->from) * hd;
        const float *original_q =
          j->q + ((size_t)qi * c->heads * c->gqa + h * c->gqa + g0 + g) * hd;
        float score = 0;
        for (uint32_t d = 0; d < hd; ++d)
          score += original_q[d] * ((int)p8->k[at + d] - 128);
        score *= p8->sk[h] * j->scale * TQ_LOG2E;
        const float next = fmaxf(m, score), alpha = exp2f(m - next),
                    prob = exp2f(score - next);
        for (uint32_t d = 0; d < hd; ++d)
          o[d] = o[d] * alpha +
                 prob * ((int)p8->v[at + d] - 128) * p8->sv[h * hd + d];
        den = den * alpha + prob;
        m = next;
        continue;
      }
      const uint8_t *p = page_at(c, j->layer, h, r);
      const float *meta = page_scales(c, (uint8_t *)p);
      const uint32_t rr = r % 32u;
      float score = 0;
      for (uint32_t d = 0; d < hd / 2; ++d) {
        const uint8_t code = p[(size_t)d * 32 + rr];
        score += q[g][2 * d] * c->codec.centroids[code & 15];
        score += q[g][2 * d + 1] * c->codec.centroids[code >> 4];
      }
      score *= meta[rr] * j->scale * TQ_LOG2E;
      const float next = fmaxf(m, score), alpha = exp2f(m - next),
                  prob = exp2f(score - next);
      const uint8_t *v = p + (size_t)hd * 16u + (size_t)(rr / 4) * hd * 2u;
      for (uint32_t d = 0; d < hd; ++d)
        o[d] =
          o[d] * alpha +
          prob * ((float)((v[2 * d + (rr % 4) / 2] >> (4 * (rr % 2))) & 15) *
                    meta[32 + rr] +
                  meta[64 + rr]);
      den = den * alpha + prob;
      m = next;
    }
    if (j->current_k && split == 0) {
      const float *original =
        j->q + ((size_t)qi * c->heads * c->gqa + h * c->gqa + g0 + g) * hd;
      float score = 0;
      for (uint32_t d = 0; d < hd; ++d)
        score += original[d] * j->current_k[(size_t)h * hd + d];
      score *= j->scale * TQ_LOG2E;
      const float next = fmaxf(m, score), alpha = den > 0 ? exp2f(m - next) : 0,
                  prob = exp2f(score - next);
      for (uint32_t d = 0; d < hd; ++d)
        o[d] = o[d] * alpha + prob * j->current_v[(size_t)h * hd + d];
      den = den * alpha + prob;
      m = next;
    }
    if (j->splits > 1) {
      o[hd] = m;
      o[hd + 1] = den;
    } else {
      for (uint32_t d = 0; d < hd; ++d)
        o[d] /= den;
    }
  }
#endif
}
static void attention_worker(uint32_t threads, uint32_t lane, void *data) {
  attention_job *j = data;
  for (uint32_t u = lane; u < j->units; u += threads)
    attention_unit(j, u);
}
#if defined(__hexagon__)
static void prefix_head_consume(void *data, uint32_t head, const float *output,
                                const float *stats) {
  attention_job *j = data;
  hvx_tq_cache *c = j->c;
  for (uint32_t g = 0; g < 2; ++g) {
    float *p = c->prefix_state + ((size_t)head * 2u + g) * (c->dim + 2u);
    for (uint32_t d = 0; d < c->dim; d += 32)
      *(HVX_UVector *)(p + d) = Q6_Vsf_vmpy_VsfVsf(
        *(const HVX_UVector *)(output + (size_t)g * c->dim + d),
        splat(stats[2 * g + 1]));
    p[c->dim] = stats[2 * g] * TQ_LOG2E;
    p[c->dim + 1] = stats[2 * g + 1];
  }
  attention_unit(j, head);
}
#endif
static int finish_decode_append(hvx_tq_cache *c, uint32_t layer, uint32_t from,
                                uint32_t window, int deferred) {
  if (!deferred)
    return AEE_SUCCESS;
  return hvx_tq_append_begin(c, layer, from, 1, window, c->decode_input,
                             c->decode_input + (size_t)c->heads * c->dim);
}
int hvx_tq_step(hvx_tq_cache *c, uint32_t layer, uint32_t from, uint32_t count,
                uint32_t window, float scale, const float *q, const float *k,
                const float *v, float *out) {
  if (!c || layer >= c->layers || !count || from > c->max_seq ||
      count > c->max_seq - from || !q || !out || !(scale > 0) ||
      !finite_row(&scale, 1) ||
      !finite_row(q, (size_t)count * c->heads * c->gqa * c->dim))
    return AEE_EINVALIDFORMAT;
  const int pending_rc = hvx_tq_append_wait(c);
  if (pending_rc)
    return pending_rc;
  if (count == 1 && k)
    release_prefill(c);
  if ((k == NULL) != (v == NULL))
    return AEE_EINVALIDFORMAT;
  const int deferred = count == 1 && k && from && c->prefix[layer];
  if (deferred) {
    if (from < c->prefix_end[layer])
      return AEE_EBADSTATE;
    if (!finite_row(k, (size_t)c->heads * c->dim) ||
        !finite_row(v, (size_t)c->heads * c->dim))
      return AEE_EINVALIDFORMAT;
    if (!c->decode_input) {
      c->decode_input = memalign(128, (size_t)c->heads * c->dim * 8u);
      if (!c->decode_input)
        return AEE_ENOMEMORY;
    }
    memcpy(c->decode_input, k, (size_t)c->heads * c->dim * 4u);
    memcpy(c->decode_input + (size_t)c->heads * c->dim, v,
           (size_t)c->heads * c->dim * 4u);
  } else if (k) {
    const int rc = hvx_tq_append(c, layer, from, count, window, k, v);
    if (rc)
      return rc;
  }
  if (c->lengths[layer] < from + (deferred ? 0u : count) ||
      window != c->windows[layer])
    return AEE_EBADSTATE;
  const uint32_t oldest = c->high_water[layer] > c->capacity[layer]
                            ? c->high_water[layer] - c->capacity[layer]
                            : 0;
  const uint32_t needed = window && from + 1 > window ? from + 1 - window : 0;
  if (needed < oldest && !prefix_covers(c, layer, needed, oldest))
    return AEE_EBADSTATE;
#if defined(__hexagon__)
  tq_q8_prefix *prefix = c->prefix[layer];
  int visible_prefix = 0;
  for (const tq_q8_prefix *p = prefix; p; p = p->next)
    if (p->from < from + count && p->from + p->count > needed) {
      visible_prefix = 1;
      break;
    }
  const int inline_prefix = count == 1 && c->heads == 8 && c->gqa == 2 &&
                            c->dim == 256 && prefix && !prefix->next &&
                            !prefix->from && from >= prefix->count &&
                            needed < prefix->count;
  if (visible_prefix) {
    int rc;
    if (inline_prefix) {
      rc = prefix_workspace(c, 1);
      if (!rc)
        rc = prefix_prepare(c, prefix);
    } else
      rc = prefix_native(c, layer, from, count, window, scale, q);
    if (rc)
      return rc;
  }
#endif
  attention_job j;
  memset(&j, 0, sizeof(j));
  j.c = c;
  j.layer = layer;
  j.from = from;
  j.count = count;
  j.window = window;
  j.scale = scale;
  j.q = q;
  if (deferred) {
    j.current_k = c->decode_input;
    j.current_v = c->decode_input + (size_t)c->heads * c->dim;
  }

#if defined(__hexagon__)
  if (visible_prefix)
    j.prefix_state = c->prefix_state;
#endif
  j.output = out;
  j.group = c->gqa >= 2 ? 2 : 1;
  j.groups = (c->gqa + j.group - 1) / j.group;
  const uint32_t workers = hvx_worker_pool_workers(c->pool) + 1u;
  j.splits = count == 1
               ? (workers + c->heads * j.groups - 1) / (c->heads * j.groups)
               : 1;
  if (!j.splits)
    j.splits = 1;
#if defined(__hexagon__)
  if (inline_prefix)
    j.splits = 1;
#endif
  j.units = count * c->heads * j.groups * j.splits;
  if (count == 1 && c->rotated_q) {
    free(c->rotated_q);
    c->rotated_q = NULL;
    c->rotated_capacity = 0;
  }
  if (j.splits > 1) {
    const uint64_t bytes = (uint64_t)j.units * j.group * (c->dim + 2u) * 4u;
    if (bytes > 0x7fffffffu)
      return AEE_ENOMEMORY;
    if (bytes > c->partial_capacity) {
      float *next = memalign(128, (size_t)bytes);
      if (!next)
        return AEE_ENOMEMORY;
      free(c->partials);
      c->partials = next;
      c->partial_capacity = (size_t)bytes;
    }
    j.partials = c->partials;
  }
#if defined(__hexagon__)
  if (inline_prefix) {
    int rc = hvx_attn_m1_forward_cached_hook(
      prefix->native, prefix->count - 1u, prefix->count - needed, scale, q,
      c->prefix_work, prefix_head_consume, &j);
    if (rc != AEE_EUNSUPPORTED)
      return rc ? rc : finish_decode_append(c, layer, from, window, deferred);
    rc = prefix_native(c, layer, from, count, window, scale, q);
    if (rc)
      return rc;
  }
#endif
  hvx_worker_pool_run(c->pool, attention_worker, &j, j.units);
  if (j.splits == 1)
    return finish_decode_append(c, layer, from, window, deferred);
  for (uint32_t qi = 0; qi < count; ++qi)
    for (uint32_t h = 0; h < c->heads; ++h)
      for (uint32_t g = 0; g < c->gqa; ++g) {
        const uint32_t chain = (qi * c->heads + h) * j.groups + g / j.group;
        float *o =
          out + ((size_t)qi * c->heads * c->gqa + h * c->gqa + g) * c->dim;
        float maximum = -INFINITY, den = 0;
        memset(o, 0, c->dim * 4u);
        for (uint32_t s = 0; s < j.splits; ++s) {
          const float *p =
            j.partials +
            ((size_t)(chain * j.splits + s) * j.group + g % j.group) *
              (c->dim + 2u);
          if (p[c->dim + 1] > 0 && p[c->dim] > maximum)
            maximum = p[c->dim];
        }
        for (uint32_t s = 0; s < j.splits; ++s) {
          const float *p =
            j.partials +
            ((size_t)(chain * j.splits + s) * j.group + g % j.group) *
              (c->dim + 2u);
          if (p[c->dim + 1] <= 0)
            continue;
          const float w = exp2f(p[c->dim] - maximum);
          den += w * p[c->dim + 1];
          for (uint32_t d = 0; d < c->dim; ++d)
            o[d] += w * p[d];
        }
        for (uint32_t d = 0; d < c->dim; ++d)
          o[d] /= den;
      }
  return finish_decode_append(c, layer, from, window, deferred);
}

#if defined(__hexagon__)
static HVX_Vector tq_value_row32(const uint8_t *p, uint32_t r) {
  HVX_Vector x = *(const HVX_UVector *)p;
  x = r < 2 ? Q6_Vb_vpacke_VhVh(x, x) : Q6_Vb_vpacko_VhVh(x, x);
  if (r & 1)
    x = Q6_Vuh_vlsr_VuhR(x, 4);
  x = Q6_V_vand_VV(x, Q6_V_vsplat_R(0x0f0f0f0f));
  return Q6_V_lo_W(Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(Q6_Wuh_vunpack_Vub(x))));
}
/** Decode one packed block into ready HMX tiles. No full-history FP16
 * materialization, inverse Hadamard, nested pool dispatch or DDR scratch. */
static void tq_hmx_tiles(void *data, uint32_t head, uint32_t row0,
                         uint32_t cols, uint8_t *kt, uint8_t *vt) {
  hvx_tq_cache *c = data;
  const uint64_t start = c->profile_us ? HAP_perf_get_time_us() : 0;
  const uint32_t hd = c->dim, layer = c->hmx_layer;
  const uint32_t length = c->lengths[layer];
  const uint32_t oldest = c->high_water[layer] > c->capacity[layer]
                            ? c->high_water[layer] - c->capacity[layer]
                            : 0;
  const HVX_Vector tab0 = *(const HVX_Vector *)c->lut_low,
                   tab1 = *(const HVX_Vector *)c->lut_high;
  const HVX_Vector mask = Q6_V_vsplat_R(0x0f0f0f0f);
  HVX_Vector maximum_bits = Q6_V_vzero();
  float tile[32][32] TQ_ALIGN;
  const float *rows[32];
  for (uint32_t r = 0; r < 32; ++r)
    rows[r] = tile[r];
  int32_t ri[32] TQ_ALIGN;
  for (uint32_t r = 0; r < 32; ++r)
    ri[r] = r;
  for (uint32_t col = 0; col < cols; ++col) {
    const uint32_t first = row0 + col * 32u;
    const uint8_t *p = page_at(c, layer, head, first);
    for (uint32_t d0 = 0; d0 < hd; d0 += 32) {
      HVX_Vector *out_k =
        (HVX_Vector *)(kt + ((size_t)col * (hd / 32) + d0 / 32) * 2048u);
      HVX_Vector *out_v =
        (HVX_Vector *)(vt + ((size_t)col * (hd / 32) + d0 / 32) * 2048u);
      if (!p || first >= length || first + 32u <= oldest) {
        memset(out_k, 0, 2048);
        memset(out_v, 0, 2048);
        continue;
      }
      const float *meta = page_scales(c, (uint8_t *)p);
      const HVX_Vector positions =
        Q6_Vw_vadd_VwVw(*(HVX_Vector *)ri, Q6_V_vsplat_R(first));
      const HVX_VectorPred visible = Q6_Q_and_QQ(
        Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R(length), positions),
        Q6_Q_not_Q(Q6_Q_vcmp_gt_VwVw(Q6_V_vsplat_R(oldest), positions)));
      for (uint32_t d = 0; d < 32; d += 2) {
        const HVX_Vector codes =
          *(const HVX_UVector *)(p + (size_t)(d0 + d) / 2 * 32u);
        HVX_Vector a =
          Q6_Vsf_vmpy_VsfVsf(lut32(Q6_V_vand_VV(codes, mask), tab0, tab1),
                             *(const HVX_Vector *)meta);
        HVX_Vector b = Q6_Vsf_vmpy_VsfVsf(
          lut32(Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(codes, 4), mask), tab0, tab1),
          *(const HVX_Vector *)meta);
        a = Q6_V_vmux_QVV(visible, a, Q6_V_vzero());
        b = Q6_V_vmux_QVV(visible, b, Q6_V_vzero());
        maximum_bits = Q6_Vw_vmax_VwVw(
          maximum_bits, Q6_V_vand_VV(a, Q6_V_vsplat_R(0x7fffffff)));
        maximum_bits = Q6_Vw_vmax_VwVw(
          maximum_bits, Q6_V_vand_VV(b, Q6_V_vsplat_R(0x7fffffff)));
        *(HVX_Vector *)tile[d] = a;
        *(HVX_Vector *)tile[d + 1] = b;
      }
      // K is already [dimension, position], so this is K^T's ready tile.
      hvx_tile_f16_rows_f32_gather_to_tile(out_k, rows, 1.0f);
      for (uint32_t r = 0; r < 32; ++r) {
        HVX_Vector values = Q6_V_vzero();
        if (first + r >= oldest && first + r < length) {
          values = Q6_Vsf_vadd_VsfVsf(
            Q6_Vsf_vmpy_VsfVsf(
              Q6_Vsf_equals_Vw(tq_value_row32(
                p + (size_t)hd * 16u + (size_t)(r / 4) * hd * 2u + d0 * 2u,
                r % 4u)),
              splat(meta[32 + r])),
            splat(meta[64 + r]));
        }
        *(HVX_Vector *)tile[r] = values;
        maximum_bits = Q6_Vw_vmax_VwVw(
          maximum_bits, Q6_V_vand_VV(values, Q6_V_vsplat_R(0x7fffffff)));
      }
      hvx_tile_f16_rows_f32_gather_to_tile(out_v, rows, 1.0f);
    }
  }
  for (uint32_t shift = 4; shift <= 64; shift *= 2)
    maximum_bits =
      Q6_Vw_vmax_VwVw(maximum_bits, Q6_V_vror_VR(maximum_bits, shift));
  if ((uint32_t)Q6_R_vextract_VR(maximum_bits, 0) > 0x477fe000u)
    c->hmx_bad_range = 1;
  if (c->profile_us)
    c->profile_us[3] += (uint32_t)(HAP_perf_get_time_us() - start);
}
#endif
typedef struct {
  const float *input;
  float *output;
  uint32_t rows, dim;
} rotate_job;
static void rotate_worker(uint32_t threads, uint32_t lane, void *data) {
  rotate_job *j = data;
  for (uint32_t row = lane; row < j->rows; row += threads) {
    float *out = j->output + (size_t)row * j->dim;
    memcpy(out, j->input + (size_t)row * j->dim, j->dim * 4u);
    nntr_tq_rotate(out, j->dim);
  }
}
int hvx_tq_prefill_hmx(hvx_tq_cache *c, uint8_t *vtcm, uint32_t top,
                       uint32_t rate, uint32_t layer, uint32_t from,
                       uint32_t count, uint32_t window, const float *q,
                       const float *k, const float *v, float *out,
                       uint32_t *profile_us) {
#if defined(__hexagon__)
  if (c && layer < c->layers && c->prefix[layer])
    return AEE_EUNSUPPORTED;
  if (!rate || !vtcm || count < 16 || (uint64_t)from + count > 65535u)
    return AEE_EUNSUPPORTED;
  if (!c || layer >= c->layers || !q || !out ||
      !finite_row(q, (size_t)count * c->heads * c->gqa * c->dim))
    return AEE_EINVALIDFORMAT;
  const uint64_t n_elements = (uint64_t)count * c->heads * c->gqa * c->dim;
  if (n_elements > 0x1fffffffu)
    return AEE_ENOMEMORY;
  const size_t elements = (size_t)n_elements;
  if (elements > c->rotated_capacity) {
    float *next = memalign(128, elements * 4u);
    if (!next)
      return AEE_ENOMEMORY;
    free(c->rotated_q);
    c->rotated_q = next;
    c->rotated_capacity = elements;
  }
  uint64_t tick = profile_us ? HAP_perf_get_time_us() : 0;
  if (profile_us)
    memset(profile_us, 0, 7 * sizeof(*profile_us));
  rotate_job rotation = {q, c->rotated_q, (uint32_t)(elements / c->dim),
                         c->dim};
  hvx_worker_pool_run(c->pool, rotate_worker, &rotation, rotation.rows);
  if (profile_us) {
    profile_us[0] = HAP_perf_get_time_us() - tick;
    tick = HAP_perf_get_time_us();
  }
  c->profile_us = profile_us;
  int rc = hvx_tq_append(c, layer, from, count, window, k, v);
  c->profile_us = NULL;
  if (profile_us) {
    profile_us[1] = HAP_perf_get_time_us() - tick;
    tick = HAP_perf_get_time_us();
  }
  if (rc)
    return rc;
  hexkl_attn_f16_shape shape = {
    count, from, from + count, c->heads * c->gqa, c->heads, c->dim, window, 0};
  hexkl_attn_f16_tiling tiling;
  hexkl_attn_f16_choose_tiling(&shape, &tiling);
  // Amortize packed tile restoration over more queries. The FP16 kernel
  // supports up to 256 grouped rows within the bounded VTCM workspace.
  tiling.br = 256u / c->gqa;
  if (tiling.br > count)
    tiling.br = count;
  if (from + count >= 128u)
    tiling.bc = 128u;
  if (hexkl_attn_f16_tiling_init(&shape, &tiling) != HEXKL_ATTN_OK)
    return AEE_EINVALIDFORMAT;
  hexkl_attn_f16_io io;
  memset(&io, 0, sizeof(io));
  io.q = c->rotated_q;
  io.q_stride = c->heads * c->gqa * c->dim;
  io.out = out;
  io.out_stride = io.q_stride;
  io.kv_tile_cols = (from + count + 31u) / 32u;
  io.load_tiles = tq_hmx_tiles;
  io.load_context = c;
  c->hmx_layer = layer;
  c->hmx_bad_range = 0;
  c->profile_us = profile_us;
  rc = hexkl_attn_f16_prefill(vtcm, top, rate, &shape, &tiling, &io, c->pool,
                              NULL);
  if (profile_us)
    profile_us[2] = HAP_perf_get_time_us() - tick;
  c->profile_us = NULL;
  if (rc == AEE_SUCCESS && c->hmx_bad_range)
    return hvx_tq_step(c, layer, from, count, window,
                       1.0f / sqrtf((float)c->dim), q, NULL, NULL, out);
  return rc;
#else
  (void)c;
  (void)vtcm;
  (void)top;
  (void)rate;
  (void)profile_us;
  (void)layer;
  (void)from;
  (void)count;
  (void)window;
  (void)q;
  (void)k;
  (void)v;
  (void)out;
  return AEE_EUNSUPPORTED;
#endif
}

#if defined(__hexagon__)
static float tq_half_float(uint16_t h) {
  const uint32_t e = (h >> 10) & 31u, m = h & 1023u,
                 sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t bits;
  if (!e) {
    float x = (float)m / 16777216.0f;
    memcpy(&bits, &x, 4);
    bits |= sign;
  } else
    bits = sign | ((e == 31u ? 255u : e + 112u) << 23) | (m << 13);
  float value;
  memcpy(&value, &bits, 4);
  return value;
}
#endif
int hvx_tq_check_tiles(hvx_tq_cache *c, uint32_t layer, uint32_t from,
                       uint32_t count) {
#if defined(__hexagon__)
  const int pending_rc = hvx_tq_append_wait(c);
  if (pending_rc)
    return pending_rc;
  if (!c || layer >= c->layers || !count || from > c->lengths[layer] ||
      count > c->lengths[layer] - from)
    return AEE_EINVALIDFORMAT;
  if (c->prefix[layer])
    return AEE_EUNSUPPORTED;
  const size_t bytes = (size_t)c->dim / 32 * 2048;
  uint8_t *kt = memalign(128, bytes), *vt = memalign(128, bytes);
  if (!kt || !vt) {
    free(kt);
    free(vt);
    return AEE_ENOMEMORY;
  }
  uint16_t rows[1024] TQ_ALIGN;
  int rc = AEE_SUCCESS;
  c->hmx_layer = layer;
  const uint32_t oldest = c->high_water[layer] > c->capacity[layer]
                            ? c->high_water[layer] - c->capacity[layer]
                            : 0;
  for (uint32_t h = 0; h < c->heads && !rc; ++h)
    for (uint32_t t = from / 32; t <= (from + count - 1) / 32 && !rc; ++t) {
      tq_hmx_tiles(c, h, t * 32, 1, kt, vt);
      const uint8_t *p = page_at(c, layer, h, t * 32);
      for (uint32_t d0 = 0; d0 < c->dim && !rc; d0 += 32)
        for (uint32_t plane = 0; plane < 2 && !rc; ++plane) {
          hvx_tile_f16_tile_to_rows(
            rows, 32,
            (const HVX_Vector *)((plane ? vt : kt) + (size_t)d0 / 32 * 2048));
          for (uint32_t r = 0; r < 32 && !rc; ++r)
            for (uint32_t d = 0; d < 32; ++d) {
              const uint32_t absolute = t * 32 + r, dim = d0 + d;
              float expected = 0;
              if (p && absolute >= oldest && absolute < c->lengths[layer]) {
                const float *meta = page_scales(c, (uint8_t *)p);
                if (!plane) {
                  const uint8_t code =
                    (p[(size_t)(dim / 2) * 32 + r] >> (4 * (dim % 2))) & 15;
                  expected = c->codec.centroids[code] * meta[r];
                } else {
                  const uint8_t *v =
                    p + (size_t)c->dim * 16 + (size_t)(r / 4) * c->dim * 2;
                  const uint8_t code =
                    (v[2 * dim + (r % 4) / 2] >> (4 * (r % 2))) & 15;
                  expected = code * meta[32 + r] + meta[64 + r];
                }
              }
              const float actual =
                tq_half_float(rows[plane ? r * 32 + d : d * 32 + r]);
              const float tolerance =
                0.001f * (fabsf(expected) > 1 ? fabsf(expected) : 1);
              if (!finite_row(&actual, 1) ||
                  fabsf(actual - expected) > tolerance) {
                rc = AEE_EFAILED;
                break;
              }
            }
        }
    }
  free(kt);
  free(vt);
  return rc;
#else
  (void)c;
  (void)layer;
  (void)from;
  (void)count;
  return AEE_EUNSUPPORTED;
#endif
}
