// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   nntr_hvx_attn_m1.c
 * @date   27 Sep 2026
 * @brief  DSP-side entries for decode attention at m=1 with the session's
 *         fp16 KV cache (hvx_attn_m1_f32.c): register / release / kv_append
 *         / forward
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Each entry validates the FastRPC lengths the kernel cannot see, then
 * calls the kernel on the caller's buffers and the session's cache. A bad
 * shape is AEE_EINVALIDFORMAT, a hole or a missing cache AEE_EBADSTATE;
 * AEE_EBADPARM stays the stale-skel symptom (rule 3), so the gtest can
 * tell the two apart. One FARF line per register / release, since the
 * cache is the session's largest heap object (plan 81 section 3.1).
 */

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <HAP_perf.h>
#include <math.h>
#include <remote.h>
#include <stdlib.h>
#include <string.h>

#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

#include "attn_m1_det.h"
#include "htp_graph_desc.h"
#include "hvx_attn_m1_f32.h"
#include "hvx_attn_softmax_f16.h"
#include "hvx_convert.h"
#include "hvx_tile_f16.h"

/* Q8 prefill masters become the persistent prefix without a second
 * quantization pass. Only decode appends use the packed four-bit format.
 * Reusable conversion/output scratch is bounded and freed at decode. */
typedef struct {
  const uint16_t *out16;
  float *out;
  const float *enc;
  uint32_t nq, heads, dim;
} tq_fresh_job;
static void tq_fresh_worker(uint32_t threads, uint32_t lane, void *data) {
  tq_fresh_job *j = data;
  for (uint32_t unit = lane; unit < j->nq / 64u; unit += threads) {
    const uint32_t off = unit * 64u, h = (off / j->dim) % j->heads;
    int32_t zp = (int32_t)j->enc[2 * h + 1];
    uint32_t bits;
    memcpy(&bits, j->enc + 2 * h, 4);
    const HVX_VectorPair w =
      Q6_Wuw_vunpack_Vuh(*(const HVX_Vector *)(j->out16 + off));
    const HVX_Vector a =
      Q6_Vsf_equals_Vw(Q6_Vw_vsub_VwVw(Q6_V_lo_W(w), Q6_V_vsplat_R(zp)));
    const HVX_Vector b =
      Q6_Vsf_equals_Vw(Q6_Vw_vsub_VwVw(Q6_V_hi_W(w), Q6_V_vsplat_R(zp)));
    *(HVX_UVector *)(j->out + off) =
      Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, Q6_V_vsplat_R(bits)));
    *(HVX_UVector *)(j->out + off + 32) =
      Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(b, Q6_V_vsplat_R(bits)));
  }
}
#define TQ_PREFILL_BUDGET (96u * 1024u * 1024u)
static void tq_workspace_free(nntr_hvx_session *s,
                              nntr_tq_prefill_workspace *w) {
  if (w->owner)
    (void)hvx_tq_append_wait(w->owner);
  if (w->has_handle)
    nntr_hvx_kv_release_q((remote_handle64)s, w->handle);
  free(w->out16);
  free(w->enc);
  memset(w, 0, sizeof(*w));
}
static void tq_workspace_release(void *ctx) {
  nntr_hvx_session *s = ctx;
  tq_workspace_free(s, &s->tq_prefill[0]);
  tq_workspace_free(s, &s->tq_prefill[1]);
}
static int tq_fresh_prefill(nntr_hvx_session *s, hvx_attn_m1_ctx *c,
                            uint32_t layer, uint32_t from, uint32_t count,
                            uint32_t window, const float *q, const float *k,
                            const float *v, float *out, uint32_t *profile,
                            uint32_t *workspace, float *out_enc, int *encoded) {
  const uint64_t nk = (uint64_t)count * c->n_kv * c->head_dim, nq = nk * c->gqa;
  const uint64_t padded = ((uint64_t)count + 31u) / 32u * 32u;
  const uint64_t budget =
    4u * (padded + 32u) * c->n_kv * c->head_dim + 2u * nq +
    12u * padded * c->n_kv +
    ((uint64_t)c->n_kv + c->n_kv * c->head_dim + 4u * c->n_kv * c->gqa) * 4u +
    4u * 1024u * 1024u;
  if (!s->hmx_locked || count > 65503u || budget > TQ_PREFILL_BUDGET)
    return AEE_EUNSUPPORTED;
  // All prefix ownership changes happen after resident work has joined.
  int prior = hvx_tq_append_wait(c->turboquant);
  if (prior)
    return prior;
  const uint32_t slot = c == s->attn_m1_b ? 1u : 0u;
  nntr_tq_prefill_workspace *w = &s->tq_prefill[slot],
                            *other = &s->tq_prefill[slot ^ 1u];
  int rc = AEE_ENOMEMORY;
  const uint32_t enc_count =
    c->n_kv + c->n_kv * c->head_dim + 4u * c->n_kv * c->gqa;
  if (w->active && (w->rows < count || w->n_kv != c->n_kv || w->gqa != c->gqa ||
                    w->dim != c->head_dim))
    tq_workspace_free(s, w);
  if (!w->active) {
    if (budget + other->budget > TQ_PREFILL_BUDGET)
      tq_workspace_free(s, other);
    w->out16 = memalign(128, (size_t)nq * 2);
    w->enc = malloc(enc_count * 4u);
    if (!w->out16 || !w->enc) {
      tq_workspace_free(s, w);
      return AEE_EUNSUPPORTED;
    }
    w->active = 1;
    w->owner = c->turboquant;
    w->rows = count;
    w->n_kv = c->n_kv;
    w->gqa = c->gqa;
    w->dim = c->head_dim;
    w->budget = (uint32_t)budget;
  }
  // One extra tile guarantees safe bounded vector reads of Q8 masters.
  const uint64_t register_tick = profile ? HAP_perf_get_time_us() : 0;
  rc = w->has_handle ? hexkl_kv_q_rearm_prefill(&s->kv_q, w->handle, count)
                     : hexkl_kv_q_register_prefill(&s->kv_q, count, c->n_kv,
                                                   c->head_dim, &w->handle);
  if (profile)
    profile[4] = HAP_perf_get_time_us() - register_tick;
  if (rc)
    return rc;
  w->has_handle = 1;
  hvx_tq_prefill_cleanup(c->turboquant, tq_workspace_release, s);
  uint16_t *out16 = out_enc ? (uint16_t *)out : w->out16;
  float *enc = w->enc;
  const uint32_t handle = w->handle;
  uint32_t stats[12] = {0};
  tq_fresh_job job = {0};
  job.nq = (uint32_t)nq;
  job.heads = c->n_kv * c->gqa;
  job.dim = c->head_dim;
  uint64_t tick = profile ? HAP_perf_get_time_us() : 0;
  rc =
    nntr_hvx_attn_q2_step_fqc_f32((remote_handle64)s, handle, count, job.heads,
                                  window, q, k, v, enc, out16, stats);
  if (profile) {
    profile[0] = stats[2];
    profile[2] = HAP_perf_get_time_us() - tick;
  }
  if (rc)
    goto done;
  if (out_enc) {
    memcpy(out_enc, enc + c->n_kv + c->n_kv * c->head_dim + 2u * job.heads,
           2u * job.heads * 4u);
    *encoded = 1;
  } else {
    job.out = out;
    job.out16 = out16;
    job.enc = enc + c->n_kv + c->n_kv * c->head_dim + 2u * job.heads;
    tick = profile ? HAP_perf_get_time_us() : 0;
    hvx_worker_pool_run(s->quant_pool, tq_fresh_worker, &job,
                        (uint32_t)nq / 64u);
    if (profile)
      profile[3] = HAP_perf_get_time_us() - tick;
  }
  tick = profile ? HAP_perf_get_time_us() : 0;
  hexkl_kv_q *kv = &s->kv_q.slots[handle];
  rc = hvx_tq_adopt_q8_prefix(c->turboquant, layer, from, count, kv->max_rows,
                              window, kv->kt4, kv->v4, kv->fs_k, kv->fs_v);
  if (rc)
    goto done;
  kv->kt4 = kv->v4 = NULL;
  kv->fs_k = kv->fs_v = NULL;
  // Keep the WH computation tiles for the next layer of this shape.
  // The adopted masters/scales are independent persistent allocations.
  // Continuation chunks retain their own Q8 calibration. Recompute their
  // output over the complete mixed prefix, with one causal softmax.
  if (from)
    rc = hvx_tq_step(c->turboquant, layer, from, count, window,
                     1.0f / sqrtf((float)c->head_dim), q, NULL, NULL, out);
  if (profile)
    profile[1] = HAP_perf_get_time_us() - tick;
  if (workspace)
    *workspace = s->tq_prefill[0].budget + s->tq_prefill[1].budget;
done:
  if (rc)
    tq_workspace_free(s, w);
  return rc == AEE_ENOMEMORY ? AEE_EUNSUPPORTED : rc;
}

int nntr_hvx_attn_tq_step(remote_handle64 handle, uint32 layer, uint32 from,
                          uint32 count, uint32 window, float scale,
                          const float *q, int qLen, const float *k, int kLen,
                          const float *v, int vLen, float *out, int outLen,
                          uint32 *stats, int statsLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s)
    return AEE_EBADPARM;
  hvx_attn_m1_ctx *c = layer & HTP_ATTN_KV_CACHE_B ? s->attn_m1_b : s->attn_m1;
  const int joined = (layer & HTP_ATTN_TQ_JOINED_QKV) != 0,
            shared_v = (layer & HTP_ATTN_TQ_SHARED_V) != 0;
  const int fresh_q8 = (layer & HTP_ATTN_TQ_PREFILL_Q8) != 0;
  const int profile_requested = (layer & HTP_ATTN_TQ_PROFILE) != 0;
  const int want_u16 = (layer & HTP_ATTN_TQ_OUT_U16) != 0;
  layer &=
    ~(HTP_ATTN_KV_CACHE_B | HTP_ATTN_TQ_PREFILL_Q8 | HTP_ATTN_TQ_OUT_U16 |
      HTP_ATTN_TQ_PROFILE | HTP_ATTN_TQ_JOINED_QKV | HTP_ATTN_TQ_SHARED_V);
  if (!c || !c->turboquant)
    return AEE_EBADSTATE;
  const uint64_t nk = (uint64_t)count * c->n_kv * c->head_dim, nq = nk * c->gqa;
  if (shared_v && !joined)
    return AEE_EINVALIDFORMAT;
  if (joined) {
    if (!q || qLen < 0 || (uint64_t)qLen != nq + (shared_v ? nk : 2u * nk) ||
        kLen || vLen)
      return AEE_EINVALIDFORMAT;
    k = q + nq;
    v = shared_v ? k : k + nk;
    qLen = (int)nq;
    kLen = vLen = (int)nk;
  }
  if (layer >= c->n_layers || from > c->max_seq || count > c->max_seq - from ||
      !count || !q || !k || !v || !out || qLen < 0 || kLen < 0 || vLen < 0 ||
      outLen < 0 || (uint64_t)qLen != nq ||
      (uint64_t)outLen != (want_u16 ? nq / 2u : nq) || (uint64_t)kLen != nk ||
      (uint64_t)vLen != nk || !stats || statsLen < 2)
    return AEE_EINVALIDFORMAT;
  const uint64_t start = HAP_perf_get_time_us();
  int rc = AEE_EUNSUPPORTED;
  uint32_t profile[7] = {0}, workspace = 0;
  const int profiling = statsLen >= 6 && (!want_u16 || profile_requested);
  uint32_t path = 0;
  int encoded = 0;
  if (want_u16 && from)
    return AEE_EUNSUPPORTED;
  if (want_u16 && statsLen < (int)(12u + 2u * c->n_kv * c->gqa))
    return AEE_EINVALIDFORMAT;
  if (fresh_q8 && fabsf(scale - 1.0f / sqrtf((float)c->head_dim)) < 1e-7f) {
    rc = tq_fresh_prefill(s, c, layer, from, count, window, q, k, v, out,
                          profiling ? profile : NULL, &workspace,
                          want_u16 ? (float *)(stats + 12) : NULL, &encoded);
    if (rc == AEE_SUCCESS)
      path = 1;
  }
  if (fresh_q8 && rc != AEE_SUCCESS)
    return rc;
  // An encoded result has half the FP32 byte length. Never let a failed
  // fast-path attempt write FP32 into that buffer; the host retries with
  // the ordinary output contract on the same authoritative packed cache.
  if (want_u16 && rc != AEE_SUCCESS)
    return rc;
  if (rc == AEE_EUNSUPPORTED &&
      fabsf(scale - 1.0f / sqrtf((float)c->head_dim)) < 1e-7f)
    rc = hvx_tq_prefill_hmx(c->turboquant, s->vtcm_base, s->config_off,
                            s->hmx_fp16_rate, layer, from, count, window, q, k,
                            v, out, profiling ? profile : NULL);
  if (rc == AEE_EUNSUPPORTED)
    rc = hvx_tq_step(c->turboquant, layer, from, count, window, scale, q, k, v,
                     out);
  stats[0] = (uint32_t)(HAP_perf_get_time_us() - start);
  stats[1] = (uint32_t)hvx_tq_payload_bytes(c->turboquant);
  if (statsLen >= 6)
    for (int i = 0; i < 7 && i + 2 < statsLen; ++i)
      stats[i + 2] = profile[i];
  if (statsLen >= 10)
    stats[9] = path;
  if (statsLen >= 11)
    stats[10] = s->tq_prefill[0].budget + s->tq_prefill[1].budget;
  if (statsLen >= 12)
    stats[11] = (uint32_t)encoded;
  if (statsLen == 3)
    stats[2] = (uint32_t)hvx_tq_check_tiles(c->turboquant, layer, from, count);
  return rc;
}

int nntr_hvx_attn_m1_register(remote_handle64 handle, uint32 n_layers,
                              uint32 n_kv, uint32 gqa, uint32 head_dim,
                              uint32 max_seq) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  /** [plan 201 S4] a second register of another shape is the second cache
     (Gemma 4's full layers beside its sliding ones); a third, or a second
     of the first one's shape, is refused as before */
  hvx_attn_m1_ctx **slot = &s->attn_m1;
  if (s->attn_m1) {
    if (s->attn_m1_b || (s->attn_m1->n_kv == n_kv && s->attn_m1->gqa == gqa &&
                         s->attn_m1->head_dim == head_dim)) {
      FARF(ERROR, "attn_m1_register: a cache is already registered");
      return AEE_EBADSTATE;
    }
    slot = &s->attn_m1_b;
  }
  int err = AEE_SUCCESS;
  *slot = hvx_attn_m1_create(n_layers, n_kv, gqa, head_dim, max_seq,
                             s->quant_pool, &err);
  if (!*slot) {
    FARF(ERROR,
         "attn_m1_register: create failed 0x%08x (layers=%u kv=%u gqa=%u "
         "head_dim=%u max_seq=%u)",
         (unsigned)err, (unsigned)n_layers, (unsigned)n_kv, (unsigned)gqa,
         (unsigned)head_dim, (unsigned)max_seq);
    return err;
  }
  FARF(HIGH,
       "attn_m1_register: layers=%u kv=%u gqa=%u head_dim=%u max_seq=%u "
       "cache=%u KiB",
       (unsigned)n_layers, (unsigned)n_kv, (unsigned)gqa, (unsigned)head_dim,
       (unsigned)max_seq,
       (unsigned)((*slot)->cache_halves * 2u * sizeof(uint16_t) / 1024u));
  return AEE_SUCCESS;
}

int nntr_hvx_attn_m1_reserve(remote_handle64 handle, uint32 cache, uint32 rows,
                             uint32 drop, uint32 base) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s || cache > 1u)
    return AEE_EBADPARM;
  hvx_attn_m1_ctx **slot = cache ? &s->attn_m1_b : &s->attn_m1;
  hvx_attn_m1_ctx *old = *slot;
  if (!old || !s->graph)
    return AEE_EBADSTATE;
  for (uint32_t i = 0; i < s->graph->n_ops; ++i) {
    const htp_graph_op *op = &s->graph->ops[i];
    if (op->kind == HTP_OP_ATTN_M1 && op->n_kv == old->n_kv &&
        op->gqa == old->gqa && op->head_dim == old->head_dim &&
        ((!op->top_k && (drop || base)) || base > s->graph->max_seq ||
         (drop &&
          (op->n_experts > UINT32_MAX - drop || base != op->n_experts + drop))))
      return AEE_EINVALIDFORMAT;
  }
  int err = AEE_SUCCESS;
  hvx_attn_m1_ctx *next = rows == old->max_seq && !drop
                            ? old
                            : hvx_attn_m1_reserve(old, rows, drop, &err);
  if (!next)
    return err;
  *slot = next;
  for (uint32_t i = 0; i < s->graph->n_ops; ++i) {
    htp_graph_op *op = &s->graph->ops[i];
    if (op->kind == HTP_OP_ATTN_M1 && op->n_kv == old->n_kv &&
        op->gqa == old->gqa && op->head_dim == old->head_dim)
      op->n_experts = base;
  }
  if (old != next)
    hvx_attn_m1_free(old);
  return AEE_SUCCESS;
}

static void snapshot_abort(nntr_hvx_session *s) {
  for (uint32_t c = 0; c < 2u; ++c) {
    hvx_attn_m1_free(s->kv_restore[c]);
    s->kv_restore[c] = NULL;
    s->kv_restore_next[c] = 0;
  }
  s->kv_restore_mask = 0;
}

int nntr_hvx_attn_m1_snapshot_control(remote_handle64 handle, uint32 action,
                                      const uint32 *metadata, int metadataLen,
                                      uint32 *result, int resultLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s || !s->graph || (!s->attn_m1 && action < 5u))
    return AEE_EBADSTATE;
  hvx_attn_m1_ctx *active[2] = {s->attn_m1, s->attn_m1_b};
  // 5 retains immutable page tables, 6 restores by sharing, 7 releases a
  // handle. Both shapes are cloned before publication; an allocation failure
  // leaves active and saved conversations untouched. No KV bytes cross FastRPC.
  if (action >= 5u && action <= 7u) {
    if (!metadata ||
        (metadataLen != 1 && !(action == 6u && metadataLen == 2)) ||
        !metadata[0] || s->kv_restore_mask)
      return AEE_EINVALIDFORMAT;
    nntr_kv_saved **link = &s->kv_saved;
    uint32_t count = 0;
    while (*link && (*link)->id != metadata[0]) {
      link = &(*link)->next;
      ++count;
    }
    nntr_kv_saved *saved = *link;
    if (action == 7u) {
      if (!saved)
        return AEE_EBADSTATE;
      *link = saved->next;
      hvx_attn_m1_free(saved->cache[0]);
      hvx_attn_m1_free(saved->cache[1]);
      free(saved);
      return AEE_SUCCESS;
    }
    if ((action == 6u && !saved) || (action == 5u && (!active[0] || saved)))
      return AEE_EBADSTATE;
    if (action == 5u && count >= 128u)
      return AEE_ENOMEMORY;
    const uint32_t prefix = action == 6u && metadataLen == 2 ? metadata[1] : 0;
    if (prefix) {
      for (uint32_t c = 0; c < 2u; ++c) {
        const hvx_attn_m1_ctx *a = saved->cache[c];
        if (!a)
          continue;
        if (prefix < saved->base[c])
          return AEE_EINVALIDFORMAT;
        const uint32_t live = prefix - saved->base[c];
        for (uint32_t l = 0; l < a->n_layers; ++l)
          if (live > a->kv_len[l])
            return AEE_EINVALIDFORMAT;
        for (uint32_t i = 0; i < s->graph->n_ops; ++i) {
          const htp_graph_op *op = &s->graph->ops[i];
          if (op->kind == HTP_OP_ATTN_M1 && op->n_kv == a->n_kv &&
              op->gqa == a->gqa && op->head_dim == a->head_dim &&
              saved->base[c] && (!op->top_k || live < op->top_k))
            return AEE_EINVALIDFORMAT;
        }
      }
    }
    nntr_kv_saved *next = calloc(1, sizeof(*next));
    if (!next)
      return AEE_ENOMEMORY;
    next->id = metadata[0];
    int rc = AEE_SUCCESS;
    for (uint32_t c = 0; c < 2u; ++c) {
      const hvx_attn_m1_ctx *src = action == 5u ? active[c] : saved->cache[c];
      if (!src)
        continue;
      next->cache[c] = hvx_attn_m1_clone(src, &rc);
      if (!next->cache[c])
        break;
      next->base[c] = action == 6u ? saved->base[c] : 0;
      if (prefix)
        for (uint32_t l = 0; l < next->cache[c]->n_layers; ++l) {
          rc = hvx_attn_m1_truncate(next->cache[c], l, prefix - next->base[c]);
          if (rc != AEE_SUCCESS)
            break;
        }
      if (rc != AEE_SUCCESS)
        break;
      if (action == 5u)
        for (uint32_t i = 0; i < s->graph->n_ops; ++i) {
          const htp_graph_op *op = &s->graph->ops[i];
          if (op->kind == HTP_OP_ATTN_M1 && op->n_kv == src->n_kv &&
              op->gqa == src->gqa && op->head_dim == src->head_dim)
            next->base[c] = op->n_experts;
        }
    }
    if (rc != AEE_SUCCESS) {
      hvx_attn_m1_free(next->cache[0]);
      hvx_attn_m1_free(next->cache[1]);
      free(next);
      return rc;
    }
    if (action == 5u) {
      *link = next;
    } else {
      for (uint32_t c = 0; c < 2u; ++c) {
        hvx_attn_m1_ctx *a = next->cache[c];
        if (a)
          for (uint32_t i = 0; i < s->graph->n_ops; ++i) {
            htp_graph_op *op = &s->graph->ops[i];
            if (op->kind == HTP_OP_ATTN_M1 && op->n_kv == a->n_kv &&
                op->gqa == a->gqa && op->head_dim == a->head_dim)
              op->n_experts = next->base[c];
          }
        hvx_attn_m1_free(active[c]);
      }
      s->attn_m1 = next->cache[0];
      s->attn_m1_b = next->cache[1];
      free(next);
    }
    return AEE_SUCCESS;
  }
  if (action == 8u) {
    if (!result || resultLen != 1)
      return AEE_EINVALIDFORMAT;
    result[0] = (uint32_t)(hvx_attn_m1_payload_bytes(active[0]) +
                           hvx_attn_m1_payload_bytes(active[1]));
    return AEE_SUCCESS;
  }
  if (action == 4u) {
    snapshot_abort(s);
    return AEE_SUCCESS;
  }
  if (action == 0u) {
    if (!result || resultLen != 17)
      return AEE_EINVALIDFORMAT;
    memset(result, 0, 17u * sizeof(uint32_t));
    for (uint32_t c = 0; c < 2u; ++c) {
      const hvx_attn_m1_ctx *a = active[c];
      if (!a)
        continue;
      if (!a->q8)
        return AEE_EBADSTATE;
      result[0] |= 1u << c;
      uint32_t *r = result + 1u + 8u * c;
      r[0] = a->n_layers;
      r[1] = a->n_kv;
      r[2] = a->gqa;
      r[3] = a->head_dim;
      r[4] = a->max_seq;
      r[5] = HVX_ATTN_M1_Q8 | (a->fast ? HVX_ATTN_M1_FAST : 0u) |
             (a->qk_i8 ? HVX_ATTN_M1_QK_I8 : 0u) |
             (a->paged ? HVX_ATTN_M1_PAGED : 0u);
      for (uint32_t i = 0; i < s->graph->n_ops; ++i) {
        const htp_graph_op *op = &s->graph->ops[i];
        if (op->kind == HTP_OP_ATTN_M1 && op->n_kv == a->n_kv &&
            op->gqa == a->gqa && op->head_dim == a->head_dim)
          r[6] = op->n_experts;
      }
    }
    return AEE_SUCCESS;
  }
  if (action == 1u) {
    if (s->kv_restore_mask)
      return AEE_EBADSTATE;
    hvx_attn_m1_park(active[0]);
    hvx_attn_m1_park(active[1]);
    return AEE_SUCCESS;
  }
  if (action == 2u) {
    snapshot_abort(s);
    if (!metadata || metadataLen != 17 || metadata[0] != (active[1] ? 3u : 1u))
      return AEE_EINVALIDFORMAT;
    int rc = AEE_SUCCESS;
    for (uint32_t c = 0; c < 2u; ++c) {
      if (!active[c])
        continue;
      const uint32_t *r = metadata + 1u + 8u * c;
      const hvx_attn_m1_ctx *a = active[c];
      const uint32_t flags = HVX_ATTN_M1_Q8 |
                             (a->fast ? HVX_ATTN_M1_FAST : 0u) |
                             (a->qk_i8 ? HVX_ATTN_M1_QK_I8 : 0u) |
                             (a->paged ? HVX_ATTN_M1_PAGED : 0u);
      if (r[0] != a->n_layers || r[1] != a->n_kv || r[2] != a->gqa ||
          r[3] != a->head_dim || r[5] != flags || r[7] || !r[4] ||
          r[4] > s->graph->max_seq || r[4] % 64u || r[6] > s->graph->max_seq) {
        rc = AEE_EINVALIDFORMAT;
        break;
      }
      s->kv_restore[c] =
        hvx_attn_m1_create(r[0], r[1], r[2], r[3],
                           r[4] | flags | HVX_ATTN_M1_LAZY, s->quant_pool, &rc);
      if (!s->kv_restore[c])
        break;
      s->kv_restore_base[c] = r[6];
    }
    if (rc != AEE_SUCCESS) {
      snapshot_abort(s);
      return rc;
    }
    s->kv_restore_mask = metadata[0];
    return AEE_SUCCESS;
  }
  if (action == 3u) {
    if (!s->kv_restore_mask)
      return AEE_EBADSTATE;
    for (uint32_t c = 0; c < 2u; ++c)
      if (s->kv_restore[c] &&
          s->kv_restore_next[c] != s->kv_restore[c]->n_layers)
        return AEE_EBADSTATE;
    // No failure points after validation: both shapes become visible together.
    for (uint32_t c = 0; c < 2u; ++c) {
      const hvx_attn_m1_ctx *a = active[c];
      if (!a)
        continue;
      for (uint32_t i = 0; i < s->graph->n_ops; ++i) {
        htp_graph_op *op = &s->graph->ops[i];
        if (op->kind == HTP_OP_ATTN_M1 && op->n_kv == a->n_kv &&
            op->gqa == a->gqa && op->head_dim == a->head_dim)
          op->n_experts = s->kv_restore_base[c];
      }
      if (c)
        s->attn_m1_b = s->kv_restore[c];
      else
        s->attn_m1 = s->kv_restore[c];
      s->kv_restore[c] = NULL;
      hvx_attn_m1_free(active[c]);
    }
    snapshot_abort(s);
    return AEE_SUCCESS;
  }
  return AEE_EINVALIDFORMAT;
}

int nntr_hvx_attn_m1_snapshot_layer(remote_handle64 handle, uint32 action,
                                    uint32 cache, uint32 layer,
                                    const uint8 *input, int inputLen,
                                    uint8 *output, int outputLen,
                                    uint32 *bytes) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s || cache > 1u || !bytes || inputLen < 0 || outputLen < 0)
    return AEE_EINVALIDFORMAT;
  hvx_attn_m1_ctx *a =
    action == 2u ? s->kv_restore[cache] : (cache ? s->attn_m1_b : s->attn_m1);
  if (!a || layer >= a->n_layers)
    return AEE_EBADSTATE;
  if (action == 2u) {
    if (!s->kv_restore_mask || layer != s->kv_restore_next[cache])
      return AEE_EBADSTATE;
    const int rc =
      hvx_attn_m1_snapshot_import(a, layer, input, (size_t)inputLen);
    if (rc == AEE_SUCCESS)
      ++s->kv_restore_next[cache];
    *bytes = (uint32)inputLen;
    return rc;
  }
  *bytes = (uint32)hvx_attn_m1_snapshot_bytes(a, layer);
  if (action == 0u)
    return *bytes ? AEE_SUCCESS : AEE_EBADSTATE;
  if (action != 1u)
    return AEE_EINVALIDFORMAT;
  return hvx_attn_m1_snapshot_export(a, layer, output, (size_t)outputLen);
}

int nntr_hvx_attn_m1_release(remote_handle64 handle) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  if (!s->attn_m1) {
    return AEE_EBADSTATE;
  }
  snapshot_abort(s);
  hvx_attn_m1_free(s->attn_m1);
  s->attn_m1 = NULL;
  hvx_attn_m1_free(s->attn_m1_b);
  s->attn_m1_b = NULL;
  FARF(HIGH, "attn_m1_release: cache freed");
  return AEE_SUCCESS;
}

int nntr_hvx_attn_m1_kv_append(remote_handle64 handle, uint32 layer,
                               uint32 kv_from, uint32 n_rows,
                               const float *k_rows, int kLen,
                               const float *v_rows, int vLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  /* [plan 201 S4] HTP_ATTN_KV_CACHE_B in layer names the second cache */
  hvx_attn_m1_ctx *c =
    (layer & HTP_ATTN_KV_CACHE_B) != 0u ? s->attn_m1_b : s->attn_m1;
  layer &= ~HTP_ATTN_KV_CACHE_B;
  if (!c) {
    return AEE_EBADSTATE;
  }
  const uint64_t want = (uint64_t)n_rows * c->n_kv * c->head_dim;
  if (kLen < 0 || vLen < 0 || (uint64_t)kLen != want ||
      (uint64_t)vLen != want) {
    FARF(ERROR, "attn_m1_kv_append: bad shape (n_rows=%u k=%d v=%d want=%u)",
         (unsigned)n_rows, kLen, vLen, (unsigned)want);
    return AEE_EINVALIDFORMAT;
  }
  return hvx_attn_m1_kv_append(c, layer, kv_from, n_rows, k_rows, v_rows);
}

int nntr_hvx_attn_m1_kv_append_f16(remote_handle64 handle, uint32 layer,
                                   uint32 kv_from, uint32 n_rows,
                                   const uint16 *k_rows, int kLen,
                                   const uint16 *v_rows, int vLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  hvx_attn_m1_ctx *c =
    (layer & HTP_ATTN_KV_CACHE_B) != 0u ? s->attn_m1_b : s->attn_m1;
  layer &= ~HTP_ATTN_KV_CACHE_B;
  if (!c) {
    return AEE_EBADSTATE;
  }
  const uint64_t want = (uint64_t)n_rows * c->n_kv * c->head_dim;
  if (kLen < 0 || vLen < 0 || (uint64_t)kLen != want ||
      (uint64_t)vLen != want) {
    FARF(ERROR,
         "attn_m1_kv_append_f16: bad shape (n_rows=%u k=%d v=%d want=%u)",
         (unsigned)n_rows, kLen, vLen, (unsigned)want);
    return AEE_EINVALIDFORMAT;
  }
  return hvx_attn_m1_kv_append_f16(c, layer, kv_from, n_rows, k_rows, v_rows);
}

int nntr_hvx_attn_m1_kv_amax_floor(remote_handle64 handle, uint32 layer,
                                   const float *k_amax, int kLen,
                                   const float *v_amax, int vLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  hvx_attn_m1_ctx *c =
    (layer & HTP_ATTN_KV_CACHE_B) != 0u ? s->attn_m1_b : s->attn_m1;
  layer &= ~HTP_ATTN_KV_CACHE_B;
  if (!c) {
    return AEE_EBADSTATE;
  }
  const uint64_t want = (uint64_t)c->n_kv * c->head_dim;
  if (kLen < 0 || vLen < 0 || (uint64_t)kLen != want ||
      (uint64_t)vLen != want) {
    FARF(ERROR, "attn_m1_kv_amax_floor: bad shape (k=%d v=%d want=%u)", kLen,
         vLen, (unsigned)want);
    return AEE_EINVALIDFORMAT;
  }
  return hvx_attn_m1_kv_amax_floor(c, layer, k_amax, v_amax);
}

int nntr_hvx_attn_m1_seed_kvq(remote_handle64 handle, uint32 layer,
                              uint32 kv_handle, uint32 n_rows,
                              const float *k_rows, int k_rowsLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  hvx_attn_m1_ctx *c =
    (layer & HTP_ATTN_KV_CACHE_B) != 0u ? s->attn_m1_b : s->attn_m1;
  layer &= ~HTP_ATTN_KV_CACHE_B;
  const hexkl_kv_q *kv = hexkl_kv_q_get(&s->kv_q, kv_handle);
  if (!c || !kv || kv->kind != HEXKL_KV_Q8 || !kv->fixed ||
      !kv->plain_masters || kv->n_head_kv != c->n_kv ||
      kv->head_dim != c->head_dim || n_rows > kv->max_rows ||
      (k_rowsLen != 0 &&
       (uint64_t)k_rowsLen != (uint64_t)n_rows * c->n_kv * c->head_dim)) {
    FARF(ERROR, "attn_m1_seed_kvq: cache %u does not fit layer %u",
         (unsigned)kv_handle, (unsigned)layer);
    return AEE_EBADSTATE;
  }
  const int rc = hvx_attn_m1_seed_q8(
    c, layer, n_rows, kv->kt4, kv->v4, (size_t)kv->max_rows * kv->head_dim,
    kv->head_dim, kv->fs_k, kv->fs_v, kv->fs_v_inv,
    k_rowsLen != 0 ? k_rows : NULL);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  return hexkl_kv_q_release(&s->kv_q, kv_handle);
}

int nntr_hvx_attn_m1_forward(remote_handle64 handle, uint32 layer, uint32 pos,
                             float scale, const float *q, int qLen,
                             const float *k, int kLen, const float *v, int vLen,
                             float *y, int yLen, float *stats, int statsLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  const hvx_attn_m1_ctx *c = s->attn_m1;
  if (!c) {
    return AEE_EBADSTATE;
  }
  const uint32_t n_q = c->n_kv * c->gqa;
  if (qLen < 0 || kLen < 0 || vLen < 0 || yLen < 0 || statsLen < 0 ||
      (uint32_t)qLen != n_q * c->head_dim ||
      (uint32_t)kLen != c->n_kv * c->head_dim || vLen != kLen || yLen != qLen ||
      (statsLen != 0 && (uint32_t)statsLen != 2u * n_q &&
       (uint32_t)statsLen != 2u * n_q + ATTN_M1_PROF_WORDS)) {
    FARF(ERROR,
         "attn_m1_forward: bad shape (q=%d k=%d v=%d y=%d stats=%d; "
         "n_q=%u head_dim=%u)",
         qLen, kLen, vLen, yLen, statsLen, (unsigned)n_q,
         (unsigned)c->head_dim);
    return AEE_EINVALIDFORMAT;
  }
  if ((uint32_t)statsLen != 2u * n_q + ATTN_M1_PROF_WORDS) {
    return hvx_attn_m1_forward(s->attn_m1, layer, pos, 0u, scale, q, k, v, y,
                               statsLen ? stats : NULL);
  }
  /* The phase words (#146) follow the (m, l) pairs, bit-copied. */
  uint32_t prof[ATTN_M1_PROF_WORDS];
  const int rc = hvx_attn_m1_forward_prof(s->attn_m1, layer, pos, 0u, scale, q,
                                          k, v, y, stats, prof);
  if (rc == AEE_SUCCESS) {
    memcpy(stats + 2u * n_q, prof, sizeof(prof));
  }
  return rc;
}
