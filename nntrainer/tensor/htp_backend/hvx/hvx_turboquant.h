// SPDX-License-Identifier: Apache-2.0
/** @file hvx_turboquant.h
 * @brief DSP-owned Q8 prefill prefixes and packed MoE decode KV pages.
 * @author NNTrainer contributors
 * @bug Native snapshot/session switching is not yet supported.
 */
#ifndef NNTR_HVX_TURBOQUANT_H
#define NNTR_HVX_TURBOQUANT_H
#include "hvx_worker_pool.h"
#include <stddef.h>
#include <stdint.h>
typedef struct hvx_tq_cache hvx_tq_cache;
hvx_tq_cache *hvx_tq_create(uint32_t layers, uint32_t heads, uint32_t gqa,
                            uint32_t dim, uint32_t max_seq,
                            hvx_worker_pool *pool, int *error);
void hvx_tq_free(hvx_tq_cache *cache);
/** Release reusable prefill scratch at decode/reset. Caller context must
 * outlive the cache; callback must be idempotent and never reenter it. */
void hvx_tq_prefill_cleanup(hvx_tq_cache *cache, void (*cleanup)(void *),
                            void *ctx);
void hvx_tq_reset(hvx_tq_cache *cache);
uint32_t *hvx_tq_lengths(hvx_tq_cache *cache);
size_t hvx_tq_payload_bytes(const hvx_tq_cache *cache);
/** Take ownership of fixed-scale offset-binary row-major Q8 masters.
 * Scales are K [heads] and V [heads][dim]. On failure ownership stays
 * with the caller. Prefill chunks must append at the current length; TQ rows
 * may precede them. All four allocations must be distinct and
 * free()-compatible. stride_rows includes at least 32 guard rows for bounded
 * HVX loads. */
int hvx_tq_adopt_q8_prefix(hvx_tq_cache *cache, uint32_t layer, uint32_t from,
                           uint32_t count, uint32_t stride_rows,
                           uint32_t window, uint8_t *keys, uint8_t *values,
                           float *key_scales, float *value_scales);
int hvx_tq_append(hvx_tq_cache *cache, uint32_t layer, uint32_t from,
                  uint32_t count, uint32_t window, const float *keys,
                  const float *values);
/** Begin a packed append in the pool's background lane. Input buffers
 * remain owned by the caller and valid until append_wait; join before
 * reading the cache or releasing caller storage. Deferred RPC work must
 * use session-owned inputs. No persistent duplicate history. */
int hvx_tq_append_begin(hvx_tq_cache *cache, uint32_t layer, uint32_t from,
                        uint32_t count, uint32_t window, const float *keys,
                        const float *values);
int hvx_tq_append_wait(hvx_tq_cache *cache);
/** Q/K have already passed RoPE. Output is synchronous FP32 and can feed
 * resident o-projection without returning to ARM. Prefix-backed one-token
 * decode attends to the current original KV while its owned input is packed
 * afterward in the background. All cache consumers join before reading it.
 * Null keys/values reads the already appended rows. */
int hvx_tq_step(hvx_tq_cache *cache, uint32_t layer, uint32_t from,
                uint32_t count, uint32_t window, float scale,
                const float *queries, const float *keys, const float *values,
                float *output);
/** HMX prefill over packed pages, with only bounded KV tiles in VTCM.
 * Returns AEE_EUNSUPPORTED on devices without FP16 HMX. profile_us is
 * NULL or seven words: rotate, append, attention, tiles, and summed worker
 * key/value/store microseconds. */
int hvx_tq_prefill_hmx(hvx_tq_cache *cache, uint8_t *vtcm, uint32_t vtcm_top,
                       uint32_t fp16_rate, uint32_t layer, uint32_t from,
                       uint32_t count, uint32_t window, const float *q,
                       const float *k, const float *v, float *out,
                       uint32_t *profile_us);
/** Diagnostic: decode and round-trip bounded HMX tiles without issuing HMX. */
int hvx_tq_check_tiles(hvx_tq_cache *cache, uint32_t layer, uint32_t from,
                       uint32_t count);
#endif
