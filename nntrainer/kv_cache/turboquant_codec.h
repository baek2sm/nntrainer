// SPDX-License-Identifier: Apache-2.0
/** @file turboquant_codec.h
 * @brief Allocation-free TurboQuant codec shared with the DSP.
 * @author NNTrainer contributors
 * @bug None known.
 */
#ifndef NNTR_TURBOQUANT_CODEC_H
#define NNTR_TURBOQUANT_CODEC_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  uint32_t dim;
  float centroids[16], boundaries[15];
} nntr_tq_codec;
int nntr_tq_init(nntr_tq_codec *codec, uint32_t dim);
void nntr_tq_rotate(float *row, uint32_t dim);
/** Caller supplies dim floats of scratch. K metadata is norm, corrected scale.
 * Return -1 for nonfinite/unrepresentable input; never allocate. */
int nntr_tq_encode_key(const nntr_tq_codec *codec, const float *row,
                       uint8_t *packed, float *scratch);
int nntr_tq_encode_value(const nntr_tq_codec *codec, const float *row,
                         uint8_t *packed);
#ifdef __cplusplus
}
#endif
#endif
