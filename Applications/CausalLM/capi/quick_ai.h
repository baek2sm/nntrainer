// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   quick_ai.h
 * @date   28 May 2026
 * @brief  Public C ABI for quick.ai single-DLL inference
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef QUICK_AI_H_
#define QUICK_AI_H_

#ifdef _WIN32
#ifdef QUICK_AI_BUILD
#define QUICK_AI_API __declspec(dllexport)
#else
#define QUICK_AI_API __declspec(dllimport)
#endif
#else
#define QUICK_AI_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
#define QUICK_AI_NOEXCEPT noexcept
#else
#define QUICK_AI_NOEXCEPT
#endif

#define QUICK_AI_ABI_VERSION_MAJOR 0
#define QUICK_AI_ABI_VERSION_MINOR 2
#define QUICK_AI_ABI_VERSION_PATCH 0
#define QUICK_AI_ABI_VERSION "0.2.0"

#define QUICK_AI_STATUS_OK 0
#define QUICK_AI_STATUS_INVALID_ARGUMENT -1
#define QUICK_AI_STATUS_MODEL_LOAD_FAILED -2
#define QUICK_AI_STATUS_GENERATION_FAILED -3
#define QUICK_AI_STATUS_CANCELLED -4
#define QUICK_AI_STATUS_BUFFER_TOO_SMALL -5
#define QUICK_AI_STATUS_INTERNAL_ERROR -6

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque CausalLM inference context.
 */
typedef struct quick_ai_ctx quick_ai_ctx;

/**
 * @brief ABI status code. Zero is success; negative values are errors.
 */
typedef int quick_ai_status;

/**
 * @brief Context creation options.
 */
typedef struct {
  int num_threads;     /**< Inference thread count; <=0 selects ABI default */
  int enable_thinking; /**< Non-zero forwards model thinking text */
} quick_ai_init_options;

/**
 * @brief Token usage populated after generation.
 */
typedef struct {
  unsigned long long prompt_tokens;
  unsigned long long completion_tokens;
} quick_ai_usage;

/**
 * @brief Performance metrics populated by the last model run.
 */
typedef struct {
  unsigned int prefill_tokens;
  double prefill_duration_ms;
  unsigned int generation_tokens;
  double generation_duration_ms;
  double total_duration_ms;
  double initialization_duration_ms;
  unsigned long long peak_memory_kb;
} quick_ai_performance_metrics;

/**
 * @brief Streaming token callback.
 * @return 0 to continue, non-zero to request cancellation.
 */
typedef int (*quick_ai_token_cb)(const char *utf8_delta, void *user_data);

/**
 * @brief Log callback.
 */
typedef void (*quick_ai_log_cb)(int level, const char *utf8_msg,
                                void *user_data);

/**
 * @brief Create a context from a model directory.
 */
QUICK_AI_API quick_ai_status quick_ai_ctx_create(
  const char *model_dir_utf8, const quick_ai_init_options *opts,
  quick_ai_ctx **out_ctx) QUICK_AI_NOEXCEPT;

/**
 * @brief Free a context. Passing NULL is allowed.
 */
QUICK_AI_API void quick_ai_ctx_free(quick_ai_ctx *ctx) QUICK_AI_NOEXCEPT;

/**
 * @brief Write model info JSON into out_buf.
 */
QUICK_AI_API quick_ai_status quick_ai_model_info(
  quick_ai_ctx *ctx, char *out_buf, int out_buf_len) QUICK_AI_NOEXCEPT;

/**
 * @brief Generate from an OpenAI chat-completions compatible JSON request.
 */
QUICK_AI_API quick_ai_status quick_ai_generate(
  quick_ai_ctx *ctx, const char *request_json_utf8, quick_ai_token_cb on_token,
  void *user_data, quick_ai_usage *usage_out) QUICK_AI_NOEXCEPT;

/**
 * @brief Encode one OpenAI-compatible embeddings request.
 *
 * The returned embedding buffer is allocated by the ABI and must be released
 * with quick_ai_free().
 */
QUICK_AI_API quick_ai_status quick_ai_embed(quick_ai_ctx *ctx,
                                            const char *request_json_utf8,
                                            float **out_embedding,
                                            int *out_count,
                                            int *out_dim) QUICK_AI_NOEXCEPT;

/**
 * @brief Free memory returned by this ABI. Passing NULL is allowed.
 */
QUICK_AI_API void quick_ai_free(void *ptr) QUICK_AI_NOEXCEPT;

/**
 * @brief Read performance metrics from the context after a model run.
 */
QUICK_AI_API quick_ai_status quick_ai_get_performance_metrics(
  quick_ai_ctx *ctx, quick_ai_performance_metrics *metrics) QUICK_AI_NOEXCEPT;

/**
 * @brief Request cancellation from another thread.
 */
QUICK_AI_API quick_ai_status quick_ai_cancel(quick_ai_ctx *ctx)
  QUICK_AI_NOEXCEPT;

/**
 * @brief Return library and ABI version text.
 */
QUICK_AI_API const char *quick_ai_version(void) QUICK_AI_NOEXCEPT;

/**
 * @brief Return the last error for ctx, or thread-local create error if NULL.
 */
QUICK_AI_API const char *
quick_ai_last_error(quick_ai_ctx *ctx) QUICK_AI_NOEXCEPT;

/**
 * @brief Set a process-wide log callback.
 */
QUICK_AI_API void quick_ai_set_log_callback(quick_ai_log_cb cb, void *user_data,
                                            int level) QUICK_AI_NOEXCEPT;

#ifdef __cplusplus
}
#endif

#endif /* QUICK_AI_H_ */
