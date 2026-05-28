// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   quick_ai.cpp
 * @date   28 May 2026
 * @brief  C ABI wrapper for quick.ai inference
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#define QUICK_AI_BUILD
#include "quick_ai.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "causal_lm.h"
#include "chat_template.h"
#include "deberta_v2.h"
#include "embedding_gemma.h"
#include "factory.h"
#include "gemma3_causallm.h"
#if !defined(_WIN32)
#include "gptoss_cached_slim_causallm.h"
#endif
#include "gptoss_causallm.h"
#include "json.hpp"
#if !defined(_WIN32) && !defined(__ANDROID__)
#include "multilingual_tinybert_16mb.h"
#endif
#include "openai_protocol.h"
#include "qwen2_causallm.h"
#include "qwen2_embedding.h"
#if !defined(_WIN32)
#include "qwen3_cached_slim_moe_causallm.h"
#endif
#include "qwen3_causallm.h"
#include "qwen3_embedding.h"
#include "qwen3_moe_causallm.h"
#include "qwen3_slim_moe_causallm.h"
#include "sentence_transformer.h"
#include "thread_manager.h"
#include "timm_vit/timm_vit_transformer.h"

#if defined(USE_BLAS) && __has_include("cblas_interface.h")
#include "cblas_interface.h"
#define QUICK_AI_HAS_CBLAS_INTERFACE 1
#else
#define QUICK_AI_HAS_CBLAS_INTERFACE 0
#endif

using json = nlohmann::json;

namespace {

std::mutex g_log_mutex;
quick_ai_log_cb g_log_cb = nullptr;
void *g_log_user_data = nullptr;
int g_log_level = 0;
thread_local std::string g_thread_last_error;

void setThreadError(const std::string &message) {
  g_thread_last_error = message;
}

void logMessage(int level, const std::string &message) {
  std::lock_guard<std::mutex> lock(g_log_mutex);
  if (g_log_cb && level >= g_log_level) {
    try {
      g_log_cb(level, message.c_str(), g_log_user_data);
    } catch (...) {
    }
  }
}

void registerModels() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    auto &factory = causallm::Factory::Instance();
    factory.registerModel("LlamaForCausalLM",
                          [](json cfg, json generation_cfg, json nntr_cfg) {
                            return std::make_unique<causallm::CausalLM>(
                              cfg, generation_cfg, nntr_cfg);
                          });
    factory.registerModel("Qwen2ForCausalLM",
                          [](json cfg, json generation_cfg, json nntr_cfg) {
                            return std::make_unique<causallm::Qwen2CausalLM>(
                              cfg, generation_cfg, nntr_cfg);
                          });
    factory.registerModel("Qwen2Embedding",
                          [](json cfg, json generation_cfg, json nntr_cfg) {
                            return std::make_unique<causallm::Qwen2Embedding>(
                              cfg, generation_cfg, nntr_cfg);
                          });
    factory.registerModel("Qwen3ForCausalLM",
                          [](json cfg, json generation_cfg, json nntr_cfg) {
                            return std::make_unique<causallm::Qwen3CausalLM>(
                              cfg, generation_cfg, nntr_cfg);
                          });
    factory.registerModel("Qwen3MoeForCausalLM",
                          [](json cfg, json generation_cfg, json nntr_cfg) {
                            return std::make_unique<causallm::Qwen3MoECausalLM>(
                              cfg, generation_cfg, nntr_cfg);
                          });
    factory.registerModel(
      "Qwen3SlimMoeForCausalLM",
      [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Qwen3SlimMoECausalLM>(
          cfg, generation_cfg, nntr_cfg);
      });
#if !defined(_WIN32)
    factory.registerModel(
      "Qwen3CachedSlimMoeForCausalLM",
      [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Qwen3CachedSlimMoECausalLM>(
          cfg, generation_cfg, nntr_cfg);
      });
#endif
    factory.registerModel("Qwen3Embedding",
                          [](json cfg, json generation_cfg, json nntr_cfg) {
                            return std::make_unique<causallm::Qwen3Embedding>(
                              cfg, generation_cfg, nntr_cfg);
                          });
    factory.registerModel("GptOssForCausalLM", [](json cfg, json generation_cfg,
                                                  json nntr_cfg) {
      return std::make_unique<causallm::GptOssForCausalLM>(cfg, generation_cfg,
                                                           nntr_cfg);
    });
#if !defined(_WIN32)
    factory.registerModel(
      "GptOssCachedSlimCausalLM",
      [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::GptOssCachedSlimCausalLM>(
          cfg, generation_cfg, nntr_cfg);
      });
#endif
    factory.registerModel("Gemma3ForCausalLM",
                          [](json cfg, json generation_cfg, json nntr_cfg) {
                            return std::make_unique<causallm::Gemma3CausalLM>(
                              cfg, generation_cfg, nntr_cfg);
                          });
    factory.registerModel("EmbeddingGemma",
                          [](json cfg, json generation_cfg, json nntr_cfg) {
                            return std::make_unique<causallm::EmbeddingGemma>(
                              cfg, generation_cfg, nntr_cfg);
                          });
    factory.registerModel("DebertaV2",
                          [](json cfg, json generation_cfg, json nntr_cfg) {
                            return std::make_unique<causallm::DebertaV2>(
                              cfg, generation_cfg, nntr_cfg);
                          });
#if !defined(_WIN32) && !defined(__ANDROID__)
    factory.registerModel(
      "MultilingualTinyBert", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::MultilingualTinyBert>(
          cfg, generation_cfg, nntr_cfg);
      });
#endif
    factory.registerModel("TimmViT", [](json cfg, json generation_cfg,
                                        json nntr_cfg) {
      return std::make_unique<causallm::TimmViTTransformer>(cfg, generation_cfg,
                                                            nntr_cfg);
    });
  });
}

std::string resolveArchitecture(std::string model_type,
                                const std::string &architecture) {
  std::transform(
    model_type.begin(), model_type.end(), model_type.begin(),
    [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  if (model_type == "embedding") {
    if (architecture == "Qwen3ForCausalLM")
      return "Qwen3Embedding";
    if (architecture == "Gemma3ForCausalLM" ||
        architecture == "Gemma3TextModel")
      return "EmbeddingGemma";
    if (architecture == "Qwen2Model")
      return "Qwen2Embedding";
    if (architecture == "BertForMaskedLM")
      return "MultilingualTinyBert";
    if (architecture == "TimmViT" ||
        architecture == "vit_base_patch16_siglip_224")
      return "TimmViT";
    if (architecture == "deberta-v2" || architecture == "DebertaV2Model" ||
        architecture == "DebertaV2ForMaskedLM")
      return "DebertaV2";
    throw std::invalid_argument("Unsupported embedding architecture: " +
                                architecture);
  }

  if (architecture == "TimmViT" ||
      architecture == "vit_base_patch16_siglip_224")
    return "TimmViT";

  return architecture;
}

std::string absoluteModelFile(const std::filesystem::path &model_dir,
                              const std::string &path) {
  const std::filesystem::path file(path);
  return file.is_absolute()
           ? file.string()
           : std::filesystem::absolute(model_dir / file).string();
}

std::string
fallbackPrompt(const std::vector<causallm::openai::ChatMessage> &messages) {
  std::ostringstream prompt;
  for (const auto &message : messages)
    prompt << message.role << ": " << message.content << '\n';
  prompt << "assistant: ";
  return prompt.str();
}

json withChatTemplateOptions(const json &chat_input, bool enable_thinking) {
  json request = chat_input;
  if (!request.contains("add_generation_prompt"))
    request["add_generation_prompt"] = true;
  request["enable_thinking"] = enable_thinking;
  return request;
}

causallm::GenerationOverrides
toGenerationOverrides(const causallm::openai::GenerationOptions &options) {
  causallm::GenerationOverrides overrides;
  overrides.temperature = options.temperature;
  overrides.top_p = options.top_p;
  overrides.top_k = options.top_k;
  overrides.max_tokens = options.max_tokens;
  return overrides;
}

bool requestDoSample(bool default_do_sample,
                     const causallm::openai::GenerationOptions &options) {
  if (options.temperature.has_value())
    return *options.temperature > 0.0F;
  if (options.top_p.has_value() || options.top_k.has_value())
    return true;
  return default_do_sample;
}

std::string modelKindFromArchitecture(const std::string &architecture) {
  if (architecture.find("Embedding") != std::string::npos ||
      architecture == "EmbeddingGemma" ||
      architecture == "MultilingualTinyBert" || architecture == "DebertaV2")
    return "embedding";
  if (architecture.find("CausalLM") != std::string::npos ||
      architecture.find("GptOss") != std::string::npos ||
      architecture.find("Llama") != std::string::npos ||
      architecture.find("Qwen") != std::string::npos)
    return "causallm";
  return "model";
}

std::string parseEmbeddingInput(const json &payload) {
  if (!payload.contains("input"))
    throw std::invalid_argument("embedding request must contain input");

  const auto &input = payload["input"];
  if (input.is_string())
    return input.get<std::string>();

  if (input.is_array()) {
    if (input.size() != 1 || !input[0].is_string()) {
      throw std::invalid_argument(
        "embedding input array must contain exactly one string");
    }
    return input[0].get<std::string>();
  }

  throw std::invalid_argument("embedding input must be a string or one-string "
                              "array");
}

std::string canonicalModelKind(std::string model_type,
                               const std::string &architecture) {
  std::transform(
    model_type.begin(), model_type.end(), model_type.begin(),
    [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  if (model_type == "embedding")
    return "embedding";
  if (model_type == "causallm" || model_type == "causal_lm")
    return "causallm";
  return modelKindFromArchitecture(architecture);
}

/**
 * @brief Scope guard that clears per-run generation callbacks and overrides.
 */
class CausalRunGuard {
public:
  explicit CausalRunGuard(causallm::CausalLM *model) : model_(model) {}
  CausalRunGuard(const CausalRunGuard &) = delete;
  CausalRunGuard &operator=(const CausalRunGuard &) = delete;
  ~CausalRunGuard() {
    if (!model_)
      return;
    model_->clearOnToken();
    model_->clearGenerationOverrides();
  }

private:
  causallm::CausalLM *model_;
};

quick_ai_status copyJsonToBuffer(const json &info, char *out_buf,
                                 int out_buf_len) {
  if (!out_buf || out_buf_len <= 0)
    return QUICK_AI_STATUS_INVALID_ARGUMENT;

  const std::string text = info.dump();
  if (static_cast<size_t>(out_buf_len) <= text.size()) {
    if (out_buf_len > 0)
      out_buf[0] = '\0';
    return QUICK_AI_STATUS_BUFFER_TOO_SMALL;
  }

  std::memcpy(out_buf, text.c_str(), text.size() + 1);
  return QUICK_AI_STATUS_OK;
}

} // namespace

/**
 * @brief Internal quick.ai context backing the public opaque C handle.
 */
struct quick_ai_ctx {
  std::string model_id;
  std::string model_path;
  std::string architecture;
  std::string model_kind;
  unsigned int ctx_len = 0;
  std::string system_head_prompt;
  std::string system_tail_prompt;
  bool do_sample = false;
  bool enable_thinking = false;
  double initialization_duration_ms = 0.0;
  std::optional<causallm::ChatTemplate> chat_template;
  std::unique_ptr<causallm::Transformer> model;
  std::mutex mutex;
  std::mutex error_mutex;
  std::atomic<bool> cancel_requested{false};
  std::string last_error;

  void setError(const std::string &message) {
    {
      std::lock_guard<std::mutex> lock(error_mutex);
      last_error = message;
    }
    setThreadError(message);
    logMessage(1, message);
  }
};

namespace {

std::unique_ptr<quick_ai_ctx> loadContext(const std::string &model_path,
                                          const quick_ai_init_options *opts) {
  registerModels();

  const int requested_threads =
    opts && opts->num_threads > 0 ? opts->num_threads : 1;
  nntrainer::ThreadManagerConfig thread_config{
    static_cast<uint32_t>(requested_threads), true};
  nntrainer::ThreadManager::setConfig(thread_config);

#if QUICK_AI_HAS_CBLAS_INTERFACE
  nntrainer::__openblas_set_num_threads(requested_threads);
#else
  if (opts && opts->num_threads > 0)
    logMessage(2, "num_threads requested but cblas_interface.h is unavailable");
#endif

  auto ctx = std::make_unique<quick_ai_ctx>();
  ctx->model_path = std::filesystem::absolute(model_path).string();
  ctx->model_id = std::filesystem::path(ctx->model_path).filename().string();
  ctx->enable_thinking = opts && opts->enable_thinking != 0;

  const std::filesystem::path model_dir(ctx->model_path);
  json cfg = causallm::LoadJsonFile((model_dir / "config.json").string());
  json generation_cfg = json::object();
  if (std::filesystem::exists(model_dir / "generation_config.json"))
    generation_cfg =
      causallm::LoadJsonFile((model_dir / "generation_config.json").string());
  json nntr_cfg =
    causallm::LoadJsonFile((model_dir / "nntr_config.json").string());
  ctx->ctx_len = nntr_cfg.value("max_seq_len", 0U);

  if (nntr_cfg.contains("tokenizer_file") &&
      nntr_cfg["tokenizer_file"].is_string()) {
    nntr_cfg["tokenizer_file"] =
      absoluteModelFile(model_dir, nntr_cfg["tokenizer_file"]);
  }
  if (nntr_cfg.contains("model_file_name") &&
      nntr_cfg["model_file_name"].is_string()) {
    nntr_cfg["model_file_name"] =
      absoluteModelFile(model_dir, nntr_cfg["model_file_name"]);
  }
  if (nntr_cfg.contains("module_config_path") &&
      nntr_cfg["module_config_path"].is_string()) {
    nntr_cfg["module_config_path"] =
      absoluteModelFile(model_dir, nntr_cfg["module_config_path"]);
  }

  if (nntr_cfg.contains("system_prompt")) {
    ctx->system_head_prompt =
      nntr_cfg["system_prompt"].value("head_prompt", "");
    ctx->system_tail_prompt =
      nntr_cfg["system_prompt"].value("tail_prompt", "");
  }

  std::string architecture;
  if (cfg.contains("architectures") && cfg["architectures"].is_array() &&
      !cfg["architectures"].empty()) {
    architecture = cfg["architectures"].get<std::vector<std::string>>()[0];
  } else if (cfg.contains("architecture") && cfg["architecture"].is_string()) {
    architecture = cfg["architecture"].get<std::string>();
  } else if (cfg.contains("model_type") && cfg["model_type"].is_string()) {
    architecture = cfg["model_type"].get<std::string>();
  } else {
    throw std::invalid_argument(
      "config.json must contain architecture metadata");
  }

  if (nntr_cfg.contains("model_type") && nntr_cfg["model_type"].is_string())
    architecture = resolveArchitecture(
      nntr_cfg["model_type"].get<std::string>(), architecture);
  ctx->architecture = architecture;
  ctx->model_kind = canonicalModelKind(
    nntr_cfg.value("model_type", std::string()), architecture);

  if (causallm::ChatTemplate::Exists(model_dir.string()))
    ctx->chat_template.emplace(
      causallm::ChatTemplate::Load(model_dir.string()));

  ctx->model = causallm::Factory::Instance().create(architecture, cfg,
                                                    generation_cfg, nntr_cfg);
  if (!ctx->model) {
    std::ostringstream os;
    os << "Unknown architecture: " << architecture
       << ". Registered architectures:";
    causallm::Factory::Instance().printRegistered(os);
    throw std::runtime_error(os.str());
  }

  const std::string weight_file =
    nntr_cfg.at("model_file_name").get<std::string>();
  const auto start_initialize = std::chrono::high_resolution_clock::now();
  ctx->model->initialize();
  ctx->model->load_weight(weight_file);
  const auto finish_initialize = std::chrono::high_resolution_clock::now();
  ctx->initialization_duration_ms = std::chrono::duration<double, std::milli>(
                                      finish_initialize - start_initialize)
                                      .count();
  ctx->do_sample = generation_cfg.value("do_sample", false);
  return ctx;
}

} // namespace

quick_ai_status quick_ai_ctx_create(const char *model_dir_utf8,
                                    const quick_ai_init_options *opts,
                                    quick_ai_ctx **out_ctx) noexcept {
  try {
    if (!model_dir_utf8 || !out_ctx) {
      setThreadError("model_dir_utf8 and out_ctx must be non-null");
      return QUICK_AI_STATUS_INVALID_ARGUMENT;
    }

    *out_ctx = nullptr;
    auto ctx = loadContext(model_dir_utf8, opts);
    *out_ctx = ctx.release();
    return QUICK_AI_STATUS_OK;
  } catch (const std::exception &e) {
    setThreadError(e.what());
    logMessage(1, e.what());
    return QUICK_AI_STATUS_MODEL_LOAD_FAILED;
  } catch (...) {
    setThreadError("unknown error while creating quick.ai context");
    return QUICK_AI_STATUS_INTERNAL_ERROR;
  }
}

void quick_ai_ctx_free(quick_ai_ctx *ctx) noexcept {
  try {
    delete ctx;
  } catch (...) {
  }
}

quick_ai_status quick_ai_model_info(quick_ai_ctx *ctx, char *out_buf,
                                    int out_buf_len) noexcept {
  try {
    if (!ctx) {
      setThreadError("ctx must be non-null");
      return QUICK_AI_STATUS_INVALID_ARGUMENT;
    }

    std::lock_guard<std::mutex> lock(ctx->mutex);
    json info = {
      {"id", ctx->model_id},
      {"model_path", ctx->model_path},
      {"architecture", ctx->architecture},
      {"model_type", ctx->model_kind},
      {"kind", ctx->model_kind},
      {"max_seq_len", ctx->model ? ctx->model->getMaxSeqLen() : 0},
      {"init_seq_len", ctx->model ? ctx->model->getInitSeqLen() : 0},
      {"num_to_generate", ctx->model ? ctx->model->getNumToGenerate() : 0},
      {"batch_size", ctx->model ? ctx->model->getBatchSize() : 0},
      {"hidden_dim", ctx->model ? ctx->model->getHiddenDim() : 0},
      {"ctx_len", ctx->ctx_len}};
    const quick_ai_status status = copyJsonToBuffer(info, out_buf, out_buf_len);
    if (status != QUICK_AI_STATUS_OK)
      ctx->setError("model info output buffer is too small");
    return status;
  } catch (const std::exception &e) {
    ctx->setError(e.what());
    return QUICK_AI_STATUS_INTERNAL_ERROR;
  } catch (...) {
    ctx->setError("unknown error while reading model info");
    return QUICK_AI_STATUS_INTERNAL_ERROR;
  }
}

quick_ai_status quick_ai_generate(quick_ai_ctx *ctx,
                                  const char *request_json_utf8,
                                  quick_ai_token_cb on_token, void *user_data,
                                  quick_ai_usage *usage_out) noexcept {
  try {
    if (!ctx || !request_json_utf8) {
      setThreadError("ctx and request_json_utf8 must be non-null");
      return QUICK_AI_STATUS_INVALID_ARGUMENT;
    }
    if (usage_out) {
      usage_out->prompt_tokens = 0;
      usage_out->completion_tokens = 0;
    }

    std::lock_guard<std::mutex> lock(ctx->mutex);
    ctx->cancel_requested.store(false);

    const json payload = json::parse(request_json_utf8);
    const auto parsed = causallm::openai::parseChatCompletionRequest(payload);

    const std::string prompt =
      ctx->chat_template.has_value()
        ? ctx->chat_template->apply(
            withChatTemplateOptions(parsed.chat_input, ctx->enable_thinking))
        : fallbackPrompt(parsed.messages);

    auto *causal_model = dynamic_cast<causallm::CausalLM *>(ctx->model.get());
    if (!causal_model) {
      ctx->setError("loaded model does not support chat generation");
      return QUICK_AI_STATUS_GENERATION_FAILED;
    }

    CausalRunGuard run_guard(causal_model);
    causallm::openai::StreamingTextFilter filter(parsed.options.stop,
                                                 ctx->enable_thinking);
    bool callback_cancelled = false;
    bool stop_requested = false;

    causal_model->resetGenerationState();
    causal_model->setGenerationOverrides(toGenerationOverrides(parsed.options));
    causal_model->setOnToken([&](const std::string &piece) {
      if (ctx->cancel_requested.load())
        return false;

      auto result = filter.push(piece);
      if (!result.text.empty() && on_token) {
        const int cb_status = on_token(result.text.c_str(), user_data);
        if (cb_status != 0) {
          callback_cancelled = true;
          return false;
        }
      }

      stop_requested = stop_requested || result.stop;
      return !stop_requested && !ctx->cancel_requested.load();
    });

    ctx->model->run(prompt, requestDoSample(ctx->do_sample, parsed.options),
                    ctx->system_head_prompt, ctx->system_tail_prompt, false);

    if (!callback_cancelled && !ctx->cancel_requested.load() &&
        !stop_requested) {
      const std::string tail = filter.flush();
      if (!tail.empty() && on_token && on_token(tail.c_str(), user_data) != 0)
        callback_cancelled = true;
    }

    const auto metrics = ctx->model->getPerformanceMetrics();
    if (usage_out) {
      usage_out->prompt_tokens = metrics.prefill_tokens;
      usage_out->completion_tokens = metrics.generation_tokens;
    }

    if (callback_cancelled || ctx->cancel_requested.load()) {
      ctx->setError("generation cancelled");
      return QUICK_AI_STATUS_CANCELLED;
    }

    return QUICK_AI_STATUS_OK;
  } catch (const std::exception &e) {
    if (ctx)
      ctx->setError(e.what());
    else
      setThreadError(e.what());
    return QUICK_AI_STATUS_GENERATION_FAILED;
  } catch (...) {
    if (ctx)
      ctx->setError("unknown error while generating text");
    else
      setThreadError("unknown error while generating text");
    return QUICK_AI_STATUS_INTERNAL_ERROR;
  }
}

quick_ai_status quick_ai_embed(quick_ai_ctx *ctx, const char *request_json_utf8,
                               float **out_embedding, int *out_count,
                               int *out_dim) noexcept {
  try {
    if (!ctx || !request_json_utf8 || !out_embedding || !out_count ||
        !out_dim) {
      setThreadError("ctx, request_json_utf8, out_embedding, out_count, and "
                     "out_dim must be non-null");
      return QUICK_AI_STATUS_INVALID_ARGUMENT;
    }

    *out_embedding = nullptr;
    *out_count = 0;
    *out_dim = 0;

    std::lock_guard<std::mutex> lock(ctx->mutex);

    auto *embedding_model =
      dynamic_cast<causallm::SentenceTransformer *>(ctx->model.get());
    if (!embedding_model) {
      ctx->setError("loaded model does not support embedding");
      return QUICK_AI_STATUS_GENERATION_FAILED;
    }

    const json payload = json::parse(request_json_utf8);
    const std::string input = parseEmbeddingInput(payload);
    const unsigned int count = 1;
    const int dim = ctx->model->getHiddenDim();
    if (dim <= 0) {
      ctx->setError("embedding model reported invalid output dimensions");
      return QUICK_AI_STATUS_INTERNAL_ERROR;
    }

    std::vector<float *> results = embedding_model->encode(
      input, ctx->system_head_prompt, ctx->system_tail_prompt);
    if (results.empty() || !results[0]) {
      ctx->setError("embedding model returned no output");
      return QUICK_AI_STATUS_GENERATION_FAILED;
    }

    const size_t total = static_cast<size_t>(dim);
    float *embedding = static_cast<float *>(std::malloc(total * sizeof(float)));
    if (!embedding) {
      for (auto *out : results)
        delete[] out;
      ctx->setError("failed to allocate embedding output buffer");
      return QUICK_AI_STATUS_INTERNAL_ERROR;
    }

    std::memcpy(embedding, results[0], total * sizeof(float));
    for (auto *out : results)
      delete[] out;

    *out_embedding = embedding;
    *out_count = 1;
    *out_dim = dim;
    return QUICK_AI_STATUS_OK;
  } catch (const std::exception &e) {
    if (ctx)
      ctx->setError(e.what());
    else
      setThreadError(e.what());
    return QUICK_AI_STATUS_GENERATION_FAILED;
  } catch (...) {
    if (ctx)
      ctx->setError("unknown error while creating embedding");
    else
      setThreadError("unknown error while creating embedding");
    return QUICK_AI_STATUS_INTERNAL_ERROR;
  }
}

void quick_ai_free(void *ptr) noexcept {
  try {
    std::free(ptr);
  } catch (...) {
  }
}

quick_ai_status quick_ai_get_performance_metrics(
  quick_ai_ctx *ctx, quick_ai_performance_metrics *metrics) noexcept {
  try {
    if (!ctx || !metrics) {
      setThreadError("ctx and metrics must be non-null");
      return QUICK_AI_STATUS_INVALID_ARGUMENT;
    }

    std::lock_guard<std::mutex> lock(ctx->mutex);
    const auto internal_metrics = ctx->model->getPerformanceMetrics();
    metrics->prefill_tokens = internal_metrics.prefill_tokens;
    metrics->prefill_duration_ms = internal_metrics.prefill_duration_ms;
    metrics->generation_tokens = internal_metrics.generation_tokens;
    metrics->generation_duration_ms = internal_metrics.generation_duration_ms;
    metrics->total_duration_ms = internal_metrics.total_duration_ms;
    metrics->initialization_duration_ms = ctx->initialization_duration_ms;
    metrics->peak_memory_kb =
      static_cast<unsigned long long>(internal_metrics.peak_memory_kb);
    return QUICK_AI_STATUS_OK;
  } catch (const std::exception &e) {
    if (ctx)
      ctx->setError(e.what());
    else
      setThreadError(e.what());
    return QUICK_AI_STATUS_INTERNAL_ERROR;
  } catch (...) {
    if (ctx)
      ctx->setError("unknown error while reading performance metrics");
    else
      setThreadError("unknown error while reading performance metrics");
    return QUICK_AI_STATUS_INTERNAL_ERROR;
  }
}

quick_ai_status quick_ai_cancel(quick_ai_ctx *ctx) noexcept {
  try {
    if (!ctx) {
      setThreadError("ctx must be non-null");
      return QUICK_AI_STATUS_INVALID_ARGUMENT;
    }
    ctx->cancel_requested.store(true);
    return QUICK_AI_STATUS_OK;
  } catch (...) {
    setThreadError("unknown error while cancelling generation");
    return QUICK_AI_STATUS_INTERNAL_ERROR;
  }
}

const char *quick_ai_version(void) noexcept {
  return "quick.ai/" QUICK_AI_ABI_VERSION;
}

const char *quick_ai_last_error(quick_ai_ctx *ctx) noexcept {
  if (ctx) {
    std::lock_guard<std::mutex> lock(ctx->error_mutex);
    g_thread_last_error = ctx->last_error;
    return g_thread_last_error.c_str();
  }
  return g_thread_last_error.c_str();
}

void quick_ai_set_log_callback(quick_ai_log_cb cb, void *user_data,
                               int level) noexcept {
  try {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_log_cb = cb;
    g_log_user_data = user_data;
    g_log_level = level;
  } catch (...) {
  }
}
