// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   openai_server.cpp
 * @date   27 May 2026
 * @brief  OpenAI-compatible REST server for CausalLM
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <climits>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include "json.hpp"
#include <factory.h>
#include <openai_protocol.h>

#include "causal_lm.h"
#include "chat_template.h"
#include "deberta_v2.h"
#include "embedding_gemma.h"
#include "gemma3_causallm.h"
#if !defined(_WIN32)
#include "gptoss_cached_slim_causallm.h"
#endif
#include "gptoss_causallm.h"
#if !defined(_WIN32) && !defined(__ANDROID__)
#include "multilingual_tinybert_16mb.h"
#endif
#include "qwen2_causallm.h"
#include "qwen2_embedding.h"
#if !defined(_WIN32)
#include "qwen3_cached_slim_moe_causallm.h"
#endif
#include "qwen3_causallm.h"
#include "qwen3_embedding.h"
#include "qwen3_moe_causallm.h"
#include "qwen3_slim_moe_causallm.h"
#include "timm_vit/timm_vit_transformer.h"

using json = nlohmann::json;

/**
 * @brief Anonymous namespace for OpenAI server implementation details
 */
namespace {

#ifdef _WIN32
using SocketHandle = SOCKET;
constexpr SocketHandle invalid_socket_handle = INVALID_SOCKET;
#else
using SocketHandle = int;
constexpr SocketHandle invalid_socket_handle = -1;
#endif

std::atomic<bool> keep_running{true};
constexpr size_t max_http_body_size = 16 * 1024 * 1024;

/**
 * @brief Close a platform socket.
 */
void closeSocket(SocketHandle socket) {
#ifdef _WIN32
  closesocket(socket);
#else
  close(socket);
#endif
}

/**
 * @brief Send all bytes in a response buffer.
 */
bool sendAll(SocketHandle socket, const std::string &data) {
  const char *ptr = data.data();
  size_t remaining = data.size();
  while (remaining > 0) {
#ifdef _WIN32
    const int chunk = static_cast<int>(std::min<size_t>(remaining, INT_MAX));
    const int sent = send(socket, ptr, chunk, 0);
#else
    const ssize_t sent = send(socket, ptr, remaining, 0);
#endif
    if (sent <= 0)
      return false;
    ptr += sent;
    remaining -= static_cast<size_t>(sent);
  }
  return true;
}

/**
 * @brief Receive bytes from a socket.
 */
int recvSome(SocketHandle socket, char *buffer, int size) {
#ifdef _WIN32
  return recv(socket, buffer, size, 0);
#else
  return static_cast<int>(recv(socket, buffer, static_cast<size_t>(size), 0));
#endif
}

/**
 * @brief Lowercase an ASCII string.
 */
std::string lowercase(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return text;
}

/**
 * @brief Trim ASCII whitespace from a string.
 */
std::string trim(std::string text) {
  auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
  text.erase(text.begin(),
             std::find_if_not(text.begin(), text.end(), is_space));
  text.erase(std::find_if_not(text.rbegin(), text.rend(), is_space).base(),
             text.end());
  return text;
}

/**
 * @brief Parsed HTTP request.
 */
struct HttpRequest {
  std::string method;                        /**< HTTP method */
  std::string path;                          /**< Request target path */
  std::map<std::string, std::string> header; /**< Lowercase headers */
  std::string body;                          /**< Request body */
};

/**
 * @brief Read one HTTP/1.1 request. Supports Content-Length bodies.
 */
HttpRequest readHttpRequest(SocketHandle socket) {
  std::string data;
  char buffer[4096];
  size_t header_end = std::string::npos;

  while ((header_end = data.find("\r\n\r\n")) == std::string::npos) {
    const int received = recvSome(socket, buffer, sizeof(buffer));
    if (received <= 0)
      throw std::runtime_error("connection closed before HTTP headers");
    data.append(buffer, static_cast<size_t>(received));
    if (data.size() > 1024 * 1024)
      throw std::runtime_error("HTTP headers are too large");
  }

  std::istringstream header_stream(data.substr(0, header_end));
  std::string request_line;
  std::getline(header_stream, request_line);
  if (!request_line.empty() && request_line.back() == '\r')
    request_line.pop_back();

  HttpRequest request;
  std::istringstream request_line_stream(request_line);
  request_line_stream >> request.method >> request.path;
  if (request.method.empty() || request.path.empty())
    throw std::runtime_error("invalid HTTP request line");

  const size_t query_pos = request.path.find('?');
  if (query_pos != std::string::npos)
    request.path.erase(query_pos);

  std::string line;
  while (std::getline(header_stream, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    const size_t colon = line.find(':');
    if (colon == std::string::npos)
      continue;
    request.header[lowercase(trim(line.substr(0, colon)))] =
      trim(line.substr(colon + 1));
  }

  size_t content_length = 0;
  if (auto it = request.header.find("content-length");
      it != request.header.end()) {
    content_length = static_cast<size_t>(std::stoul(it->second));
  }
  if (content_length > max_http_body_size)
    throw std::runtime_error("HTTP body is too large");

  const size_t body_start = header_end + 4;
  request.body = data.substr(body_start);
  while (request.body.size() < content_length) {
    const int received = recvSome(socket, buffer, sizeof(buffer));
    if (received <= 0)
      throw std::runtime_error("connection closed before HTTP body");
    request.body.append(buffer, static_cast<size_t>(received));
  }
  if (request.body.size() > content_length)
    request.body.resize(content_length);

  return request;
}

/**
 * @brief Create a JSON HTTP response.
 */
std::string httpJson(int status, const std::string &reason,
                     const json &payload) {
  const std::string body = payload.dump();
  std::ostringstream response;
  response << "HTTP/1.1 " << status << ' ' << reason << "\r\n"
           << "Content-Type: application/json\r\n"
           << "Content-Length: " << body.size() << "\r\n"
           << "Connection: close\r\n\r\n"
           << body;
  return response.str();
}

/**
 * @brief Create an SSE HTTP response header.
 */
std::string httpSseHeader() {
  return "HTTP/1.1 200 OK\r\n"
         "Content-Type: text/event-stream\r\n"
         "Cache-Control: no-cache\r\n"
         "Connection: close\r\n\r\n";
}

/**
 * @brief Create one SSE data frame.
 */
std::string sseData(const std::string &payload) {
  return "data: " + payload + "\n\n";
}

/**
 * @brief Send one OpenAI-style SSE JSON frame.
 */
bool sendSseJson(SocketHandle client, const json &payload) {
  return sendAll(client, sseData(payload.dump()));
}

/**
 * @brief Send the OpenAI SSE stream terminator.
 */
bool sendSseDone(SocketHandle client) {
  return sendAll(client, sseData("[DONE]"));
}

/**
 * @brief Resolve config architecture names to registered factory names.
 */
std::string resolveArchitecture(std::string model_type,
                                const std::string &architecture) {
  std::transform(model_type.begin(), model_type.end(), model_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (model_type == "embedding") {
    if (architecture == "Qwen3ForCausalLM") {
      return "Qwen3Embedding";
    } else if (architecture == "Gemma3ForCausalLM" ||
               architecture == "Gemma3TextModel") {
      return "EmbeddingGemma";
    } else if (architecture == "Qwen2Model") {
      return "Qwen2Embedding";
    } else if (architecture == "BertForMaskedLM") {
      return "MultilingualTinyBert";
    } else if (architecture == "TimmViT" ||
               architecture == "vit_base_patch16_siglip_224") {
      return "TimmViT";
    } else if (architecture == "deberta-v2" ||
               architecture == "DebertaV2Model" ||
               architecture == "DebertaV2ForMaskedLM") {
      return "DebertaV2";
    }
    throw std::invalid_argument("Unsupported embedding architecture: " +
                                architecture);
  }

  if (architecture == "TimmViT" ||
      architecture == "vit_base_patch16_siglip_224") {
    return "TimmViT";
  }

  return architecture;
}

/**
 * @brief Register runnable CausalLM models in the local factory.
 */
void registerModels() {
  auto &factory = causallm::Factory::Instance();
  factory.registerModel("LlamaForCausalLM", [](json cfg, json generation_cfg,
                                               json nntr_cfg) {
    return std::make_unique<causallm::CausalLM>(cfg, generation_cfg, nntr_cfg);
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
  factory.registerModel("Qwen3SlimMoeForCausalLM", [](json cfg,
                                                      json generation_cfg,
                                                      json nntr_cfg) {
    return std::make_unique<causallm::Qwen3SlimMoECausalLM>(cfg, generation_cfg,
                                                            nntr_cfg);
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
  factory.registerModel("GptOssForCausalLM",
                        [](json cfg, json generation_cfg, json nntr_cfg) {
                          return std::make_unique<causallm::GptOssForCausalLM>(
                            cfg, generation_cfg, nntr_cfg);
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
  factory.registerModel("DebertaV2", [](json cfg, json generation_cfg,
                                        json nntr_cfg) {
    return std::make_unique<causallm::DebertaV2>(cfg, generation_cfg, nntr_cfg);
  });
#if !defined(_WIN32) && !defined(__ANDROID__)
  factory.registerModel(
    "MultilingualTinyBert", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::MultilingualTinyBert>(
        cfg, generation_cfg, nntr_cfg);
    });
#endif
  factory.registerModel("TimmViT",
                        [](json cfg, json generation_cfg, json nntr_cfg) {
                          return std::make_unique<causallm::TimmViTTransformer>(
                            cfg, generation_cfg, nntr_cfg);
                        });
}

/**
 * @brief Loaded server model state.
 */
struct LoadedModel {
  std::string model_id;           /**< Public model id */
  std::string model_path;         /**< Model directory */
  std::string architecture;       /**< Factory architecture */
  std::string system_head_prompt; /**< Optional head prompt */
  std::string system_tail_prompt; /**< Optional tail prompt */
  bool do_sample = false;         /**< Sampling mode */
  bool enable_thinking = false;   /**< Qwen thinking template mode */
  std::optional<causallm::ChatTemplate>
    chat_template;                              /**< Optional chat template */
  std::unique_ptr<causallm::Transformer> model; /**< Loaded model */
  std::mutex mutex; /**< Single-model request guard */
};

/**
 * @brief Convert a possibly relative model file name to an absolute path.
 */
std::string absoluteModelFile(const std::filesystem::path &model_dir,
                              const std::string &path) {
  const std::filesystem::path file(path);
  return file.is_absolute() ? file.string() : (model_dir / file).string();
}

/**
 * @brief Load a CausalLM model from a file-based model directory.
 */
std::unique_ptr<LoadedModel> loadModel(const std::string &model_path,
                                       const std::string &model_id,
                                       bool enable_thinking) {
  registerModels();

  auto loaded = std::make_unique<LoadedModel>();
  loaded->model_path = std::filesystem::absolute(model_path).string();
  loaded->model_id =
    model_id.empty()
      ? std::filesystem::path(loaded->model_path).filename().string()
      : model_id;
  loaded->enable_thinking = enable_thinking;

  const std::filesystem::path model_dir(loaded->model_path);
  json cfg = causallm::LoadJsonFile((model_dir / "config.json").string());
  json generation_cfg = json::object();
  if (std::filesystem::exists(model_dir / "generation_config.json")) {
    generation_cfg =
      causallm::LoadJsonFile((model_dir / "generation_config.json").string());
  }
  json nntr_cfg =
    causallm::LoadJsonFile((model_dir / "nntr_config.json").string());

  if (nntr_cfg.contains("tokenizer_file") &&
      nntr_cfg["tokenizer_file"].is_string()) {
    nntr_cfg["tokenizer_file"] =
      absoluteModelFile(model_dir, nntr_cfg["tokenizer_file"]);
  }

  if (nntr_cfg.contains("system_prompt")) {
    loaded->system_head_prompt =
      nntr_cfg["system_prompt"]["head_prompt"].get<std::string>();
    loaded->system_tail_prompt =
      nntr_cfg["system_prompt"]["tail_prompt"].get<std::string>();
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
      "config.json must contain architecture metadata.");
  }

  if (nntr_cfg.contains("model_type")) {
    architecture = resolveArchitecture(
      nntr_cfg["model_type"].get<std::string>(), architecture);
  }
  loaded->architecture = architecture;

  const std::string weight_file = absoluteModelFile(
    model_dir, nntr_cfg["model_file_name"].get<std::string>());

  if (causallm::ChatTemplate::Exists(model_dir.string()))
    loaded->chat_template.emplace(
      causallm::ChatTemplate::Load(model_dir.string()));

  loaded->model = causallm::Factory::Instance().create(
    architecture, cfg, generation_cfg, nntr_cfg);
  if (!loaded->model) {
    std::ostringstream os;
    os << "Unknown architecture: " << architecture
       << ". Registered architectures:";
    causallm::Factory::Instance().printRegistered(os);
    throw std::runtime_error(os.str());
  }

  loaded->model->initialize();
  loaded->model->load_weight(weight_file);
  loaded->do_sample = generation_cfg.value("do_sample", false);
  return loaded;
}

/**
 * @brief Join messages when no tokenizer chat template is available.
 */
std::string
fallbackPrompt(const std::vector<causallm::openai::ChatMessage> &messages) {
  std::ostringstream prompt;
  for (const auto &message : messages) {
    prompt << message.role << ": " << message.content << '\n';
  }
  prompt << "assistant: ";
  return prompt.str();
}

/**
 * @brief Add server-controlled template settings to parsed chat input.
 */
json withServerChatTemplateOptions(const json &chat_input,
                                   bool enable_thinking) {
  json request = chat_input;
  if (!request.contains("add_generation_prompt"))
    request["add_generation_prompt"] = true;
  request["enable_thinking"] = enable_thinking;
  return request;
}

/**
 * @brief Restore temporary CausalLM request state when a run exits.
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

/**
 * @brief Convert OpenAI request options into CausalLM generation overrides.
 */
causallm::GenerationOverrides
toGenerationOverrides(const causallm::openai::GenerationOptions &options) {
  causallm::GenerationOverrides overrides;
  overrides.temperature = options.temperature;
  overrides.top_p = options.top_p;
  overrides.top_k = options.top_k;
  overrides.max_tokens = options.max_tokens;
  return overrides;
}

/**
 * @brief Choose sampling mode for one request.
 */
bool requestDoSample(bool default_do_sample,
                     const causallm::openai::GenerationOptions &options) {
  if ((options.temperature.has_value() && *options.temperature > 0.0F) ||
      options.top_p.has_value() || options.top_k.has_value()) {
    return true;
  }

  if (options.temperature.has_value())
    return false;

  return default_do_sample;
}

/**
 * @brief Clean a full generated string with the streaming filter rules.
 */
std::string
cleanGeneratedText(const std::string &raw,
                   const causallm::openai::GenerationOptions &options,
                   bool enable_thinking) {
  causallm::openai::StreamingTextFilter filter(options.stop, enable_thinking);
  auto result = filter.push(raw);
  std::string text = std::move(result.text);
  text += filter.flush();
  return trim(text);
}

/**
 * @brief Run one prompt through the loaded model.
 */
std::pair<std::string, TransformerPerformanceMetrics>
runPrompt(LoadedModel &loaded, const std::string &prompt,
          const causallm::openai::GenerationOptions &options) {
  std::lock_guard<std::mutex> lock(loaded.mutex);

  auto *causal_model = dynamic_cast<causallm::CausalLM *>(loaded.model.get());
  CausalRunGuard run_guard(causal_model);
  if (causal_model) {
    causal_model->resetGenerationState();
    causal_model->setGenerationOverrides(toGenerationOverrides(options));
  }

  loaded.model->run(prompt, requestDoSample(loaded.do_sample, options),
                    loaded.system_head_prompt, loaded.system_tail_prompt,
                    false);

  std::string output;
  if (causal_model) {
    output = causal_model->getOutput(0);
  }

  return {cleanGeneratedText(output, options, loaded.enable_thinking),
          loaded.model->getPerformanceMetrics()};
}

/**
 * @brief Stream one prompt through the loaded CausalLM model.
 */
void streamPrompt(LoadedModel &loaded, const std::string &prompt,
                  const causallm::openai::GenerationOptions &options,
                  SocketHandle client) {
  std::lock_guard<std::mutex> lock(loaded.mutex);

  auto *causal_model = dynamic_cast<causallm::CausalLM *>(loaded.model.get());
  if (!causal_model) {
    throw std::invalid_argument("loaded model does not support chat streaming");
  }

  CausalRunGuard run_guard(causal_model);
  causallm::openai::StreamingTextFilter filter(options.stop,
                                               loaded.enable_thinking);
  bool client_open = true;
  bool stop_requested = false;

  causal_model->resetGenerationState();
  causal_model->setGenerationOverrides(toGenerationOverrides(options));

  client_open = sendSseJson(client, causallm::openai::makeChatCompletionChunk(
                                      loaded.model_id, "assistant", "", ""));
  if (!client_open)
    return;

  causal_model->setOnToken([&](const std::string &piece) {
    auto result = filter.push(piece);
    if (!result.text.empty()) {
      client_open =
        sendSseJson(client, causallm::openai::makeChatCompletionChunk(
                              loaded.model_id, "", result.text, ""));
    }
    stop_requested = stop_requested || result.stop;
    return client_open && !stop_requested;
  });

  loaded.model->run(prompt, requestDoSample(loaded.do_sample, options),
                    loaded.system_head_prompt, loaded.system_tail_prompt,
                    false);

  if (client_open && !stop_requested) {
    const std::string tail = filter.flush();
    if (!tail.empty()) {
      client_open =
        sendSseJson(client, causallm::openai::makeChatCompletionChunk(
                              loaded.model_id, "", tail, ""));
    }
  }

  if (!client_open)
    return;

  // A stop marker is a normal stream completion, so still emit terminal chunks.
  client_open = sendSseJson(client, causallm::openai::makeChatCompletionChunk(
                                      loaded.model_id, "", "", "stop"));
  if (client_open)
    sendSseDone(client);
}

/**
 * @brief Verify that a request targets the loaded model.
 */
void verifyRequestedModel(const LoadedModel &loaded,
                          const std::string &requested_model) {
  if (requested_model != loaded.model_id) {
    throw std::invalid_argument("model is not loaded: " + requested_model);
  }
}

/**
 * @brief Handle one HTTP request and write the HTTP response.
 */
void handleRequest(LoadedModel &loaded, const HttpRequest &request,
                   SocketHandle client) {
  try {
    if (request.method == "GET" && request.path == "/health") {
      sendAll(client, httpJson(200, "OK", {{"status", "ok"}}));
      return;
    }

    if (request.method == "GET" && request.path == "/v1/models") {
      sendAll(client, httpJson(200, "OK",
                               {{"object", "list"},
                                {"data",
                                 {{{"id", loaded.model_id},
                                   {"object", "model"},
                                   {"created", 0},
                                   {"owned_by", "nntrainer"}}}}}));
      return;
    }

    if (request.method != "POST") {
      sendAll(client,
              httpJson(405, "Method Not Allowed",
                       causallm::openai::makeErrorResponse(
                         "method is not allowed", "invalid_request_error")));
      return;
    }

    const json payload =
      json::parse(request.body.empty() ? "{}" : request.body);
    if (request.path == "/v1/chat/completions") {
      const auto parsed = causallm::openai::parseChatCompletionRequest(payload);
      verifyRequestedModel(loaded, parsed.model);
      const std::string prompt =
        loaded.chat_template.has_value()
          ? loaded.chat_template->apply(withServerChatTemplateOptions(
              parsed.chat_input, loaded.enable_thinking))
          : fallbackPrompt(parsed.messages);
      if (parsed.stream) {
        if (!sendAll(client, httpSseHeader()))
          return;

        try {
          streamPrompt(loaded, prompt, parsed.options, client);
        } catch (const std::exception &e) {
          sendSseJson(client, causallm::openai::makeErrorResponse(
                                e.what(), "server_error"));
          sendSseDone(client);
        }
        return;
      }

      auto [content, metrics] = runPrompt(loaded, prompt, parsed.options);
      sendAll(client,
              httpJson(200, "OK",
                       causallm::openai::makeChatCompletionResponse(
                         loaded.model_id, content, metrics.prefill_tokens,
                         metrics.generation_tokens)));
      return;
    }

    if (request.path == "/v1/completions") {
      const auto parsed = causallm::openai::parseCompletionRequest(payload);
      verifyRequestedModel(loaded, parsed.model);
      if (parsed.stream) {
        throw std::invalid_argument(
          "streaming text completions are not supported");
      }

      auto [content, metrics] =
        runPrompt(loaded, parsed.prompt, parsed.options);
      sendAll(client,
              httpJson(200, "OK",
                       causallm::openai::makeCompletionResponse(
                         loaded.model_id, content, metrics.prefill_tokens,
                         metrics.generation_tokens)));
      return;
    }

    sendAll(client, httpJson(404, "Not Found",
                             causallm::openai::makeErrorResponse(
                               "unknown endpoint", "invalid_request_error")));
  } catch (const json::parse_error &e) {
    sendAll(client, httpJson(400, "Bad Request",
                             causallm::openai::makeErrorResponse(
                               std::string("invalid JSON: ") + e.what(),
                               "invalid_request_error")));
  } catch (const std::invalid_argument &e) {
    sendAll(client, httpJson(400, "Bad Request",
                             causallm::openai::makeErrorResponse(
                               e.what(), "invalid_request_error")));
  } catch (const std::exception &e) {
    sendAll(client, httpJson(500, "Internal Server Error",
                             causallm::openai::makeErrorResponse(
                               e.what(), "server_error")));
  }
}

/**
 * @brief Create and bind a listening socket.
 */
SocketHandle listenSocket(const std::string &host, unsigned short port) {
  SocketHandle server = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (server == invalid_socket_handle)
    throw std::runtime_error("failed to create socket");

  int opt = 1;
#ifdef _WIN32
  setsockopt(server, SOL_SOCKET, SO_REUSEADDR,
             reinterpret_cast<const char *>(&opt), sizeof(opt));
#else
  setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(port);
  if (inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1) {
    closeSocket(server);
    throw std::runtime_error("host must be an IPv4 address");
  }

  if (bind(server, reinterpret_cast<sockaddr *>(&address), sizeof(address)) !=
      0) {
    closeSocket(server);
    throw std::runtime_error("failed to bind server socket");
  }

  if (listen(server, 16) != 0) {
    closeSocket(server);
    throw std::runtime_error("failed to listen on server socket");
  }

  return server;
}

/**
 * @brief Print command usage.
 */
void printUsage(const char *program) {
  std::cerr << "Usage: " << program
            << " <model_path> [--host 127.0.0.1] [--port 8000]"
               " [--model qwen3-0.6b] [--enable-thinking]\n";
}

/**
 * @brief Process Ctrl+C where supported.
 */
void handleSignal(int) { keep_running.store(false); }

} // namespace

/**
 * @brief Entry point for the OpenAI-compatible CausalLM server.
 */
int main(int argc, char **argv) {
  if (argc < 2) {
    printUsage(argv[0]);
    return EXIT_FAILURE;
  }

  std::string model_path = argv[1];
  std::string host = "127.0.0.1";
  unsigned short port = 8000;
  std::string model_id;
  bool enable_thinking = false;

  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--host" && i + 1 < argc) {
      host = argv[++i];
    } else if (arg == "--port" && i + 1 < argc) {
      try {
        const int parsed_port = std::stoi(argv[++i]);
        if (parsed_port < 1 || parsed_port > 65535)
          throw std::out_of_range("port");
        port = static_cast<unsigned short>(parsed_port);
      } catch (const std::exception &) {
        std::cerr << "Invalid --port value\n";
        return EXIT_FAILURE;
      }
    } else if (arg == "--model" && i + 1 < argc) {
      model_id = argv[++i];
    } else if (arg == "--enable-thinking") {
      enable_thinking = true;
    } else {
      printUsage(argv[0]);
      return EXIT_FAILURE;
    }
  }

#ifdef _WIN32
  WSADATA wsa_data{};
  if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
    std::cerr << "WSAStartup failed\n";
    return EXIT_FAILURE;
  }
#endif

  std::signal(SIGINT, handleSignal);
#if !defined(_WIN32)
  std::signal(SIGPIPE, SIG_IGN);
#endif

  try {
    auto loaded = loadModel(model_path, model_id, enable_thinking);
    SocketHandle server = listenSocket(host, port);
    std::cout << "nntrainer OpenAI-compatible server listening on http://"
              << host << ':' << port << "\n"
              << "model: " << loaded->model_id << "\n";

    while (keep_running.load()) {
      sockaddr_in client_addr{};
#ifdef _WIN32
      int client_len = sizeof(client_addr);
#else
      socklen_t client_len = sizeof(client_addr);
#endif
      SocketHandle client =
        accept(server, reinterpret_cast<sockaddr *>(&client_addr), &client_len);
      if (client == invalid_socket_handle) {
        if (!keep_running.load())
          break;
        continue;
      }

      try {
        const auto request = readHttpRequest(client);
        handleRequest(*loaded, request, client);
      } catch (const std::exception &e) {
        sendAll(client, httpJson(400, "Bad Request",
                                 causallm::openai::makeErrorResponse(
                                   e.what(), "invalid_request_error")));
      }
      closeSocket(client);
    }

    closeSocket(server);
  } catch (const std::exception &e) {
    std::cerr << "[!] FATAL ERROR: " << e.what() << "\n";
#ifdef _WIN32
    WSACleanup();
#endif
    return EXIT_FAILURE;
  }

#ifdef _WIN32
  WSACleanup();
#endif
  return EXIT_SUCCESS;
}
