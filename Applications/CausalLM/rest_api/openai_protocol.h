// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   openai_protocol.h
 * @date   27 May 2026
 * @brief  OpenAI-compatible CausalLM JSON protocol helpers
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __CAUSALLM_OPENAI_PROTOCOL_H__
#define __CAUSALLM_OPENAI_PROTOCOL_H__

#include <json.hpp>

#include <optional>
#include <string>
#include <vector>

/**
 * @brief Namespace for OpenAI-compatible CausalLM protocol helpers
 */
namespace causallm::openai {

/**
 * @brief OpenAI-style chat message used by protocol parsing.
 */
struct ChatMessage {
  std::string role;    /**< Message role, such as system, user, assistant */
  std::string content; /**< Plain text message content */
};

/**
 * @brief Optional generation settings parsed from an OpenAI-compatible request
 */
struct GenerationOptions {
  std::optional<float> temperature;       /**< Per-request temperature */
  std::optional<float> top_p;             /**< Per-request nucleus sampling */
  std::optional<unsigned int> top_k;      /**< Per-request top-k sampling */
  std::optional<unsigned int> max_tokens; /**< Per-request generation limit */
  std::vector<std::string> stop;          /**< Stop marker strings */
};

/**
 * @brief Parsed OpenAI-compatible chat completion request
 */
struct ChatCompletionRequest {
  std::string model;                 /**< Requested model id */
  std::vector<ChatMessage> messages; /**< Conversation messages */
  nlohmann::json chat_input;         /**< Template-ready chat request */
  bool stream = false;               /**< Whether to stream SSE chunks */
  GenerationOptions options;         /**< Optional generation settings */
};

/**
 * @brief Parsed OpenAI-compatible text completion request
 */
struct CompletionRequest {
  std::string model;         /**< Requested model id */
  std::string prompt;        /**< Prompt text */
  bool stream = false;       /**< Whether streaming was requested */
  GenerationOptions options; /**< Optional generation settings */
};

/**
 * @brief Output from one incremental streaming text filter push.
 */
struct StreamingFilterResult {
  std::string text;  /**< Clean text safe to send to the client */
  bool stop = false; /**< Whether a stop marker ended the stream */
};

/**
 * @brief Parse a /v1/chat/completions request body.
 * @param payload JSON request body
 * @return Parsed request
 * @throws std::invalid_argument on unsupported or malformed payloads
 */
ChatCompletionRequest parseChatCompletionRequest(const nlohmann::json &payload);

/**
 * @brief Parse a /v1/completions request body.
 * @param payload JSON request body
 * @return Parsed request
 * @throws std::invalid_argument on unsupported or malformed payloads
 */
CompletionRequest parseCompletionRequest(const nlohmann::json &payload);

/**
 * @brief Remove reasoning tags, end markers, and surrounding whitespace.
 * @param generated Raw generated text
 * @return Clean assistant-visible text
 */
std::string cleanGeneratedText(std::string generated);

/**
 * @brief Build a chat completion response body.
 */
nlohmann::json makeChatCompletionResponse(const std::string &model,
                                          const std::string &content,
                                          unsigned int prompt_tokens,
                                          unsigned int completion_tokens);

/**
 * @brief Build a chat completion streaming chunk body.
 */
nlohmann::json makeChatCompletionChunk(const std::string &model,
                                       const std::string &role,
                                       const std::string &content,
                                       const std::string &finish_reason);

/**
 * @brief Build a text completion response body.
 */
nlohmann::json makeCompletionResponse(const std::string &model,
                                      const std::string &content,
                                      unsigned int prompt_tokens,
                                      unsigned int completion_tokens);

/**
 * @brief Build an OpenAI-style error response body.
 */
nlohmann::json makeErrorResponse(const std::string &message,
                                 const std::string &type,
                                 const std::string &code = "");

/**
 * @brief Incrementally clean model output for SSE streaming.
 */
class StreamingTextFilter {
public:
  /**
   * @brief Construct a streaming filter.
   * @param stop_markers User-provided stop markers
   * @param enable_thinking Whether <think> blocks should be forwarded
   */
  explicit StreamingTextFilter(std::vector<std::string> stop_markers = {},
                               bool enable_thinking = false);

  /**
   * @brief Push a raw decoded piece and return clean visible text.
   */
  StreamingFilterResult push(const std::string &piece);

  /**
   * @brief Flush any non-marker pending text after generation ends.
   */
  std::string flush();

private:
  std::vector<std::string> stop_markers_;
  std::vector<std::string> marker_prefixes_;
  std::string pending_;
  bool enable_thinking_;
  bool in_think_ = false;
  bool stopped_ = false;
};

} // namespace causallm::openai

#endif // __CAUSALLM_OPENAI_PROTOCOL_H__
