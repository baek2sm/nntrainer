// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   openai_protocol.cpp
 * @date   27 May 2026
 * @brief  OpenAI-compatible CausalLM JSON protocol helpers
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <openai_protocol.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

/**
 * @brief Namespace for OpenAI-compatible CausalLM protocol helpers
 */
namespace causallm::openai {

/**
 * @brief Anonymous namespace for local protocol helper functions
 */
namespace {

/**
 * @brief Return current Unix timestamp in seconds.
 */
long long nowSeconds() {
  return std::chrono::duration_cast<std::chrono::seconds>(
           std::chrono::system_clock::now().time_since_epoch())
    .count();
}

/**
 * @brief Read required string field from a JSON object.
 */
std::string requireString(const nlohmann::json &payload,
                          const std::string &field) {
  if (!payload.contains(field) || !payload[field].is_string()) {
    throw std::invalid_argument("missing or invalid '" + field + "'");
  }
  return payload[field].get<std::string>();
}

/**
 * @brief Read text from a chat content field for non-template fallback.
 */
std::string chatContentText(const nlohmann::json &message) {
  if (!message.contains("content") || message["content"].is_null())
    return "";

  const auto &content = message["content"];
  if (content.is_string())
    return content.get<std::string>();

  if (!content.is_array()) {
    throw std::invalid_argument(
      "message content must be a string or text parts");
  }

  std::string text;
  for (const auto &part : content) {
    if (!part.is_object() || !part.contains("type") ||
        !part["type"].is_string()) {
      throw std::invalid_argument("chat content parts must include a type");
    }

    if (part["type"].get<std::string>() != "text") {
      throw std::invalid_argument(
        "non-text chat content parts are unsupported");
    }

    if (part.contains("text") && part["text"].is_string())
      text += part["text"].get<std::string>();
  }
  return text;
}

/**
 * @brief Build the JSON request forwarded to ChatTemplate.
 */
nlohmann::json makeChatTemplateInput(const nlohmann::json &payload) {
  nlohmann::json input;
  input["messages"] = payload["messages"];

  for (const char *field : {"tools", "functions", "add_generation_prompt",
                            "continue_final_message", "template_name"}) {
    if (payload.contains(field))
      input[field] = payload[field];
  }

  return input;
}

/**
 * @brief Read an optional floating point field.
 */
std::optional<float> optionalFloat(const nlohmann::json &payload,
                                   const std::string &field) {
  if (!payload.contains(field))
    return std::nullopt;
  if (!payload[field].is_number()) {
    throw std::invalid_argument("invalid '" + field + "'");
  }
  try {
    const float value = payload[field].get<float>();
    if (!std::isfinite(value)) {
      throw std::invalid_argument("invalid '" + field + "'");
    }
    return value;
  } catch (const nlohmann::json::exception &) {
    throw std::invalid_argument("invalid '" + field + "'");
  }
}

/**
 * @brief Read an optional boolean field.
 */
bool optionalBool(const nlohmann::json &payload, const std::string &field,
                  bool default_value) {
  if (!payload.contains(field))
    return default_value;
  if (!payload[field].is_boolean()) {
    throw std::invalid_argument("invalid '" + field + "'");
  }
  return payload[field].get<bool>();
}

/**
 * @brief Read an optional positive unsigned integer field.
 */
std::optional<unsigned int> optionalUInt(const nlohmann::json &payload,
                                         const std::string &field) {
  if (!payload.contains(field))
    return std::nullopt;
  if (!payload[field].is_number_integer()) {
    throw std::invalid_argument("invalid '" + field + "'");
  }
  long long value = 0;
  try {
    value = payload[field].get<long long>();
  } catch (const nlohmann::json::exception &) {
    throw std::invalid_argument("invalid '" + field + "'");
  }
  if (value < 1 || value > static_cast<long long>(
                             std::numeric_limits<unsigned int>::max())) {
    throw std::invalid_argument("invalid '" + field + "'");
  }
  return static_cast<unsigned int>(value);
}

/**
 * @brief Read stop field as string or string array.
 */
std::vector<std::string> optionalStop(const nlohmann::json &payload) {
  std::vector<std::string> stop;
  if (!payload.contains("stop") || payload["stop"].is_null())
    return stop;

  if (payload["stop"].is_string()) {
    stop.push_back(payload["stop"].get<std::string>());
    return stop;
  }

  if (!payload["stop"].is_array()) {
    throw std::invalid_argument("invalid 'stop'");
  }

  for (const auto &item : payload["stop"]) {
    if (!item.is_string()) {
      throw std::invalid_argument("invalid 'stop'");
    }
    stop.push_back(item.get<std::string>());
  }
  return stop;
}

/**
 * @brief Parse shared generation options from request JSON.
 */
GenerationOptions parseGenerationOptions(const nlohmann::json &payload) {
  GenerationOptions options;
  options.temperature = optionalFloat(payload, "temperature");
  options.top_p = optionalFloat(payload, "top_p");
  options.top_k = optionalUInt(payload, "top_k");
  options.max_tokens = optionalUInt(payload, "max_tokens");
  options.stop = optionalStop(payload);
  return options;
}

/**
 * @brief Trim ASCII whitespace in place.
 */
void trimInPlace(std::string &text) {
  auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
  text.erase(text.begin(),
             std::find_if_not(text.begin(), text.end(), is_space));
  text.erase(std::find_if_not(text.rbegin(), text.rend(), is_space).base(),
             text.end());
}

/**
 * @brief Erase every occurrence of a marker string.
 */
void eraseAll(std::string &text, const std::string &needle) {
  if (needle.empty())
    return;

  size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    text.erase(pos, needle.size());
  }
}

/**
 * @brief Return whether text starts with marker.
 */
bool startsWith(const std::string &text, const std::string &marker) {
  return text.size() >= marker.size() &&
         text.compare(0, marker.size(), marker) == 0;
}

/**
 * @brief Return whether text is a non-empty prefix of marker.
 */
bool isPartialPrefix(const std::string &text, const std::string &marker) {
  return !text.empty() && text.size() < marker.size() &&
         marker.compare(0, text.size(), text) == 0;
}

/**
 * @brief Build response usage object.
 */
nlohmann::json makeUsage(unsigned int prompt_tokens,
                         unsigned int completion_tokens) {
  return {
    {"prompt_tokens", prompt_tokens},
    {"completion_tokens", completion_tokens},
    {"total_tokens", prompt_tokens + completion_tokens},
  };
}

} // namespace

ChatCompletionRequest
parseChatCompletionRequest(const nlohmann::json &payload) {
  if (!payload.is_object()) {
    throw std::invalid_argument("request body must be a JSON object");
  }

  ChatCompletionRequest request;
  request.model = requireString(payload, "model");
  request.stream = optionalBool(payload, "stream", false);
  request.options = parseGenerationOptions(payload);

  if (!payload.contains("messages") || !payload["messages"].is_array() ||
      payload["messages"].empty()) {
    throw std::invalid_argument("'messages' must be a non-empty array");
  }

  for (const auto &item : payload["messages"]) {
    if (!item.is_object()) {
      throw std::invalid_argument("each message must be a JSON object");
    }

    ChatMessage message;
    message.role = requireString(item, "role");
    message.content = chatContentText(item);
    request.messages.push_back(std::move(message));
  }
  request.chat_input = makeChatTemplateInput(payload);

  return request;
}

CompletionRequest parseCompletionRequest(const nlohmann::json &payload) {
  if (!payload.is_object()) {
    throw std::invalid_argument("request body must be a JSON object");
  }

  CompletionRequest request;
  request.model = requireString(payload, "model");
  request.prompt = requireString(payload, "prompt");
  request.stream = optionalBool(payload, "stream", false);
  request.options = parseGenerationOptions(payload);
  return request;
}

std::string cleanGeneratedText(std::string generated) {
  const std::string think_end = "</think>";
  const size_t think_end_pos = generated.find(think_end);
  if (think_end_pos != std::string::npos) {
    generated.erase(0, think_end_pos + think_end.size());
  }

  const std::vector<std::string> stop_markers = {
    "<|im_end|>",
    "<|endoftext|>",
    "</s>",
  };
  for (const auto &marker : stop_markers) {
    const size_t pos = generated.find(marker);
    if (pos != std::string::npos) {
      generated.erase(pos);
    }
  }

  eraseAll(generated, "<think>");
  eraseAll(generated, "</think>");
  eraseAll(generated, "\xEF\xBF\xBD");
  trimInPlace(generated);
  return generated;
}

nlohmann::json makeChatCompletionResponse(const std::string &model,
                                          const std::string &content,
                                          unsigned int prompt_tokens,
                                          unsigned int completion_tokens) {
  return {
    {"id", "chatcmpl-nntrainer"},
    {"object", "chat.completion"},
    {"created", nowSeconds()},
    {"model", model},
    {"choices",
     {{{"index", 0},
       {"message", {{"role", "assistant"}, {"content", content}}},
       {"finish_reason", "stop"}}}},
    {"usage", makeUsage(prompt_tokens, completion_tokens)},
  };
}

nlohmann::json makeChatCompletionChunk(const std::string &model,
                                       const std::string &role,
                                       const std::string &content,
                                       const std::string &finish_reason) {
  nlohmann::json delta = nlohmann::json::object();
  if (!role.empty())
    delta["role"] = role;
  if (!content.empty())
    delta["content"] = content;

  return {
    {"id", "chatcmpl-nntrainer"},
    {"object", "chat.completion.chunk"},
    {"created", nowSeconds()},
    {"model", model},
    {"choices",
     {{{"index", 0},
       {"delta", delta},
       {"finish_reason", finish_reason.empty()
                           ? nlohmann::json(nullptr)
                           : nlohmann::json(finish_reason)}}}},
  };
}

nlohmann::json makeCompletionResponse(const std::string &model,
                                      const std::string &content,
                                      unsigned int prompt_tokens,
                                      unsigned int completion_tokens) {
  return {
    {"id", "cmpl-nntrainer"},
    {"object", "text_completion"},
    {"created", nowSeconds()},
    {"model", model},
    {"choices", {{{"index", 0}, {"text", content}, {"finish_reason", "stop"}}}},
    {"usage", makeUsage(prompt_tokens, completion_tokens)},
  };
}

nlohmann::json makeErrorResponse(const std::string &message,
                                 const std::string &type,
                                 const std::string &code) {
  nlohmann::json error = {
    {"message", message},
    {"type", type},
    {"param", nullptr},
    {"code", nullptr},
  };
  if (!code.empty())
    error["code"] = code;
  return {{"error", error}};
}

StreamingTextFilter::StreamingTextFilter(std::vector<std::string> stop_markers,
                                         bool enable_thinking) :
  stop_markers_(std::move(stop_markers)), enable_thinking_(enable_thinking) {
  stop_markers_.erase(
    std::remove_if(stop_markers_.begin(), stop_markers_.end(),
                   [](const std::string &marker) { return marker.empty(); }),
    stop_markers_.end());
  stop_markers_.push_back("<|im_end|>");
  stop_markers_.push_back("<|endoftext|>");
  stop_markers_.push_back("</s>");

  marker_prefixes_ = stop_markers_;
  if (!enable_thinking_) {
    marker_prefixes_.push_back("<think>");
    marker_prefixes_.push_back("</think>");
  }
}

StreamingFilterResult StreamingTextFilter::push(const std::string &piece) {
  StreamingFilterResult result;
  if (stopped_)
    return result;

  pending_.append(piece);
  eraseAll(pending_, "\xEF\xBF\xBD");

  while (!pending_.empty()) {
    bool matched_stop = false;
    for (const auto &marker : stop_markers_) {
      if (startsWith(pending_, marker)) {
        stopped_ = true;
        pending_.clear();
        result.stop = true;
        matched_stop = true;
        break;
      }
    }
    if (matched_stop)
      break;

    if (!enable_thinking_ && !in_think_ && startsWith(pending_, "<think>")) {
      pending_.erase(0, 7);
      in_think_ = true;
      continue;
    }

    if (!enable_thinking_ && in_think_ && startsWith(pending_, "</think>")) {
      pending_.erase(0, 8);
      in_think_ = false;
      continue;
    }

    bool waiting_for_marker = false;
    for (const auto &marker : marker_prefixes_) {
      if (isPartialPrefix(pending_, marker)) {
        waiting_for_marker = true;
        break;
      }
    }
    if (waiting_for_marker)
      break;

    const char ch = pending_.front();
    pending_.erase(0, 1);
    if (!in_think_)
      result.text.push_back(ch);
  }

  return result;
}

std::string StreamingTextFilter::flush() {
  if (stopped_ || in_think_) {
    pending_.clear();
    return "";
  }

  std::string text = pending_;
  pending_.clear();
  return text;
}

} // namespace causallm::openai
