// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_causallm_openai_protocol.cpp
 * @date   27 May 2026
 * @brief  OpenAI-compatible CausalLM protocol unit tests
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <openai_protocol.h>

#include <gtest/gtest.h>

TEST(CausalLMOpenAIProtocolTest, ParsesChatCompletionMessages) {
  nlohmann::json payload = {
    {"model", "qwen3-0.6b"},
    {"messages",
     {{{"role", "system"}, {"content", "Be concise."}},
      {{"role", "user"}, {"content", "Say hello."}}}},
  };

  auto request = causallm::openai::parseChatCompletionRequest(payload);

  ASSERT_EQ(request.model, "qwen3-0.6b");
  ASSERT_EQ(request.messages.size(), 2u);
  EXPECT_EQ(request.messages[0].role, "system");
  EXPECT_EQ(request.messages[0].content, "Be concise.");
  EXPECT_EQ(request.messages[1].role, "user");
  EXPECT_EQ(request.messages[1].content, "Say hello.");
  EXPECT_EQ(request.chat_input["messages"], payload["messages"]);
}

TEST(CausalLMOpenAIProtocolTest, PreservesTemplateChatInput) {
  nlohmann::json payload = {
    {"model", "qwen3-0.6b"},
    {"messages",
     {{{"role", "user"},
       {"content",
        {{{"type", "text"}, {"text", "Say "}},
         {{"type", "text"}, {"text", "hello."}}}}}}},
    {"tools",
     {{{"type", "function"},
       {"function", {{"name", "noop"}, {"parameters", {}}}}}}},
    {"add_generation_prompt", false},
  };

  auto request = causallm::openai::parseChatCompletionRequest(payload);

  ASSERT_EQ(request.messages.size(), 1u);
  EXPECT_EQ(request.messages[0].content, "Say hello.");
  EXPECT_EQ(request.chat_input["messages"], payload["messages"]);
  EXPECT_EQ(request.chat_input["tools"], payload["tools"]);
  EXPECT_FALSE(request.chat_input["add_generation_prompt"].get<bool>());
  EXPECT_FALSE(request.chat_input.contains("model"));
  EXPECT_FALSE(request.chat_input.contains("stream"));
}

TEST(CausalLMOpenAIProtocolTest, ParsesStringCompletionPrompt) {
  nlohmann::json payload = {
    {"model", "qwen3-0.6b"},
    {"prompt", "Say hello."},
  };

  auto request = causallm::openai::parseCompletionRequest(payload);

  EXPECT_EQ(request.model, "qwen3-0.6b");
  EXPECT_EQ(request.prompt, "Say hello.");
}

TEST(CausalLMOpenAIProtocolTest, ParsesStreamingRequests) {
  nlohmann::json payload = {
    {"model", "qwen3-0.6b"},
    {"messages", {{{"role", "user"}, {"content", "Hi"}}}},
    {"stream", true},
  };

  auto request = causallm::openai::parseChatCompletionRequest(payload);

  EXPECT_TRUE(request.stream);
}

TEST(CausalLMOpenAIProtocolTest, ParsesStreamingGenerationOptions) {
  nlohmann::json payload = {
    {"model", "qwen3-0.6b"},
    {"messages", {{{"role", "user"}, {"content", "Hi"}}}},
    {"stream", true},
    {"temperature", 0.2},
    {"top_p", 0.75},
    {"max_tokens", 11},
    {"stop", {"<END>", "</custom>"}},
  };

  auto request = causallm::openai::parseChatCompletionRequest(payload);

  EXPECT_TRUE(request.stream);
  ASSERT_TRUE(request.options.temperature.has_value());
  EXPECT_FLOAT_EQ(*request.options.temperature, 0.2f);
  ASSERT_TRUE(request.options.top_p.has_value());
  EXPECT_FLOAT_EQ(*request.options.top_p, 0.75f);
  ASSERT_TRUE(request.options.max_tokens.has_value());
  EXPECT_EQ(*request.options.max_tokens, 11u);
  ASSERT_EQ(request.options.stop.size(), 2u);
  EXPECT_EQ(request.options.stop[0], "<END>");
  EXPECT_EQ(request.options.stop[1], "</custom>");
}

TEST(CausalLMOpenAIProtocolTest, RejectsInvalidGenerationOptions) {
  nlohmann::json zero_tokens = {
    {"model", "qwen3-0.6b"},
    {"messages", {{{"role", "user"}, {"content", "Hi"}}}},
    {"max_tokens", 0},
  };
  nlohmann::json invalid_stream = {
    {"model", "qwen3-0.6b"},
    {"messages", {{{"role", "user"}, {"content", "Hi"}}}},
    {"stream", "true"},
  };
  nlohmann::json huge_tokens = nlohmann::json::parse(
    R"({"model":"qwen3-0.6b","messages":[{"role":"user","content":"Hi"}],"max_tokens":18446744073709551616})");

  EXPECT_THROW(causallm::openai::parseChatCompletionRequest(zero_tokens),
               std::invalid_argument);
  EXPECT_THROW(causallm::openai::parseChatCompletionRequest(invalid_stream),
               std::invalid_argument);
  EXPECT_THROW(causallm::openai::parseChatCompletionRequest(huge_tokens),
               std::invalid_argument);
}

TEST(CausalLMOpenAIProtocolTest, RejectsMissingModel) {
  nlohmann::json payload = {
    {"messages", {{{"role", "user"}, {"content", "Hi"}}}},
  };

  EXPECT_THROW(causallm::openai::parseChatCompletionRequest(payload),
               std::invalid_argument);
}

TEST(CausalLMOpenAIProtocolTest, CleansReasoningAndSpecialEndToken) {
  const std::string generated =
    "<think>\nchecking\n</think>\n\nHello from NNTrainer! <|im_end|>\n";

  EXPECT_EQ(causallm::openai::cleanGeneratedText(generated),
            "Hello from NNTrainer!");
}

TEST(CausalLMOpenAIProtocolTest, BuildsChatCompletionResponseShape) {
  auto response =
    causallm::openai::makeChatCompletionResponse("qwen3-0.6b", "Hello.", 3, 2);

  EXPECT_EQ(response["object"], "chat.completion");
  EXPECT_EQ(response["model"], "qwen3-0.6b");
  EXPECT_EQ(response["choices"][0]["message"]["role"], "assistant");
  EXPECT_EQ(response["choices"][0]["message"]["content"], "Hello.");
  EXPECT_EQ(response["usage"]["prompt_tokens"], 3);
  EXPECT_EQ(response["usage"]["completion_tokens"], 2);
}

TEST(CausalLMOpenAIProtocolTest, BuildsChatCompletionChunkShape) {
  auto role_chunk = causallm::openai::makeChatCompletionChunk(
    "qwen3-0.6b", "assistant", "", "");
  auto content_chunk =
    causallm::openai::makeChatCompletionChunk("qwen3-0.6b", "", "Hello", "");
  auto finish_chunk =
    causallm::openai::makeChatCompletionChunk("qwen3-0.6b", "", "", "stop");

  EXPECT_EQ(role_chunk["object"], "chat.completion.chunk");
  EXPECT_EQ(role_chunk["choices"][0]["delta"]["role"], "assistant");
  EXPECT_EQ(role_chunk["choices"][0]["finish_reason"], nullptr);
  EXPECT_EQ(content_chunk["choices"][0]["delta"]["content"], "Hello");
  EXPECT_EQ(finish_chunk["choices"][0]["delta"], nlohmann::json::object());
  EXPECT_EQ(finish_chunk["choices"][0]["finish_reason"], "stop");
}

TEST(CausalLMOpenAIProtocolTest, StreamingFilterSuppressesThinkAndStops) {
  causallm::openai::StreamingTextFilter filter({"<STOP>"}, false);

  EXPECT_EQ(filter.push("<thi").text, "");
  EXPECT_EQ(filter.push("nk>hidden</thi").text, "");
  auto visible = filter.push("nk>Hel");
  EXPECT_FALSE(visible.stop);
  EXPECT_EQ(visible.text, "Hel");
  auto stopped = filter.push("lo<ST");
  EXPECT_FALSE(stopped.stop);
  EXPECT_EQ(stopped.text, "lo");
  stopped = filter.push("OP>ignored");
  EXPECT_TRUE(stopped.stop);
  EXPECT_EQ(stopped.text, "");
  EXPECT_EQ(filter.flush(), "");
}

TEST(CausalLMOpenAIProtocolTest, StreamingFilterFlushesPartialStopPrefix) {
  causallm::openai::StreamingTextFilter filter({"<STOP>"}, false);

  auto visible = filter.push("Value <ST");

  EXPECT_EQ(visible.text, "Value ");
  EXPECT_FALSE(visible.stop);
  EXPECT_EQ(filter.flush(), "<ST");
}
