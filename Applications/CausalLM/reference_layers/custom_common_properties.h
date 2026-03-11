// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file   custom_common_properties.h
 * @date   05 April 2024
 * @brief  This file contains list of custom common properties widely
 * used across custom layers
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __CUSTOM_COMMON_PROPERTIES_H__
#define __CUSTOM_COMMON_PROPERTIES_H__

#include <base_properties.h>
#include <climits>

namespace custom {

namespace props {

/**
 * @brief indicated this layer is for smart reply application
 *
 */
class SmartReply : public nntrainer::Property<bool> {
public:
  /**
   * @brief Construct a new SmartReply object
   *
   */
  SmartReply(bool value = false) { set(value); }
  static constexpr const char *key = "smart_reply";
  using prop_tag = nntrainer::bool_prop_tag;
};

/**
 * @brief NumHeads property, NumHeads is number of head in multi head attention
 * of Q
 *
 */
class NumHeads_KV : public nntrainer::PositiveIntegerProperty {
public:
  /**
   * @brief Construct a new NumHeads object with default value 1
   *
   */
  NumHeads_KV(unsigned int value = 1) { set(value); };
  static constexpr const char *key =
    "num_heads_KV";                          /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};

/**
 * @brief LoRA alpha parameter
 * @details It is used to set the scaling factor of LoRA, which is calculated as
 * `scaling = alpha / rank` in the original paper.
 */
class LoraAlpha : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "lora_alpha"; /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag;       /**< property type */
};

/**
 * @brief indicated whether do vocab selection or not
 *
 */
class UseVocabSelection : public nntrainer::Property<bool> {
public:
  /**
   * @brief Construct a new UseVocabSelection object
   *
   */
  UseVocabSelection(bool value = false) { set(value); }
  static constexpr const char *key = "use_vocab_selection";
  using prop_tag = nntrainer::bool_prop_tag;
};

/**
 * @brief LshChoices property, indicate how many words will be choose
 *
 */
class LshChoices : public nntrainer::PositiveIntegerProperty {
public:
  /**
   * @brief Construct a new LshChoices object with a default value 1
   *
   */
  LshChoices(unsigned int value = 1) { set(value); };
  static constexpr const char *key = "lsh_choices"; /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag;        /**< property type */
};

class LoraEnable : public nntrainer::Property<bool> {
public:
  static constexpr const char *key = "lora_enable";
  using prop_tag = nntrainer::bool_prop_tag;
};

class LocalWindowSize : public nntrainer::Property<unsigned int> {
public:
  LocalWindowSize(unsigned int value = UINT_MAX) { set(value); };
  static constexpr const char *key =
    "local_window_size";                     /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};

class MaxNewTokens : public nntrainer::Property<unsigned int> {
public:
  MaxNewTokens(unsigned int value = 1) { set(value); };
  static constexpr const char *key =
    "max_new_tokens";                        /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};

/**
 * @brief Theta property for ROPE
 *
 */
class Theta : public nntrainer::Property<float> {
public:
  static constexpr const char *key = "theta"; /**< unique key to access */
  using prop_tag = nntrainer::float_prop_tag; /**< property type */
};

/**
 * @brief Whether the layer should skip prefill or not.
 *
 */
class SkipPrefill : public nntrainer::Property<bool> {
public:
  /**
   * @brief Construct a new SkipPrefill object
   *
   */
  SkipPrefill(bool value = false) { set(value); }
  static constexpr const char *key = "skip_prefill";
  using prop_tag = nntrainer::bool_prop_tag;
};

} // namespace props
} // namespace custom

#endif /* __CUSTOM_COMMON_PROPERTIES_H__ */
