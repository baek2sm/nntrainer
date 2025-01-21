// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   onnx_interpreter.h
 * @date   12 February 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is onnx converter interface for c++ API
 */

#ifndef __ONNX_INTERPRETER_H__
#define __ONNX_INTERPRETER_H__
#ifdef ENABLE_ONNX_INTERPRETER

#include <app_context.h>
#include <interpreter.h>
#include <layer.h>
#include <layer_node.h>
#include <model.h>
#include <onnx.pb.h>
#include <string>

/**
 * @brief make "key=value" from key and value
 *
 * @tparam T type of a value
 * @param key key
 * @param value value
 * @return std::string with "key=value"
 */
template <typename T>
static std::string withKey(const std::string &key, const T &value) {
  std::stringstream ss;
  ss << key << "=" << value;
  return ss.str();
}

/**
 * @brief make "key=value1,value2,...valueN" from key and multiple values
 *
 * @tparam T type of a value
 * @param key key
 * @param value list of values
 * @return std::string with "key=value1,value2,...valueN"
 */
template <typename T>
static std::string withKey(const std::string &key,
                           std::initializer_list<T> value) {
  if (std::empty(value)) {
    throw std::invalid_argument("empty data cannot be converted");
  }
  std::stringstream ss;
  ss << key << "=";
  auto iter = value.begin();
  for (; iter != value.end() - 1; ++iter) {
    ss << *iter << ',';
  }
  ss << *iter;
  return ss.str();
}

namespace nntrainer {
/**
 * @brief ONNX Interpreter class for converting onnx model to nntrainer model.
 *
 */
class ONNXInterpreter {
public:
  /**
   * @brief Construct a new ONNXInterpreter object
   *
   */
  ONNXInterpreter(){};

  /**
   * @brief Destroy the ONNXInterpreter object
   *
   */
  ~ONNXInterpreter(){};

  /**
   * @brief Load onnx model from given path and convert to nntrainer model.
   *
   * @param path path of onnx model file.
   */
  std::unique_ptr<ml::train::Model> load(std::string path);

private:
  onnx::ModelProto onnx_model; // parsed onnx model
  std::unique_ptr<ml::train::Model>
    nntrainer_model; // converted nntrainer model
  std::unordered_map<std::string, std::string>
    layerOutputMap; // key: name of output, value: name of layer
  std::unordered_map<std::string, onnx::TensorProto>
    initializers; // initializers are used to identify weights

  /**
   * @brief Clean the name of the layer to be used in nntrainer model
   *
   * @param name name of the layer
   */
  std::string cleanName(std::string name);

  /**
   * @brief Transform dimension string to nntrainer's format.
   *
   * @param shape ONNX TensorShapeProto
   */
  std::string transformDimString(onnx::TensorShapeProto shape);

  /**
   * @brief Transform dimension string to nntrainer's format.
   *
   * @param initializer ONNX TensorProto
   */
  std::string transformDimString(onnx::TensorProto initializer);
};
} // namespace nntrainer

#endif // ENABLE_ONNX_INTERPRETER
#endif // __ONNX_INTERPRETER_H__
