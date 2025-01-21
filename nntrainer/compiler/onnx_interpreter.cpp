// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   onnx_interpreter.cpp
 * @date   12 February 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is onnx converter interface for c++ API
 */

#include "onnx_interpreter.h"

namespace nntrainer {
namespace ONNXInterpreter {

std::unique_ptr<ml::train::Model> load(std::string path) {
  // Load and parse onnx file with protobuf
  std::ifstream file(path, std::ios::binary);
  onnx_model.ParseFromIstream(&file);

  // Create nntrainer model instance
  nntrainer_model = ml::train::createModel();
  std::vector<std::shared_ptr<ml::train::Layer>> layers;

  // Create initializer(weight) unordered map and create weight layer
  for (auto &initializer : onnx_model.graph().initializer()) {
    // initializers are used to identify weights in the model
    initializers.insert({cleanName(initializer.name()), initializer});
    std::string dim = transformDimString(initializer);

    // weight layer should be modified not to use input_shape as a parameter
    layers.push_back(ml::train::createLayer(
      "weight", {withKey("name", cleanName(initializer.name())),
                 withKey("dim", dim), withKey("input_shape", dim)}));
  }

  // Create input & constant tensor layer
  for (const auto &input : onnx_model.graph().input()) {
    auto shape = input.type().tensor_type().shape();
    if (shape.dim_size() >= 4 || shape.dim_size() == 0) {
      throw std::runtime_error(
        "Tensors with batch dimensions of 5 or more, or zero_dimensional "
        "tensors are not supported.");
    }

    std::string dim = transformDimString(shape);
    if (input.name().find("input") != std::string::npos) { // Create input layer
      layers.push_back(ml::train::createLayer(
        "input", {withKey("name", cleanName(input.name())),
                  withKey("input_shape", dim)}));
    } else { // Create constant tensor layer
      throw std::runtime_error("Constant tensors are not supported yet.");
    }
  }

  // Create graph
  for (const auto &node : onnx_model.graph().node()) {
    /**
     * @brief While NNTrainer represents graphs as connections between
     * operations, ONNX represents graphs as connections between operations
     * and tensors, requiring remapping of the names of output tensors from
     * operations.
     */
    std::vector<std::string> inputNames;
    auto outputRemap = [&layerOutputMap](std::string &input_layer_name) {
      if (layerOutputMap.find(input_layer_name) != layerOutputMap.end()) {
        input_layer_name = layerOutputMap.find(input_layer_name)->second;
      }
    };
    for (auto &input : node.input()) {
      std::string inputName = cleanName(input);
      outputRemap(inputName);
      inputNames.push_back(inputName);
    }

    if (node.op_type() == "Add") {
      layerOutputMap.insert(
        {cleanName(node.output()[0]), cleanName(node.name())});

      layers.push_back(ml::train::createLayer(
        "add", {"name=" + cleanName(node.name()),
                withKey("input_layers", inputNames[0] + "," + inputNames[1])}));
    } else {
      throw std::runtime_error("Unsupported operation type: " + node.op_type());
    }
  }

  for (auto &layer : layers) {
    nntrainer_model->addLayer(layer);
  }

  return std::move(nntrainer_model);
}

std::string cleanName(std::string name) {
  if (!name.empty() && name[0] == '/') {
    name.erase(0, 1);
  }
  std::replace(name.begin(), name.end(), '/', '_');
  std::replace(name.begin(), name.end(), '.', '_');
  std::transform(name.begin(), name.end(), name.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return name;
}

std::string transformDimString(onnx::TensorShapeProto shape) {
  std::string dim = "";
  for (int i = 0; i < shape.dim_size(); ++i) {
    if (shape.dim()[i].has_dim_param()) {
      throw std::runtime_error("Dynamic dimensions are not supported");
    }
    dim += std::to_string(shape.dim()[i].dim_value());
    if (i < shape.dim_size() - 1) {
      dim += ":";
    }
  }

  if (shape.dim_size() == 1) {
    dim = "1:1:" + dim;
  } else if (shape.dim_size() == 2) {
    dim = "1:" + dim;
  }

  return dim;
}

std::string transformDimString(onnx::TensorProto initializer) {
  std::string dim = "";
  for (int i = 0; i < initializer.dims_size(); ++i) {
    dim += std::to_string(initializer.dims()[i]);
    if (i < initializer.dims_size() - 1) {
      dim += ":";
    }
  }

  if (initializer.dims_size() == 1) {
    dim = "1:1:" + dim;
  } else if (initializer.dims_size() == 2) {
    dim = "1:" + dim;
  }

  return dim;
}

} // namespace ONNXInterpreter
} // namespace nntrainer
