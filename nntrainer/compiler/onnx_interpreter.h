// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   protobuf_interpreter.h
 * @date   1 January 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is onnx converter interface for c++ API
 */

#ifndef __ONNX_INTERPRETER_H__
#define __ONNX_INTERPRETER_H__

#include <app_context.h>
#include <fstream>
#include <interpreter.h>
#include <iostream>
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

class ONNXInterpreter {
public:
  ONNXInterpreter(){};
  ~ONNXInterpreter(){};

  std::unique_ptr<ml::train::Model> load(std::string path) {
    // Load and parse onnx file with protobuf
    std::ifstream file(path, std::ios::binary);
    onnx_model.ParseFromIstream(&file);

    // Create graph
    std::vector<std::shared_ptr<ml::train::Layer>> layers;
    model = ml::train::createModel();

    // Create initializer(weight) unordered map and create weight layer
    for (auto &initializer : onnx_model.graph().initializer()) {
      initializers.insert({cleanName(initializer.name()), initializer});
      // Weight of the matmul operation should be created by the fc layer.
      if (initializer.name().find("onnx::MatMul") != std::string::npos)
        continue;
      else {
        // Create the weight layer
        std::string dim = transformDimString(initializer);
        std::cout << "[Create layer] " << cleanName(initializer.name())
                  << ", shape: " << dim << std::endl;
        layers.push_back(ml::train::createLayer(
          "weight", {withKey("name", cleanName(initializer.name())),
                     withKey("dim", dim), withKey("input_shape", dim)}));
      }
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
      if (input.name().find("input") !=
          std::string::npos) { // Create input layer
        std::cout << "[Create layer] " << cleanName(input.name())
                  << ", shape: " << dim << std::endl;
        layers.push_back(ml::train::createLayer(
          "input", {withKey("name", cleanName(input.name())),
                    withKey("input_shape", dim)}));
      } else { // Create constant tensor layer
        throw std::runtime_error("Constant tensors are not supported yet.");
      }
    }

    // Create graph
    for (const auto &node : onnx_model.graph().node()) {
      // std::cout << "\nname: " << node.name() << std::endl;
      // for (int j = 0; j < node.input_size(); ++j) {
      //   std::cout << "  input: " << cleanName(node.input(j)) << std::endl;
      // }
      // for (int j = 0; j < node.output_size(); ++j) {
      //   std::cout << "  output: " << cleanName(node.output(j)) << std::endl;
      // }

      // find the name of the layer that created the input tensor of this
      // layer
      if (node.op_type() == "MatMul") {
        // check if it is a fully connected layer
        if (node.input()[1].find("onnx::MatMul") != std::string::npos) {
          std::string input_layer_name = cleanName(node.input()[0]);
          if (layerNameMap.find(input_layer_name) != layerNameMap.end()) {
            input_layer_name = layerNameMap.find(input_layer_name)->second;
          }
          layerNameMap.insert(
            {cleanName(node.output()[0]), cleanName(node.name())});

          // check if the bias is enabled
          std::string disable_bias = "true";
          // std::string bias_name = cleanName(node.name());
          // bias_name.replace(bias_name.find("MatMul"), 6, "bias");
          // if (initializers.find(bias_name) != initializers.end())
          //   disable_bias = "false";

          // create fc layer
          std::cout << "[Create layer] " << cleanName(node.name())
                    << ", input_layers: " << input_layer_name << std::endl;
          layers.push_back(ml::train::createLayer(
            "fully_connected",
            {withKey("name", cleanName(node.name())),
             withKey("unit",
                     initializers[cleanName(node.input()[1])].dims()[1]),
             withKey("disable_bias",
                     disable_bias), // not supported enable_bias yet
             withKey("input_layers", input_layer_name)}));

        }
        // create matmul operation layer
        else {
          throw std::runtime_error("Matmul layer is not supported yet.");
        }
      } else if (node.op_type() == "Add") {
        std::string input_layer_name0 = cleanName(node.input()[0]);
        if (layerNameMap.find(input_layer_name0) != layerNameMap.end()) {
          input_layer_name0 = layerNameMap.find(input_layer_name0)->second;
        }
        std::string input_layer_name1 = cleanName(node.input()[1]);
        if (layerNameMap.find(input_layer_name1) != layerNameMap.end()) {
          input_layer_name1 = layerNameMap.find(input_layer_name1)->second;
        }
        layerNameMap.insert(
          {cleanName(node.output()[0]), cleanName(node.name())});

        layers.push_back(ml::train::createLayer(
          "add", {"name=" + cleanName(node.name()),
                  withKey("input_layers",
                          input_layer_name0 + "," + input_layer_name1)}));
      } else if (node.op_type() == "Subtract") {
        std::string input_layer_name0 = cleanName(node.input()[0]);
        if (layerNameMap.find(input_layer_name0) != layerNameMap.end()) {
          input_layer_name0 = layerNameMap.find(input_layer_name0)->second;
        }
        std::string input_layer_name1 = cleanName(node.input()[1]);
        if (layerNameMap.find(input_layer_name1) != layerNameMap.end()) {
          input_layer_name1 = layerNameMap.find(input_layer_name1)->second;
        }
        layerNameMap.insert(
          {cleanName(node.output()[0]), cleanName(node.name())});

        layers.push_back(ml::train::createLayer(
          "subtract", {"name=" + cleanName(node.name()),
                       withKey("input_layers",
                               input_layer_name0 + "," + input_layer_name1)}));
      } else if (node.op_type() == "Multiply") {
        std::string input_layer_name0 = cleanName(node.input()[0]);
        if (layerNameMap.find(input_layer_name0) != layerNameMap.end()) {
          input_layer_name0 = layerNameMap.find(input_layer_name0)->second;
        }
        std::string input_layer_name1 = cleanName(node.input()[1]);
        if (layerNameMap.find(input_layer_name1) != layerNameMap.end()) {
          input_layer_name1 = layerNameMap.find(input_layer_name1)->second;
        }
        layerNameMap.insert(
          {cleanName(node.output()[0]), cleanName(node.name())});

        layers.push_back(ml::train::createLayer(
          "multiply", {"name=" + cleanName(node.name()),
                       withKey("input_layers",
                               input_layer_name0 + "," + input_layer_name1)}));
      } else if (node.op_type() == "Divide") {
        std::string input_layer_name0 = cleanName(node.input()[0]);
        if (layerNameMap.find(input_layer_name0) != layerNameMap.end()) {
          input_layer_name0 = layerNameMap.find(input_layer_name0)->second;
        }
        std::string input_layer_name1 = cleanName(node.input()[1]);
        if (layerNameMap.find(input_layer_name1) != layerNameMap.end()) {
          input_layer_name1 = layerNameMap.find(input_layer_name1)->second;
        }
        layerNameMap.insert(
          {cleanName(node.output()[0]), cleanName(node.name())});

        layers.push_back(ml::train::createLayer(
          "divide", {"name=" + cleanName(node.name()),
                     withKey("input_layers",
                             input_layer_name0 + "," + input_layer_name1)}));
      } else if (node.op_type() == "Relu") {
        std::cout << "other layer type: " << node.op_type() << std::endl;
        std::string input_layer_name = cleanName(node.input()[0]);
        if (layerNameMap.find(input_layer_name) != layerNameMap.end()) {
          input_layer_name = layerNameMap.find(input_layer_name)->second;
        }
        layerNameMap.insert(
          {cleanName(node.output()[0]), cleanName(node.name())});
        layers.push_back(ml::train::createLayer(
          "activation", {withKey("name", cleanName(node.name())),
                         withKey("activation", "relu"),
                         withKey("input_layers", input_layer_name)}));
      }
    }

    for (auto &layer : layers) {
      model->addLayer(layer);
    }

    return std::move(model);
  }

private:
  std::unique_ptr<ml::train::Model> model;
  onnx::ModelProto onnx_model;
  std::unordered_map<std::string, std::string>
    layerNameMap; // key: name of output, value: name of layer
  std::unordered_map<std::string, onnx::TensorProto> initializers;

  std::string cleanName(std::string name) {
    // create copy because original string should not be modified
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
};

#endif // __ONNX_INTERPRETER_H__
