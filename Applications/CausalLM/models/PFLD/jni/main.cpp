// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   main.cpp
 * @date   03 July 2026
 * @brief  Face landmark inference with PFLD-like backbone
 * @see    https://github.com/nntrainer/nntrainer
 */

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <model.h>
#include <optimizer.h>
#include <tensor_api.h>
#include <util_func.h>

#include <app_context.h>
#include <engine.h>

using ml::train::createLayer;
using ml::train::LayerHandle;
using ml::train::Tensor;
using ModelHandle = std::unique_ptr<ml::train::Model>;

namespace {

constexpr unsigned int INPUT_H = 128;
constexpr unsigned int INPUT_W = 128;
constexpr unsigned int INPUT_C = 1;
constexpr unsigned int NUM_LANDMARKS = 13 * 2 * 14;

constexpr char PRELU_PLUGIN_NAME[] = "libprelu_layer.so";

std::string ks(int k) { return std::to_string(k) + "," + std::to_string(k); }

std::string stride(int s) {
  return std::to_string(s) + "," + std::to_string(s);
}

std::string pad(int p) { return std::to_string(p) + "," + std::to_string(p); }

LayerHandle conv2d(const std::string &name, int filters, int kernel, int s,
                   const std::string &padding, const std::string &activation,
                   bool bias, int groups) {
  std::vector<std::string> props = {
    "name=" + name, "filters=" + std::to_string(filters),
    "kernel_size=" + ks(kernel), "stride=" + stride(s), "padding=" + padding};
  if (groups > 1) {
    props.push_back("groups=" + std::to_string(groups));
  }
  if (!activation.empty()) {
    props.push_back("activation=" + activation);
  }
  if (!bias) {
    props.push_back("disable_bias=true");
  }
  return LayerHandle(createLayer("conv2d", props));
}

LayerHandle batchNorm(const std::string &name,
                      const std::string &activation = "") {
  std::vector<std::string> props = {"name=" + name, "momentum=0.9",
                                    "epsilon=0.00001"};
  if (!activation.empty()) {
    props.push_back("activation=" + activation);
  }
  return LayerHandle(createLayer("batch_normalization", props));
}

LayerHandle prelu(const std::string &name) {
  return LayerHandle(createLayer("prelu", {"name=" + name}));
}

LayerHandle relu(const std::string &name) {
  return LayerHandle(
    createLayer("activation", {"name=" + name, "activation=relu"}));
}

LayerHandle sigmoid(const std::string &name) {
  return LayerHandle(
    createLayer("activation", {"name=" + name, "activation=sigmoid"}));
}

LayerHandle avgPool(const std::string &name, int pool, int s) {
  std::vector<std::string> props = {"name=" + name, "pooling=average"};
  if (pool > 0) {
    props.push_back("pool_size=" + ks(pool));
    props.push_back("stride=" + stride(s));
  } else {
    props.push_back("pooling=global_average");
  }
  return LayerHandle(createLayer("pooling2d", props));
}

LayerHandle flatten(const std::string &name) {
  return LayerHandle(createLayer("flatten", {"name=" + name}));
}

LayerHandle fc(const std::string &name, int unit,
               const std::string &activation = "") {
  std::vector<std::string> props = {"name=" + name,
                                    "unit=" + std::to_string(unit)};
  if (!activation.empty()) {
    props.push_back("activation=" + activation);
  }
  return LayerHandle(createLayer("fully_connected", props));
}

Tensor mdconv(const std::string &prefix, Tensor input, int in_c,
              const std::vector<int> &splits, const std::vector<int> &kernels,
              const std::vector<std::string> &names, int s) {
  NNTR_THROW_IF(splits.size() > names.size(), std::invalid_argument)
    << "mdconv needs a name per split";
  NNTR_THROW_IF(splits.size() != kernels.size(), std::invalid_argument)
    << "mdconv needs a kernel per split";
  NNTR_THROW_IF(std::accumulate(splits.begin(), splits.end(), 0) != in_c,
                std::invalid_argument)
    << "mdconv split sums must equal input channels";

  /**
   * PyTorch MDConv splits the input tensor along the channel axis and applies
   * an independent depthwise convolution to each slice. nntrainer's split layer
   * only supports even chunks, so we use explicit slice layers for uneven
   * channel splits. The downsampling stride is performed by the depthwise
   * convolutions, not by the preceding 1x1 expansion.
   */
  std::vector<Tensor> parts;
  int start = 0;
  for (size_t i = 0; i < splits.size(); ++i) {
    int c = splits[i];
    int k = kernels[i];
    int end = start + c;

    LayerHandle slice(createLayer(
      "slice", {"name=" + prefix + "_slice_" + std::to_string(i), "axis=1",
                "start_index=" + std::to_string(start + 1),
                "end_index=" + std::to_string(end + 1)}));
    LayerHandle dw(createLayer(
      "conv2d", {"name=" + names[i], "filters=" + std::to_string(c),
                 "kernel_size=" + ks(k), "stride=" + stride(s),
                 "padding=" + pad((k - 1) / 2), "groups=" + std::to_string(c),
                 "disable_bias=true"}));
    parts.push_back(dw(slice(input)));
    start = end;
  }

  LayerHandle concat(
    createLayer("concat", {"name=" + prefix + "_concat", "axis=1"}));
  return concat(parts);
}

Tensor squeezeAndExcite(const std::string &prefix, Tensor input,
                        int in_channels, int reduction) {
  auto se = avgPool(prefix + "_se_gap", 0, 1)(input);
  se = conv2d(prefix + "_se_reduce", reduction, 1, 1, pad(0), "", true, 1)(se);
  se = relu(prefix + "_se_relu")(se);
  se =
    conv2d(prefix + "_se_expand", in_channels, 1, 1, pad(0), "", true, 1)(se);
  se = sigmoid(prefix + "_se_sigmoid")(se);

  return input.multiply(se);
}

Tensor mixedDepthwiseBlock(const std::string &prefix, Tensor input, int in_c,
                           int mid_c, int out_c, int stride,
                           const std::vector<int> &splits,
                           const std::vector<int> &kernels, int reduction,
                           bool residual) {
  auto x = conv2d(prefix + "_expand", mid_c, 1, 1, pad(0), "", true, 1)(input);
  x = prelu(prefix + "_expand_prelu")(x);

  x = mdconv(
    prefix + "_dw", x, mid_c, splits, kernels,
    {prefix + "_dw_0", prefix + "_dw_1", prefix + "_dw_2", prefix + "_dw_3"},
    stride);
  x = batchNorm(prefix + "_dw_bn")(x);
  x = prelu(prefix + "_dw_prelu")(x);

  x = squeezeAndExcite(prefix, x, mid_c, reduction);

  x = conv2d(prefix + "_project", out_c, 1, 1, pad(0), "", true, 1)(x);

  if (residual) {
    Tensor skip = input;
    if (stride != 1) {
      skip = avgPool(prefix + "_skip_pool", stride, stride)(skip);
    }
    x = x.add(skip);
  }
  return x;
}

Tensor buildPFLDGraph(Tensor input) {
  auto x = conv2d("conv1", 28, 3, 2, pad(1), "", true, 1)(input);
  x = prelu("conv1_prelu")(x);
  x = conv2d("conv2_dw", 28, 3, 1, pad(1), "", true, 28)(x);
  x = prelu("conv2_dw_prelu")(x);

  x = mixedDepthwiseBlock("conv_23", x, 28, 48, 28, 2, {24, 12, 12}, {3, 5, 7},
                          14, false);

  for (int i = 0; i < 4; ++i) {
    x = mixedDepthwiseBlock("conv_3_" + std::to_string(i), x, 28, 48, 28, 1,
                            {36, 12}, {3, 5}, 14, true);
  }

  x = mixedDepthwiseBlock("conv_34", x, 28, 96, 48, 2, {48, 24, 24}, {3, 5, 7},
                          14, false);

  for (int i = 0; i < 6; ++i) {
    x = mixedDepthwiseBlock("conv_4_" + std::to_string(i), x, 48, 96, 48, 1,
                            {72, 24}, {3, 5}, 24, true);
  }

  x = mixedDepthwiseBlock("conv_45", x, 48, 192, 48, 2, {48, 48, 48, 48},
                          {3, 5, 7, 9}, 24, false);

  for (int i = 0; i < 2; ++i) {
    x = mixedDepthwiseBlock("conv_5_" + std::to_string(i), x, 48, 96, 48, 1,
                            {32, 32, 32}, {3, 5, 7}, 24, true);
  }

  x = conv2d("block6_2", 32, 1, 1, pad(0), "", true, 1)(x);
  x = prelu("block6_2_prelu")(x);

  auto x1 = avgPool("avg_pool1", 8, 8)(x);

  auto x2 = conv2d("conv7", 64, 3, 2, pad(1), "", true, 1)(x);
  x2 = batchNorm("conv7_bn", "relu")(x2);
  auto x2_pooled = avgPool("avg_pool2", 4, 4)(x2);

  auto x3 = conv2d("conv8", 128, 4, 1, pad(0), "", true, 1)(x2);
  x3 = batchNorm("conv8_bn", "relu")(x3);

  LayerHandle concat(createLayer("concat", {"name=tail_concat", "axis=1"}));
  auto tail = concat({x1, x2_pooled, x3});

  tail = flatten("flatten")(tail);
  tail = fc("fc", NUM_LANDMARKS)(tail);

  return tail;
}

} // namespace

std::string getPluginDir(const std::string &exe_path) {
  std::filesystem::path p(exe_path);
  if (p.has_parent_path()) {
    std::error_code ec;
    auto canonical = std::filesystem::canonical(p, ec);
    if (!ec) {
      return canonical.parent_path().string();
    }
  }
  return std::filesystem::current_path().string();
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0]
              << " <weight_bin> [input.raw] [plugin_dir]\n"
                 "  plugin_dir defaults to the executable's directory.\n";
    return EXIT_FAILURE;
  }

  std::string weight_path = argv[1];
  std::string input_path = (argc > 2) ? argv[2] : "";
  std::string plugin_dir = (argc > 3) ? argv[3] : getPluginDir(argv[0]);

  try {
    auto &engine = nntrainer::Engine::Global();
    auto app_context =
      static_cast<nntrainer::AppContext *>(engine.getRegisteredContext("cpu"));
    app_context->registerLayer(PRELU_PLUGIN_NAME, plugin_dir);
  } catch (std::exception &e) {
    std::cerr << "failed to register prelu layer: " << e.what() << '\n';
    return EXIT_FAILURE;
  }

  auto input = Tensor({1, INPUT_C, INPUT_H, INPUT_W}, "input0");
  auto output = buildPFLDGraph(input);

  ModelHandle model =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET, {"loss=mse"});
  model->setProperty({"batch_size=1", "epochs=1"});

  auto optimizer = ml::train::createOptimizer("adam", {"learning_rate=0.0"});
  if (model->setOptimizer(std::move(optimizer))) {
    throw std::invalid_argument("failed to set optimizer");
  }

  if (model->compile(input, output, ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("model compilation failed");
  }

  model->load(weight_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);

  std::vector<float> input_data(INPUT_C * INPUT_H * INPUT_W);
  if (!input_path.empty()) {
    std::ifstream fs(input_path, std::ios::binary);
    if (!fs) {
      throw std::invalid_argument("cannot open input file " + input_path);
    }
    fs.read(reinterpret_cast<char *>(input_data.data()),
            input_data.size() * sizeof(float));
  } else {
    std::fill(input_data.begin(), input_data.end(), 0.0f);
  }

  std::vector<float *> in = {input_data.data()};
  auto result = model->inference(1, in);
  if (result.empty()) {
    throw std::runtime_error("inference failed");
  }

  std::cout << "landmarks:";
  for (unsigned int i = 0; i < NUM_LANDMARKS; ++i) {
    std::cout << ' ' << result[0][i];
  }
  std::cout << '\n';

  return EXIT_SUCCESS;
}
