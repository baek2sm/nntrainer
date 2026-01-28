// SPDX-License-Identifier: Apache-2.0
/**
 * @file   main.cpp
 * @brief  SimpleConv2D application - demonstrates PyTorch to NNTrainer workflow
 * @see    https://github.com/nntrainer/nntrainer
 */

#include <iomanip>
#include <iostream>
#include <vector>

#include <layer.h>
#include <model.h>
#include <util_func.h>

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;

std::vector<LayerHandle> createModel() {
  using ml::train::createLayer;

  std::vector<LayerHandle> layers;

  // Input layer: 3x32x32
  layers.push_back(
    createLayer("input", {nntrainer::withKey("name", "input0"),
                          nntrainer::withKey("input_shape", "3:32:32")}));

  // Single Conv2D layer: 3 in, 3 out, 16x16 kernel
  layers.push_back(
    createLayer("conv2d", {nntrainer::withKey("name", "conv1"),
                           nntrainer::withKey("filters", 3),
                           nntrainer::withKey("kernel_size", {16, 16}),
                           nntrainer::withKey("stride", {1, 1}),
                           nntrainer::withKey("padding", "same"),
                           nntrainer::withKey("input_layers", "input0")}));

  return layers;
}

int main(int argc, char *argv[]) {
  std::cout << "==========================================" << std::endl;
  std::cout << "SimpleConv2D: NNTrainer Inference" << std::endl;
  std::cout << "==========================================" << std::endl;

  try {
    // Create and compile model
    std::cout << "\n[Step 1] Creating model..." << std::endl;
    ModelHandle model =
      ml::train::createModel(ml::train::ModelType::NEURAL_NET);

    for (auto &layer : createModel()) {
      model->addLayer(layer);
    }

    model->setProperty({nntrainer::withKey("batch_size", 1)});
    model->compile(ml::train::ExecutionMode::INFERENCE);
    model->initialize(ml::train::ExecutionMode::INFERENCE);

    std::cout << "Model created successfully" << std::endl;

    // Load PyTorch weights
    std::string weight_file = "../PyTorch/conv2d_weights.bin";
    std::cout << "\n[Step 2] Loading weights from " << weight_file << std::endl;
    model->load(weight_file, ml::train::ModelFormat::MODEL_FORMAT_BIN);
    std::cout << "Weights loaded successfully" << std::endl;

    // Create all-ones input (NCHW format: 1, 3, 32, 32)
    std::cout << "\n[Step 3] Creating all-ones input..." << std::endl;
    const int batch_size = 1;
    const int channels = 3;
    const int height = 32;
    const int width = 32;
    const int input_size = batch_size * channels * height * width;

    std::vector<float> input(input_size, 1.0f); // All ones

    std::vector<float *> inputs;
    inputs.push_back(input.data());

    std::cout << "Input shape: [" << batch_size << ", " << channels << ", "
              << height << ", " << width << "]" << std::endl;

    // Run inference
    std::cout << "\n[Step 4] Running inference..." << std::endl;
    auto outputs = model->inference(batch_size, inputs);

    auto output = outputs[0];
    const int output_size = batch_size * channels * height * width;
    std::cout << "Output shape: [" << batch_size << ", " << channels << ", "
              << height << ", " << width << "]" << std::endl;

    // Print comparison values
    std::cout << "\n[Step 5] Comparison values:" << std::endl;
    std::cout << std::fixed << std::setprecision(6);

    std::cout << "nntrainer_input[0:5] = [";
    for (int i = 0; i < 5 && i < input_size; i++) {
      std::cout << input[i];
      if (i < 4)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "nntrainer_output[0:5] = [";
    for (int i = 0; i < 5 && i < output_size; i++) {
      std::cout << output[i];
      if (i < 4)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "\n==========================================" << std::endl;
    std::cout << "Inference completed successfully!" << std::endl;
    std::cout << "Compare these values with PyTorch output." << std::endl;
    std::cout << "==========================================" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
