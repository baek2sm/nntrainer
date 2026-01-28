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

  // Input layer: 3x224x224 (same as timm_vit)
  layers.push_back(
    createLayer("input", {nntrainer::withKey("name", "input0"),
                          nntrainer::withKey("input_shape", "3:224:224")}));

  // Single Conv2D layer: 3 in, 768 out, 16x16 kernel (same as timm_vit)
  // With kernel=16, stride=16, and padding=0, 224x224 input -> 14x14 output
  layers.push_back(
    createLayer("conv2d", {nntrainer::withKey("name", "conv1"),
                           nntrainer::withKey("filters", 768),
                           nntrainer::withKey("kernel_size", {16, 16}),
                           nntrainer::withKey("stride", {16, 16}),
                           nntrainer::withKey("padding", "0,0"),
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
    model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);

    std::cout << "Model created successfully" << std::endl;

    // Load PyTorch weights
    std::string weight_file = "../PyTorch/conv2d_weights.bin";
    std::cout << "\n[Step 2] Loading weights from " << weight_file << std::endl;
    model->load(weight_file, ml::train::ModelFormat::MODEL_FORMAT_BIN);
    std::cout << "Weights loaded successfully" << std::endl;

    // Create all-ones input (NCHW format: 1, 3, 224, 224) - same as timm_vit
    std::cout << "\n[Step 3] Creating all-ones input..." << std::endl;
    const int batch_size = 1;
    const int channels = 3;
    const int height = 224;
    const int width = 224;
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
    // Output shape with kernel=16, stride=16: ((224-16)//16)+1 = 14x14
    const int out_channels = 768; // Output channels (filters)
    const int out_height = 14;
    const int out_width = 14;
    const int output_size = batch_size * out_channels * out_height * out_width;
    std::cout << "Output shape: [" << batch_size << ", " << out_channels << ", "
              << out_height << ", " << out_width << "]" << std::endl;

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
