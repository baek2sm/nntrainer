#include <iostream>
#include <layer.h>
#include <model.h>
#include <nntrainer-api-common.h>
#include <onnx.h>
#include <optimizer.h>
#include <sstream>
#include <util_func.h>

using ModelHandle = std::unique_ptr<ml::train::Model>;

int main() {
  ModelHandle model = ml::train::loadONNX("../../../../Applications/ONNX/"
                                          "jni/add_example.onnx");

  model->setProperty({nntrainer::withKey("batch_size", 1)});

  try {
    model->compile();
  } catch (const std::exception &e) {
    std::cerr << "Error during compile: " << e.what() << "\n";
    return 1;
  }

  try {
    model->initialize();
  } catch (const std::exception &e) {
    std::cerr << "Error during initialize: " << e.what() << "\n";
    return 1;
  }

  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);

  return 0;
}
