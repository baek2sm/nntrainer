#include <onnx.h>

using ModelHandle = std::unique_ptr<ml::train::Model>;

int main() {
  ModelHandle model =
    ml::train::loadONNX("/home/seungbaek/projects/ONNX/test.onnx");

  model->setProperty({withKey("batch_size", 1), withKey("epochs", 1)});
  auto optimizer = ml::train::createOptimizer("sgd", {"learning_rate=0.001"});
  model->setOptimizer(std::move(optimizer));

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
