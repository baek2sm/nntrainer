#include <dot_wrapper.h>

namespace causallm {

void custom_dot(nntrainer::Tensor &output, nntrainer::Tensor weight,
                nntrainer::Tensor input) {
  if (weight.getDataType() == nntrainer::TensorDim::DataType::BCQ) {
    nntrainer::Tensor input_T = input.transpose("0:2:1");

    ml::train::TensorDim output_T_dim = output.getDim();
    output_T_dim.height(output.width());
    output_T_dim.width(output.height());
    nntrainer::Tensor output_T(output_T_dim);
    weight.dot(input_T, output_T, false, true);
    output_T.transpose("0:2:1", output);
  } else {
    input.dot(weight, output);
  }
}

void custom_dot(nntrainer::Tensor &output, nntrainer::Tensor weight,
                nntrainer::Tensor input, unsigned int from, unsigned int to) {

  if (weight.getDataType() == nntrainer::TensorDim::DataType::BCQ) {
    if (to - from == 1) {
      weight.dot(input, output);
    } else {
      nntrainer::Tensor input_T = input.transpose("0:2:1");
      ml::train::TensorDim output_T_dim = output.getDim();
      output_T_dim.height(output.width());
      output_T_dim.width(output.height());

      nntrainer::Tensor output_T(output_T_dim);
      weight.dot(input_T, output_T, false, true);
      output_T.transpose("0:2:1", output);
    }
  } else {
    input.dot(weight, output);
  }
}

void custom_dot(std::vector<nntrainer::Tensor *> outputs,
                std::vector<nntrainer::Tensor *> weights,
                nntrainer::Tensor input, unsigned int from, unsigned int to) {
  // input.dot(weights, outputs);

  // input.dot(*(weights[0]), *(outputs[0]));
  // input.dot(*(weights[1]), *(outputs[1]));
  // input.dot(*(weights[2]), *(outputs[2]));

  input.dot(weights, outputs);
}

} // namespace causallm
