// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   multi_data.h
 * @date   15 Jun 2023
 * @brief  multi data loader
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "multi_data.h"

#include <cstring>
#include <iostream>
#include <nntrainer_error.h>
#include <random>

namespace nntrainer::util {

namespace {

/**
 * @brief fill label to the given memory
 *
 * @param data data to fill
 * @param length size of the data
 * @param label label
 */
void fillLabel(float *data, unsigned int length, unsigned int label) {
  if (length == 1) {
    *data = label;
    return;
  }

  memset(data, 0, length * sizeof(float));
  *(data + label) = 1;
}

/**
 * @brief fill last to the given memory
 * @note this function increases iteration value, if last is set to true,
 * iteration resets to 0
 *
 * @param[in/out] iteration current iteration
 * @param data_size Data size
 * @return bool true if iteration has finished
 */
bool updateIteration(unsigned int &iteration, unsigned int data_size) {
  if (iteration++ == data_size) {
    iteration = 0;
    return true;
  }
  return false;
};

} // namespace

MultiDataLoader::MultiDataLoader(const std::vector<TensorDim> &input_shapes,
                                   const std::vector<TensorDim> &output_shapes,
                                   int data_size_) :
  iteration(0),
  data_size(data_size_),
  input_shapes(input_shapes),
  output_shapes(output_shapes),
  input_dist(0, 255),
  label_dist(0, output_shapes.front().width() - 1) {
  NNTR_THROW_IF(output_shapes.empty(), std::invalid_argument)
    << "output_shape size empty not supported";
  NNTR_THROW_IF(output_shapes.size() > 1, std::invalid_argument)
    << "output_shape size > 1 is not supported";
}

void MultiDataLoader::next(float **input, float **label, bool *last) {

  auto fill_input = [this](float *input, unsigned int length, unsigned int value) {
    for (unsigned int i = 0; i < length; ++i) {
      *input = value;
      input++;
    }
  };

  auto fill_label = [this](float *label, unsigned int batch,
                           unsigned int length) {
    unsigned int generated_label = label_dist(rng);
    fillLabel(label, length, generated_label);
    label += length;
  };

  if (updateIteration(iteration, data_size)) {
    *last = true;
    return;
  }

  float **cur_input_tensor = input;
  for (unsigned int i = 0; i < input_shapes.size(); ++i) {
    fill_input(*cur_input_tensor, input_shapes.at(i).getFeatureLen(), i);
    cur_input_tensor++;
  }

  float **cur_label_tensor = label;
  for (unsigned int i = 0; i < output_shapes.size(); ++i) {
    fill_label(*label, output_shapes.at(i).batch(),
               output_shapes.at(i).getFeatureLen());
    cur_label_tensor++;
  }
}


} // namespace nntrainer::util
