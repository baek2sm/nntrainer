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

  // set label data
  std::string label_file = "./input_data.txt";
  float cur_loaded[55];
  std::memset(cur_loaded, 0.0, 55 * sizeof(float));

  std::ifstream file(label_file);
  std::string cur_line;

  int line_idx = 0;
  while (getline(file, cur_line)) {
    std::stringstream ss(cur_line);
    std::string cur_value;
    int row_idx = 0;
    while (getline(ss, cur_value, ' ')) {
      cur_loaded[row_idx] = std::stof(cur_value);
      row_idx++;
    }
    line_idx++;

    float* loaded = new float[55];
    std::copy(cur_loaded, cur_loaded+55, loaded);
    data_list.push_back(loaded);

    std::cout << "loaded data:" << std::endl;
    for (int i=0; i<55; i++) {
        std::cout << loaded[i] << " ";
    } std::cout << std::endl;
  }

  file.close();
 
  // set index and shuffle data
  idxes = std::vector<unsigned int>(data_list.size());
  std::cout<< data_list.size() <<std::endl;
  std::iota(idxes.begin(), idxes.end(), 0);
  std::shuffle(idxes.begin(), idxes.end(), rng);

  count = 0;
}

void MultiDataLoader::next(float **input, float **label, bool *last) {  
  auto fill_input = [this](float *input, unsigned int length, int db_index, int input_index) {
    input_index = (input_index == 0) ? 0 : 49+input_index;
    std::copy(data_list[db_index]+input_index, data_list[db_index]+input_index+length, input);
    input += length;
  };

  auto fill_label = [this](float *label, int db_index,
                           unsigned int length) {
    *label = *(data_list[db_index]+54);
    label++;
  };

  if (updateIteration(iteration, data_size)) {
    *last = true;
    return;
  }
  
  float **cur_input_tensor = input;  
  for (unsigned int i = 0; i < input_shapes.size(); ++i) {    
    fill_input(*cur_input_tensor, input_shapes.at(i).getFeatureLen(), idxes[count], i);
    cur_input_tensor++;
  }

  float **cur_label_tensor = label;
  for (unsigned int i = 0; i < output_shapes.size(); ++i) {
    fill_label(*label, idxes[count],
               output_shapes.at(i).getFeatureLen());
    cur_label_tensor++;
  }

  count++;

  if (count < data_size) {
    *last = false;
  } else {
    *last = true;
    count = 0;
    std::shuffle(idxes.begin(), idxes.end(), rng);
  }

}
} // namespace nntrainer::util
