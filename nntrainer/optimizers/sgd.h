// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   sgd.h
 * @date   6 October 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the SGD optimizer.
 */
#ifndef __SGD_H__
#define __SGD_H__
#ifdef __cplusplus

#include <optimizer_devel.h>

namespace nntrainer {

/**
 * @class   SGD optimizer class
 * @brief   Stochastic Gradient Descent optimizer class
 */
class SGD : public Optimizer {
public:
  /**
   * @brief Construct a new SGD object
   *
   */
  SGD() {}

  /**
   * @copydoc Optimizer::getDefaultLearningRate()
   *
   */
  double getDefaultLearningRate() const override { return 0.0001; }

  /**
   * @copydoc applyGradient(RunOptimizerContext &context)
   */
  void applyGradient(RunOptimizerContext &context) override;

  /**
   * @copydoc Optimizer::getType()
   */
  const std::string getType() const override { return SGD::type; }

  /**
   * @copydoc Optimizer::getOptimizerVariableDim(const TensorDim &dim)
   */
  std::vector<TensorDim>
  getOptimizerVariableDim(const TensorDim &dim) override {
    return {};
  }

  static constexpr const char *type = "sgd";
};
} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __SGD_H__ */
