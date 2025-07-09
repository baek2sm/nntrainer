
// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file unittest_layers_slice.cpp
 * @date 9 April 2025
 * @brief Slice Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <slice_layer.h>

auto semantic_slice = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::SliceLayer>, nntrainer::SliceLayer::type,
  {"start_index=1", "end_index=2", "axis=3"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

auto semantic_slice_multi = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::SliceLayer>, nntrainer::SliceLayer::type,
  {"start_index=2", "end_index=3", "axis=3"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 2);

GTEST_PARAMETER_TEST(slice, LayerSemantics,
                     ::testing::Values(semantic_slice, semantic_slice_multi));
