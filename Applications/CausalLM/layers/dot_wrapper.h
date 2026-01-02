// SPDX-License-Identifier: Apache-2.0
/**
 * @file	dot_wrapper.h
 * @date	02 October 2025
 * @brief	Wrapper for dot operation
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __CUSTOM_DOT_WRAPPER_H__
#define __CUSTOM_DOT_WRAPPER_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <tensor.h>

namespace custom {

WIN_EXPORT void custom_dot(nntrainer::Tensor &output, nntrainer::Tensor weight,
                           nntrainer::Tensor input);

WIN_EXPORT void custom_dot(nntrainer::Tensor &output, nntrainer::Tensor weight,
                           nntrainer::Tensor input, unsigned int from,
                           unsigned int to);

WIN_EXPORT void custom_dot(std::vector<nntrainer::Tensor *> outputs,
                           std::vector<nntrainer::Tensor *> weight,
                           nntrainer::Tensor input, unsigned int from,
                           unsigned int to);

} // namespace custom

#endif // __CUSTOM_DOT_WRAPPER_H__
