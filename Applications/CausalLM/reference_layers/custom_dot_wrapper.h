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
