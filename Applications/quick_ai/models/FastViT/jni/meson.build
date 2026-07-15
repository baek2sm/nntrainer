# SPDX-License-Identifier: Apache-2.0
# FastViT-S12 backbone feature extractor inference example

fastvit_backbone_infer_src = [
  meson.current_source_dir() / 'main.cpp',
  meson.current_source_dir() / 'fastvit_attention_layer.cpp',
]

fastvit_inc = include_directories('.')

fastvit_backbone_infer = executable('fastvit_backbone_infer',
  fastvit_backbone_infer_src,
  include_directories: [fastvit_inc],
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
  build_by_default: get_option('enable-app'),
  install: false,
)
