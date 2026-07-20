# SPDX-License-Identifier: Apache-2.0
# YOLOv7ReIDtiny inference example

yoloreid_infer_src = [
  meson.current_source_dir() / 'main.cpp',
]

yoloreid_inc = include_directories('.')

yoloreid_infer = executable('yoloreid_infer',
  yoloreid_infer_src,
  include_directories: [yoloreid_inc],
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
  build_by_default: get_option('enable-app'),
  install: false,
)
