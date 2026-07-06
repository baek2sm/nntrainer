fs = import('fs')

face_landmark_app_src = files('main.cpp')
face_landmark_prelu_src = files('prelu_layer.cpp')

face_landmark_inc = include_directories('.')

cpp_args = []
if get_option('default_library') == 'shared'
  cpp_args += [ '-DPLUGGABLE' ]
endif

prelu_layer = shared_library('prelu_layer',
  face_landmark_prelu_src,
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
  include_directories: face_landmark_inc,
  install: true,
  install_dir: application_install_dir,
  cpp_args: cpp_args
)

prelu_layer_dep = declare_dependency(
  link_with: prelu_layer,
  include_directories: face_landmark_inc
)

face_landmark_dependencies = [
  app_utils_dep,
  nntrainer_dep,
  nntrainer_ccapi_dep,
  prelu_layer_dep
]

e = executable('nntrainer_pfld',
  face_landmark_app_src,
  include_directories: [face_landmark_inc],
  dependencies: face_landmark_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)
