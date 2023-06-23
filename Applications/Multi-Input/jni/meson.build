resnet_sources = [
  'main.cpp',
  'multi_data.cpp'
]

resnet_dependencies = [app_utils_dep,
  iniparser_dep,
  nntrainer_dep,
  nntrainer_ccapi_dep
]

e = executable('nntrainer_multiinput',
  resnet_sources,
  include_directories: [include_directories('.')],
  dependencies: resnet_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)
