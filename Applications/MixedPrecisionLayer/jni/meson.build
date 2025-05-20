mixed_sources = [
  'main.cpp'
]

executable('nntrainer_mixed_precision_layer',
  mixed_sources,
  dependencies: [iniparser_dep, nntrainer_dep, nntrainer_ccapi_dep, app_utils_dep],
  include_directories: [include_directories('.')],
  install: get_option('install-app'),
  install_dir: application_install_dir
)
