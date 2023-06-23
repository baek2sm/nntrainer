vd_sources = [
  'main.cpp',
  'multi_data.cpp'
]

vd_dependencies = [app_utils_dep,
  iniparser_dep,
  nntrainer_dep,
  nntrainer_ccapi_dep
]

if get_option('enable-test')
  vd_dependencies += [gtest_dep]
endif

e = executable('nntrainer_vd',
  vd_sources,
  include_directories: [include_directories('.')],
  dependencies: vd_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)