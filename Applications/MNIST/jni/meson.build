fs = import('fs')
app_res_dir = fs.parent(meson.current_source_dir()) / 'res'
build_app_res_dir = nntr_app_resdir / 'MNIST'

if host_machine.system() == 'windows'
  app_res_dir_win = app_res_dir.replace('/', '\\')
  build_app_res_dir_win = build_app_res_dir.replace('/', '\\')
  if not fs.exists (build_app_res_dir_win)
    run_command('cmd.exe', '/C', 'mkdir', build_app_res_dir_win)
  endif
  run_command('cmd.exe', '/C', 'xcopy',app_res_dir_win, build_app_res_dir_win)
else
  run_command('cp', '-lr', app_res_dir, build_app_res_dir)
endif

mnist_sources = [
  'main.cpp'
]

if not gtest_dep.found()
  error('Gtest dependency not found for MNIST application')
endif

e = executable('nntrainer_mnist',
  mnist_sources,
  dependencies: [iniparser_dep, nntrainer_ccapi_dep, gtest_dep],
  include_directories: include_directories('.'),
  install: get_option('install-app'),
  install_dir: application_install_dir
)

test(
  'app_mnist',
  e,
  args: [
    build_app_res_dir / 'mnist.ini',
    build_app_res_dir / 'mnist_trainingSet.dat'
  ],
  timeout: 60
)

