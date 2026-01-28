fs = import('fs')
pytorch_dir = fs.parent(meson.current_source_dir()) / 'PyTorch'
build_pytorch_dir = meson.current_build_dir() / 'PyTorch'

if build_machine.system() == 'windows'
  pytorch_dir_win = pytorch_dir.replace('/', '\\')
  build_pytorch_dir_win = build_pytorch_dir.replace('/', '\\')
  if not fs.exists(build_pytorch_dir_win)
    run_command([prog_win_cmd, '/C', 'mkdir', build_pytorch_dir_win], check: true)
  endif
  run_command(['xcopy', '/C', '/Y', pytorch_dir_win + '\\*', build_pytorch_dir_win], check: true)
else
  if not fs.exists(build_pytorch_dir)
    run_command(['cp', '-lr', pytorch_dir, build_pytorch_dir], check: true)
  endif
endif

e = executable('nntrainer_simple_conv2d',
  'main.cpp',
  dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
  install: get_option('install-app'),
  install_dir: application_install_dir
)
