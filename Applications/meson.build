fs = import('fs')

nntr_app_resdir = nntrainer_resdir / 'app'

if build_machine.system() == 'windows'
  nntr_app_resdir_win = nntr_app_resdir.replace('/', '\\')
  if not fs.exists (nntr_app_resdir_win)
    run_command([prog_win_cmd, '/C', 'mkdir', nntr_app_resdir_win], check: true)
  endif
else
  run_command(['mkdir', '-p', nntr_app_resdir], check: true)
endif

subdir('utils')
subdir('Examples')

subdir('AlexNet/jni')
subdir('VGG/jni')
subdir('Resnet/jni')

if host_machine.system() != 'windows'
  subdir('YOLOv2/jni')
  subdir('YOLOv3/jni')
endif

if get_option('enable-transformer')
  subdir('LLaMA/jni')
  if host_machine.system() != 'windows'
  subdir('PicoGPT/jni')
  endif
  if (get_option('platform') != 'tizen') and (get_option('platform') != 'windows')
    subdir('CausalLM')
  endif
endif
