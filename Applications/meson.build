fs = import('fs')

nntr_app_resdir = nntrainer_resdir / 'app'

if host_machine.system() == 'windows'
  nntr_app_resdir_win = nntr_app_resdir.replace('/', '\\')
  if not fs.exists (nntr_app_resdir_win)
    run_command('cmd.exe', '/C', 'mkdir', nntr_app_resdir_win)
  endif
else
  run_command('mkdir', '-p', nntr_app_resdir)
endif

subdir('utils')
subdir('LogisticRegression/jni')
if enable_ccapi
  subdir('MNIST/jni')
  if get_option('enable-fp16')
    subdir('MixedPrecision/jni')
  endif
endif
subdir('VGG/jni')
subdir('Resnet/jni')
subdir('Multi_input/jni')

# if enable_capi
#   subdir('TransferLearning/Draw_Classification/jni')
# endif

subdir('ProductRatings/jni')
subdir('AlexNet/jni')
subdir('Layers/jni')
if get_option('enable-tflite-backbone')
  subdir('SimpleShot')
endif

subdir('SimpleFC/jni')

if host_machine.system() != 'windows'
  subdir('KNN/jni')
  subdir('YOLOv2/jni')
  subdir('YOLOv3/jni')
  subdir('LLaMA/jni')
  subdir('ReinforcementLearning/DeepQ/jni')
  subdir('TransferLearning/CIFAR_Classification/jni')
  subdir('Custom')
  subdir('PicoGPT/jni')
endif
