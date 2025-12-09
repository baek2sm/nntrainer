subdir('LogisticRegression/jni')
if enable_ccapi
  subdir('MNIST/jni')
  if get_option('enable-fp16')
    subdir('MixedPrecision/jni')
  endif
endif
subdir('Multi_input/jni')

subdir('ProductRatings/jni')
subdir('Layers/jni')
if get_option('enable-tflite-backbone')
  subdir('SimpleShot')
endif
subdir('ONNX/jni')
subdir('SimpleFC/jni')

if host_machine.system() != 'windows'
  if get_option('enable-tflite-backbone')
    subdir('KNN/jni')
    subdir('TransferLearning/CIFAR_Classification/jni')
  endif
  subdir('ReinforcementLearning/DeepQ/jni')
  subdir('Custom')
  if get_option('enable-tflite-interpreter')
  subdir('TFlite_export')
  endif
endif
