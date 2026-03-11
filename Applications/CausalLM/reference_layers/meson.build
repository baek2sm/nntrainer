custom_layers_inc_abs = [meson.current_source_dir()]

custom_layers_inc = [include_directories('.')]

custom_dot_wrapper_src_abs = [meson.current_source_dir() / 'custom_dot_wrapper.cpp']
custom_rms_norm_src_abs = [meson.current_source_dir() / 'custom_rms_norm.cpp']
custom_swiglu_src_abs = [meson.current_source_dir() / 'custom_swiglu.cpp']
# custom_mha_src_abs = [meson.current_source_dir() / 'custom_multi_head_attention_layer.cpp']
custom_fc_src_abs = [meson.current_source_dir() / 'custom_fc_layer.cpp']
custom_embedding_src_abs = [meson.current_source_dir() / 'custom_embedding_layer.cpp']
custom_lm_head_src_abs = [meson.current_source_dir() / 'custom_lm_head_layer.cpp']
custom_vocab_selection_src_abs = [meson.current_source_dir() / 'custom_vocab_selection.cpp']
custom_fc_lora_src_abs = [meson.current_source_dir() / 'custom_fc_lora_layer.cpp']
custom_qkv_src_abs = [meson.current_source_dir() / 'custom_qkv_layer.cpp']
custom_ffn_src_abs = [meson.current_source_dir() / 'custom_ffn_layer.cpp']
custom_mha_core_src_abs = [meson.current_source_dir() / 'custom_mha_core_layer.cpp']
custom_mha_core_v2_src_abs = [meson.current_source_dir() / 'custom_mha_core_v2_layer.cpp']
custom_tie_word_embedding_abs = [meson.current_source_dir() / 'custom_tie_word_embedding_layer.cpp']

openmp_dep = dependency('openmp')

if get_option('platform') != 'android'
  if get_option('default_library') == 'static'
    custom_dot_wrapper = static_library('custom_dot_wrapper',
      custom_dot_wrapper_src_abs,
      dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
      include_directories: custom_layers_inc,
      install: true,
      install_dir: application_install_dir,
    )
  else
    custom_dot_wrapper = shared_library('custom_dot_wrapper',
      custom_dot_wrapper_src_abs,
      dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
      include_directories: custom_layers_inc,
      install: true,
      install_dir: application_install_dir,
      cpp_args: '-DPLUGGABLE'
    )
  endif
  custom_dot_wrapper_dep = declare_dependency(
    link_with: custom_dot_wrapper,
      include_directories: custom_layers_inc
  )

  if get_option('default_library') == 'static'
    custom_rms_norm_layer = static_library('custom_rms_norm',
      custom_rms_norm_src_abs,
      dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
      include_directories: custom_layers_inc,
      install: true,
      install_dir: application_install_dir,
    )
  else
    custom_rms_norm_layer = shared_library('custom_rms_norm',
      custom_rms_norm_src_abs,
      dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
      include_directories: custom_layers_inc,
      install: true,
      install_dir: application_install_dir,
      cpp_args: '-DPLUGGABLE'
    )
  endif
  custom_rms_norm_dep = declare_dependency(
    link_with: custom_rms_norm_layer,
      include_directories: custom_layers_inc
  )

  if get_option('default_library') == 'static'
    custom_swiglu_layer = static_library('custom_swiglu',
      custom_swiglu_src_abs,
      dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
      include_directories: custom_layers_inc,
      install: true,
      install_dir: application_install_dir,
    )
  else
    custom_swiglu_layer = shared_library('custom_swiglu',
      custom_swiglu_src_abs,
      dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
      include_directories: custom_layers_inc,
      install: true,
      install_dir: application_install_dir,
      cpp_args: '-DPLUGGABLE'
    )
  endif
  custom_swiglu_dep = declare_dependency(
    link_with: custom_swiglu_layer,
      include_directories: custom_layers_inc
  )

  # custom_mha_layer = shared_library('custom_multi_head_attention',
  #   custom_mha_src_abs,
  #   dependencies: [nntrainer_dep, nntrainer_ccapi_dep, custom_dot_wrapper_dep, openmp_dep],
  #   include_directories: custom_layers_inc,
  #   install: true,
  #   install_dir: application_install_dir,
  #   cpp_args: '-DPLUGGABLE'
  # )
  # custom_mha_dep = declare_dependency(
  #   link_with: custom_mha_layer,
  #     include_directories: custom_layers_inc
  # )

  if get_option('default_library') == 'static'
    custom_fc_layer = static_library('custom_fc',
      custom_fc_src_abs,
      dependencies: [nntrainer_dep, nntrainer_ccapi_dep, custom_dot_wrapper_dep, openmp_dep],
      include_directories: custom_layers_inc,
      install: true,
      install_dir: application_install_dir,
    )
  else
    custom_fc_layer = shared_library('custom_fc',
      custom_fc_src_abs,
      dependencies: [nntrainer_dep, nntrainer_ccapi_dep, custom_dot_wrapper_dep, openmp_dep],
      include_directories: custom_layers_inc,
      install: true,
      install_dir: application_install_dir,
      cpp_args: '-DPLUGGABLE'
    )
  endif
  custom_fc_dep = declare_dependency(
    link_with: custom_fc_layer,
      include_directories: custom_layers_inc
  )

  if get_option('default_library') == 'static'
    custom_embedding_layer = static_library('custom_embedding',
      custom_embedding_src_abs,
      dependencies: [nntrainer_dep, nntrainer_ccapi_dep, openmp_dep],
      include_directories: custom_layers_inc,
      install: true,
      install_dir: application_install_dir,
    )
  else
    custom_embedding_layer = shared_library('custom_embedding',
      custom_embedding_src_abs,
      dependencies: [nntrainer_dep, nntrainer_ccapi_dep, openmp_dep],
      include_directories: custom_layers_inc,
      install: true,
      install_dir: application_install_dir,
      cpp_args: '-DPLUGGABLE'
    )
  endif
  custom_embedding_dep = declare_dependency(
    link_with: custom_embedding_layer,
      include_directories: custom_layers_inc
  )

  custom_vocab_selection_dep = []

  if get_option('enable-fp16')
    if get_option('default_library') == 'static'
      custom_vocab_selection_layer = static_library('custom_vocab_selection',
        custom_vocab_selection_src_abs,
        dependencies: [nntrainer_dep, nntrainer_ccapi_dep, openmp_dep],
        include_directories: custom_layers_inc,
        install: true,
        install_dir: application_install_dir,
      )
    else
      custom_vocab_selection_layer = shared_library('custom_vocab_selection',
        custom_vocab_selection_src_abs,
        dependencies: [nntrainer_dep, nntrainer_ccapi_dep, openmp_dep],
        include_directories: custom_layers_inc,
        install: true,
        install_dir: application_install_dir,
        cpp_args: '-DPLUGGABLE'
      )
    endif
  custom_vocab_selection_dep = declare_dependency(
    link_with: custom_vocab_selection_layer,
      include_directories: custom_layers_inc
  )
  endif

  if get_option('default_library') == 'static'
    custom_lm_head_layer = static_library('custom_lm_head',
      custom_lm_head_src_abs,
      dependencies: [nntrainer_dep, nntrainer_ccapi_dep, custom_dot_wrapper_dep, openmp_dep, custom_vocab_selection_dep],
      include_directories: custom_layers_inc,
      install: true,
      install_dir: application_install_dir,
    )
  else
    custom_lm_head_layer = shared_library('custom_lm_head',
      custom_lm_head_src_abs,
      dependencies: [nntrainer_dep, nntrainer_ccapi_dep, custom_dot_wrapper_dep, openmp_dep, custom_vocab_selection_dep],
      include_directories: custom_layers_inc,
      install: true,
      install_dir: application_install_dir,
      cpp_args: '-DPLUGGABLE'
    )
  endif
  custom_lm_head_dep = declare_dependency(
    link_with: custom_lm_head_layer,
      include_directories: custom_layers_inc
  )

  if get_option('default_library') == 'static'
    custom_fc_lora_layer = static_library('custom_fc_lora',
      custom_fc_lora_src_abs,
      dependencies: [nntrainer_dep, nntrainer_ccapi_dep, custom_dot_wrapper_dep, openmp_dep],
      include_directories: custom_layers_inc,
      install: true,
      install_dir: application_install_dir,
    )
  else
    custom_fc_lora_layer = shared_library('custom_fc_lora',
      custom_fc_lora_src_abs,
      dependencies: [nntrainer_dep, nntrainer_ccapi_dep, custom_dot_wrapper_dep, openmp_dep],
      include_directories: custom_layers_inc,
      install: true,
      install_dir: application_install_dir,
      cpp_args: '-DPLUGGABLE'
    )
  endif
  custom_fc_lora_dep = declare_dependency(
    link_with: custom_fc_lora_layer,
      include_directories: custom_layers_inc
  )

  if get_option('default_library') == 'static'
    custom_qkv_layer = static_library('custom_qkv',
      custom_qkv_src_abs,
      dependencies: [nntrainer_dep, nntrainer_ccapi_dep, custom_dot_wrapper_dep, openmp_dep],
      include_directories: custom_layers_inc,
      install: true,
      install_dir: application_install_dir,
    )
  else
    custom_qkv_layer = shared_library('custom_qkv',
      custom_qkv_src_abs,
      dependencies: [nntrainer_dep, nntrainer_ccapi_dep, custom_dot_wrapper_dep, openmp_dep],
      include_directories: custom_layers_inc,
      install: true,
      install_dir: application_install_dir,
      cpp_args: '-DPLUGGABLE'
    )
  endif
  custom_qkv_dep = declare_dependency(
    link_with: custom_qkv_layer,
      include_directories: custom_layers_inc
  )

  if get_option('default_library') == 'static'
    custom_ffn_layer = static_library('custom_ffn',
      custom_ffn_src_abs,
      dependencies: [nntrainer_dep, nntrainer_ccapi_dep, custom_dot_wrapper_dep, openmp_dep],
      include_directories: custom_layers_inc,
      install: true,
      install_dir: application_install_dir,
    )
  else
    custom_ffn_layer = shared_library('custom_ffn',
      custom_ffn_src_abs,
      dependencies: [nntrainer_dep, nntrainer_ccapi_dep, custom_dot_wrapper_dep, openmp_dep],
      include_directories: custom_layers_inc,
      install: true,
      install_dir: application_install_dir,
      cpp_args: '-DPLUGGABLE'
    )
  endif
  custom_ffn_dep = declare_dependency(
    link_with: custom_ffn_layer,
      include_directories: custom_layers_inc
  )

  if get_option('default_library') == 'static'
    custom_mha_core_layer = static_library('custom_mha_core',
      custom_mha_core_src_abs,
      dependencies: [nntrainer_dep, nntrainer_ccapi_dep, openmp_dep],
      include_directories: custom_layers_inc,
      install: true,
      install_dir: application_install_dir,
    )
  else
    custom_mha_core_layer = shared_library('custom_mha_core',
      custom_mha_core_src_abs,
      dependencies: [nntrainer_dep, nntrainer_ccapi_dep, openmp_dep],
      include_directories: custom_layers_inc,
      install: true,
      install_dir: application_install_dir,
      cpp_args: '-DPLUGGABLE'
    )
  endif
  custom_mha_core_dep = declare_dependency(
    link_with: custom_mha_core_layer,
      include_directories: custom_layers_inc
  )

  if get_option('default_library') == 'static'
    custom_mha_core_v2_layer = static_library('custom_mha_core_v2',
      custom_mha_core_v2_src_abs,
      dependencies: [nntrainer_dep, nntrainer_ccapi_dep, openmp_dep],
      include_directories: custom_layers_inc,
      install: true,
      install_dir: application_install_dir,
    )
  else
    custom_mha_core_v2_layer = shared_library('custom_mha_core_v2',
      custom_mha_core_v2_src_abs,
      dependencies: [nntrainer_dep, nntrainer_ccapi_dep, openmp_dep],
      include_directories: custom_layers_inc,
      install: true,
      install_dir: application_install_dir,
      cpp_args: '-DPLUGGABLE'
    )
  endif
  custom_mha_core_v2_dep = declare_dependency(
    link_with: custom_mha_core_v2_layer,
      include_directories: custom_layers_inc
  )
  
  if get_option('default_library') == 'static'
    custom_tie_word_embedding_layer = static_library('custom_tie_word_embedding',
      custom_tie_word_embedding_abs,
      dependencies: [nntrainer_dep, nntrainer_ccapi_dep, custom_dot_wrapper_dep, openmp_dep],
      include_directories: custom_layers_inc,
      install: true,
      install_dir: application_install_dir,
    )
  else
    custom_tie_word_embedding_layer = shared_library('custom_tie_word_embedding',
      custom_tie_word_embedding_abs,
      dependencies: [nntrainer_dep, nntrainer_ccapi_dep, custom_dot_wrapper_dep, openmp_dep],
      include_directories: custom_layers_inc,
      install: true,
      install_dir: application_install_dir,
      cpp_args: '-DPLUGGABLE'
    )
  endif
  custom_tie_word_embedding_dep = declare_dependency(
    link_with: custom_tie_word_embedding_layer,
      include_directories: custom_layers_inc
  )

else
  custom_layers_dir = meson.current_build_dir() / 'libs' / 'arm64-v8a'
  run_command('mkdir', '-p', custom_layers_dir, check: true)
  subdir('jni')
endif
