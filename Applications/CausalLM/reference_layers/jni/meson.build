ndk_build_command = [ndk_build]
foreach key, val : ndk_base_args
  ndk_build_command += '@0@=@1@'.format(key, val)
endforeach

ndk_args = {}
foreach key, val : ndk_args
  ndk_build_command += '@0@=@1@'.format(key, val)
endforeach

ndk_build_command += ['-C', meson.current_build_dir()]

message('@0@'.format(ndk_build_command))

and_conf = configuration_data()

and_conf.set('MESON_CUSTOM_DOT_WRAPPER_SRCS', ' '.join(custom_dot_wrapper_src_abs))
and_conf.set('MESON_CUSTOM_RMS_SRCS', ' '.join(custom_rms_norm_src_abs))
and_conf.set('MESON_CUSTOM_SWIGLU_SRCS', ' '.join(custom_swiglu_src_abs))
and_conf.set('MESON_CUSTOM_MHA_SRCS', ' '.join(custom_mha_src_abs))
and_conf.set('MESON_CUSTOM_FC_SRCS', ' '.join(custom_fc_src_abs))
and_conf.set('MESON_CUSTOM_EMBEDDING_SRCS', ' '.join(custom_embedding_src_abs))
and_conf.set('MESON_CUSTOM_LM_HEAD_SRCS', ' '.join(custom_lm_head_src_abs))
and_conf.set('MESON_CUSTOM_VOCAB_SELECTION_SRCS', ' '.join(custom_vocab_selection_src_abs))
and_conf.set('MESON_BIQGEMM_INCS', ' '.join(biqgemm_inc_abs))
and_conf.set('MESON_CUSTOM_FC_LORA_SRCS', ' '.join(custom_fc_lora_src_abs))
and_conf.set('MESON_CUSTOM_MHA_CORE_SRCS', ' '.join(custom_mha_core_src_abs))

and_conf.set('MESON_CUSTOM_LAYERS_INCS', ' '.join(custom_layers_inc_abs))

configure_file(input: 'Android.mk.in', output: 'Android.mk',
  configuration: and_conf
)
configure_file(input: 'Application.mk', output: 'Application.mk',
  configuration: and_conf
)

custom_layers_custom_tgt = custom_target('ndk-build',
                          output: 'jni',
                          build_by_default: true,
                        #  install: true,
                        #  install_dir: application_install_dir,
                          command: [ndk_build_command])
