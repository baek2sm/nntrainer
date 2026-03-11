causallm_layer_inc_abs = [meson.current_source_dir()]
causallm_layer_inc = [include_directories('.')]

causallm_rms_norm_src_abs = [meson.current_source_dir() / 'rms_norm.cpp']
causallm_swiglu_src_abs = [meson.current_source_dir() / 'swiglu.cpp']
causallm_tie_word_embedding_src_abs = [meson.current_source_dir() / 'tie_word_embedding.cpp']
causallm_lmhead_src_abs = [meson.current_source_dir() / 'lm_head.cpp']
causallm_mha_core_abs = [meson.current_source_dir()/ 'mha_core.cpp']
causallm_embedding_src_abs = [meson.current_source_dir() / 'embedding_layer.cpp']
causallm_reshaped_rms_norm_src_abs = [meson.current_source_dir() / 'reshaped_rms_norm.cpp']
causallm_qkv_layer_src_abs = [meson.current_source_dir() / 'qkv_layer.cpp']
causallm_embedding_pooling_src_abs = [meson.current_source_dir() / 'embedding_pooling_layer.cpp']
causallm_embedding_normalize_src_abs = [meson.current_source_dir() / 'embedding_normalize_layer.cpp']
causallm_dot_wrapper_src_abs = [meson.current_source_dir() / 'dot_wrapper.cpp']
causallm_vocab_selection_src_abs = [meson.current_source_dir() / 'vocab_selection.cpp']
causallm_fc_src_abs = [meson.current_source_dir() / 'fc_layer.cpp']
causallm_fc_lora_src_abs = [meson.current_source_dir() / 'fc_lora_layer.cpp']
causallm_ffn_src_abs = [meson.current_source_dir() / 'ffn_layer.cpp']


openmp_dep = dependency('openmp')
causallm_rms_norm = shared_library(
    'rms_norm_layer', 
    causallm_rms_norm_src_abs, 
    include_directories: causallm_layer_inc, 
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_rms_norm_dep = declare_dependency(
    link_with: causallm_rms_norm,
    include_directories: causallm_layer_inc
)

causallm_swiglu = shared_library(
    'swiglu_layer',
    causallm_swiglu_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_swiglu_dep = declare_dependency(
    link_with: causallm_swiglu,
    include_directories: causallm_layer_inc
)

causallm_tie_word_embedding = shared_library(
    'tie_word_embedding_layer',
    causallm_tie_word_embedding_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, openmp_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_tie_word_embedding_dep = declare_dependency(
    link_with: causallm_tie_word_embedding,
    include_directories: causallm_layer_inc
)

causallm_lm_head = shared_library(
    'lm_head',
    causallm_lmhead_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_lm_head_dep = declare_dependency(
    link_with: causallm_lm_head,
    include_directories: causallm_layer_inc
)

causallm_mha_core = shared_library(
    'mha_core_layer',
    causallm_mha_core_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, openmp_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_mha_core_dep = declare_dependency(
    link_with: causallm_mha_core,
    include_directories: causallm_layer_inc
)

causallm_embedding_layer = shared_library(
    'embedding_layer',
    causallm_embedding_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, openmp_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_embedding_layer_dep = declare_dependency(
    link_with: causallm_embedding_layer,
    include_directories: causallm_layer_inc
)

causallm_embedding_pooling_layer = shared_library(
    'embedding_pooling_layer',
    causallm_embedding_pooling_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, openmp_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_embedding_pooling_layer_dep = declare_dependency(
    link_with: causallm_embedding_pooling_layer,
    include_directories: causallm_layer_inc
)

causallm_embedding_normalize_layer = shared_library(
    'embedding_normalize_layer',
    causallm_embedding_normalize_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

causallm_embedding_normalize_layer_dep = declare_dependency(
    link_with: causallm_embedding_normalize_layer,
    include_directories: causallm_layer_inc
)

causallm_reshaped_rms_norm = shared_library(
    'reshaped_rms_norm_layer', 
    causallm_reshaped_rms_norm_src_abs, 
    include_directories: causallm_layer_inc, 
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_reshaped_rms_norm_dep = declare_dependency(
    link_with: causallm_reshaped_rms_norm,
    include_directories: causallm_layer_inc
)

causallm_qkv_layer = shared_library(
    'qkv_layer',
    causallm_qkv_layer_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

causallm_qkv_layer_dep = declare_dependency(
    link_with: causallm_qkv_layer,
    include_directories: causallm_layer_inc
)

causallm_dot_wrapper = shared_library(
    'dot_wrapper',
    causallm_dot_wrapper_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_dot_wrapper_dep = declare_dependency(
    link_with: causallm_dot_wrapper,
    include_directories: causallm_layer_inc
)

causallm_vocab_selection = shared_library(
    'vocab_selection',
    causallm_vocab_selection_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, openmp_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_vocab_selection_dep = declare_dependency(
    link_with: causallm_vocab_selection,
    include_directories: causallm_layer_inc
)

causallm_fc_layer = shared_library(
    'fc_layer',
    causallm_fc_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, causallm_dot_wrapper_dep, openmp_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_fc_layer_dep = declare_dependency(
    link_with: causallm_fc_layer,
    include_directories: causallm_layer_inc
)

causallm_fc_lora_layer = shared_library(
    'fc_lora_layer',
    causallm_fc_lora_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, causallm_dot_wrapper_dep, openmp_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_fc_lora_layer_dep = declare_dependency(
    link_with: causallm_fc_lora_layer,
    include_directories: causallm_layer_inc
)

causallm_ffn_layer = shared_library(
    'ffn_layer',
    causallm_ffn_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, causallm_dot_wrapper_dep, openmp_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_ffn_layer_dep = declare_dependency(
    link_with: causallm_ffn_layer,
    include_directories: causallm_layer_inc
)
