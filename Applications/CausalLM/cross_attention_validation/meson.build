cross_attention_validation_src = files('main.cpp')

cross_attention_validation_inc = [include_directories('.')]

executable('nntr_cross_attention_validation',
    cross_attention_validation_src,
    include_directories: causallm_inc + cross_attention_validation_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, causallm_mha_core_dep],
)
