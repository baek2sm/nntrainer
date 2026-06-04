# LFM2-VL multimodal (vision tower + connector + LLM).
subdir('vision')

lfm2_vl_src = [
    meson.current_source_dir() / 'lfm2_vl_connector.cpp',
    meson.current_source_dir() / 'lfm2_vl_model.cpp',
]

lfm2_vl_inc = include_directories('.')

causallm_src += lfm2_vl_src
causallm_inc += lfm2_vl_inc
