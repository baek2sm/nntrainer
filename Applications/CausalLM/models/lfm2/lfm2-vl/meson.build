lfm2_vl_src = [
    meson.current_source_dir() / 'lfm2_vl_projector.cpp',
]

lfm2_vl_inc = include_directories('.')

causallm_src += lfm2_vl_src
causallm_inc += lfm2_vl_inc

# LFM2.5-VL multimodal model (vision tower + LLM + multimodal projector).
# Each component is wired in below. Future additions (integrated top-level
# model) become sibling subdir entries.
subdir('vision')
