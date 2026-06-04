# LFM2 model family.
# lfm2_causallm.cpp ??LFM2 LM (decoder-only CausalLM, V-JEPA multimodal harness).
# lfm2-vl/          ??LFM2-VL multimodal (SigLIP2 vision + connector + LLM).

lfm2_src = [
    meson.current_source_dir() / 'lfm2_causallm.cpp',
]

lfm2_inc = include_directories('.')

causallm_src += lfm2_src
causallm_inc += lfm2_inc

subdir('lfm2-vl')
