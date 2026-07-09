# SPDX-License-Identifier: Apache-2.0
# YOLOv11 as a Factory-registered, config-driven CausalLM model class.
#
# This compiles the Yolov11Transformer model class (and the C2PSA custom layer)
# into causallm_src so the standard nntr_causallm / nntr_quantize binaries can
# build, run, and quantize YOLOv11 like any LLM — no YOLO_* env vars, driven by
# nntr_config.json. The model's custom layers and graph/post-proc headers live
# alongside the transformer in this directory, mirroring how gemma3 and gpt_oss
# colocate their custom layer sources (no jni/ subdir, no standalone binary).

causallm_src += [
    meson.current_source_dir() / 'yolov11_transformer.cpp',
    meson.current_source_dir() / 'c2psa_layer.cpp',
]

causallm_inc += include_directories('.')
