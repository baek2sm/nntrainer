# LFM2-VL-450M Weight Conversion

Convert LiquidAI/LFM2-VL-450M from HuggingFace to the nntrainer binary
format consumed by Lfm2CausalLM and ClipVitTransformer.

## Prerequisites
pip install torch safetensors numpy transformers pillow huggingface_hub

## 1. Download the HuggingFace model
huggingface-cli download LiquidAI/LFM2-VL-450M --local-dir /path/to/hf_model
(scripts only need model.safetensors)

## 2. Convert each component

### Vision tower
python convert_vision_hf.py --hf-model /path/to/hf_model/model.safetensors --out-dir /path/to/out --out-name lfm2_vl_450m_vision.bin
Note: GGUF path (vision/gguf_to_nntrainer.py) additionally writes the learnable positional
embedding; the HF safetensors path does not.

### Language model
python convert_lm.py --hf-model /path/to/hf_model/model.safetensors --out-dir /path/to/out

### Connector
python convert_connector.py --hf-model /path/to/hf_model/model.safetensors --out-dir /path/to/out
Note: connector includes LayerNorm (3072-dim) before linear_1 and linear_2.
projector_hidden_size = 2560. Expected file size: 41,981,952 bytes.

### Embedding table
python convert_embedding.py --hf-model /path/to/hf_model/model.safetensors --out-dir /path/to/out
Shape [65536, 1024] FP32. Expected file size: 268,435,456 bytes.

## 3. Output directory layout
lfm2_vl_450m_vision.bin       SigLIP2 vision encoder (ClipVitTransformer)
lfm2_vl_450m_lm.bin           LFM2 language model (16x full_attention)
lfm2_vl_450m_connector.bin    MultiModalProjector (LayerNorm + linear_1 + linear_2)
lfm2_vl_450m_embedding.bin    LM embedding table [65536 x 1024] FP32
tokenizer.json                (copy from HF model dir)
tokenizer_config.json
special_tokens_map.json
nntr_config.json              (see Applications/CausalLM/res/lfm2-vl/nntr_config.json)

## 4. Key verified facts
- Image wrapper tokens: image_start=498, image=396, image_end=499
- Connector LayerNorm: 3072-dim, before linear projections
- projector_hidden_size: 2560
- Vocab size: 65536
- Vision: SigLIP2-NaFlex 86M, 12 layers, hidden 768, patch 16, image 256x256
- Downsample factor: 2 (pixel_unshuffle: 256 patches to 64 tokens, 768 to 3072 dim)
- LM: 16x full_attention, hidden 1024, intermediate 3584, GQA 16/8 heads

## 5. Running inference
Build nntr_causallm.exe (see Applications/CausalLM/README.md), then:
  nntr_causallm.exe /path/to/out "Describe the image."

To pass a real image file (jpg/png/bmp), set `image_path` in nntr_config.json:
  "image_path": "/path/to/photo.jpg"
The binary decodes the file, resizes to 256x256, and normalizes internally
(SigLIP2: mean=std=0.5). For a pre-made FP32 NCHW binary use `image_tensor_path`;
if both are set, `image_path` takes precedence.
