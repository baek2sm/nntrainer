# LFM2-VL-450M Weight Conversion

Convert LiquidAI/LFM2-VL-450M from HuggingFace to the nntrainer binary
format consumed by Lfm2CausalLM and ClipVitTransformer.

## Prerequisites
pip install torch safetensors numpy transformers pillow huggingface_hub

## 1. Download the HuggingFace model
```
huggingface-cli download LiquidAI/LFM2-VL-450M --local-dir /path/to/hf_model
```
Scripts only need `model.safetensors` from the download.

## 2. Create the output model directory
```
mkdir /path/to/model_dir
```

## 3. Convert each weight component

Run all four converters from the `res/lfm2-vl/` directory:

### Vision tower
```
python convert_vision_hf.py \
  --hf-model /path/to/hf_model/model.safetensors \
  --out-dir /path/to/model_dir \
  --out-name lfm2_vl_450m_vision.bin
```

### Language model
```
python convert_lm.py \
  --hf-model /path/to/hf_model/model.safetensors \
  --out-dir /path/to/model_dir
```
Output: `lfm2_vl_450m_lm.bin`

### Connector
```
python convert_connector.py \
  --hf-model /path/to/hf_model/model.safetensors \
  --out-dir /path/to/model_dir
```
Output: `lfm2_vl_450m_connector.bin` (includes LayerNorm 3072-dim + linear_1 + linear_2;
projector_hidden_size=2560; expected size: 41,981,952 bytes)

### Embedding table
```
python convert_embedding.py \
  --hf-model /path/to/hf_model/model.safetensors \
  --out-dir /path/to/model_dir
```
Output: `lfm2_vl_450m_embedding.bin` (shape [65536, 1024] FP32; expected size: 268,435,456 bytes)

## 4. Copy tokenizer and config files
```
cp /path/to/hf_model/tokenizer.json          /path/to/model_dir/
cp /path/to/hf_model/tokenizer_config.json   /path/to/model_dir/
cp /path/to/hf_model/special_tokens_map.json /path/to/model_dir/
```

## 5. Copy config files
`config.json` and `generation_config.json` ship with the HuggingFace download;
only `nntr_config.json` comes from this repo.
```
cp /path/to/hf_model/config.json            /path/to/model_dir/
cp /path/to/hf_model/generation_config.json /path/to/model_dir/
cp res/lfm2-vl/nntr_config.json             /path/to/model_dir/
```

## 6. Place an input image
Copy any image as the file named by `image_path` in `nntr_config.json` (default `sample.png`):
```
cp /path/to/your/photo.jpg /path/to/model_dir/sample.png
```
The binary decodes the file, resizes to 256x256, and normalizes internally
(SigLIP2: mean=std=0.5). Format is auto-detected from file content (stb_image):
JPEG, PNG, BMP, GIF, TGA, PSD, HDR, PIC, PNM — the file extension does not matter.

## 7. Expected model directory layout
After steps 3-6 the directory must contain exactly these files:
```
lfm2_vl_450m_lm.bin           LFM2 language model (16x hybrid conv/attn layers)
lfm2_vl_450m_vision.bin       SigLIP2 vision encoder (ClipVitTransformer)
lfm2_vl_450m_connector.bin    MultiModalProjector (LayerNorm + linear_1 + linear_2)
lfm2_vl_450m_embedding.bin    LM embedding table [65536 x 1024] FP32
tokenizer.json                HuggingFace tokenizer
tokenizer_config.json         HuggingFace tokenizer config
special_tokens_map.json       HuggingFace special tokens map
config.json                   Model architecture config (from hf_model)
nntr_config.json              NNTrainer runtime config (from res/lfm2-vl/)
generation_config.json        Generation parameters (from hf_model)
sample.png                    Input image (any format stb_image decodes)
```

## 8. Build nntr_causallm
See `Applications/CausalLM/README.md` for build instructions.

## 9. Run inference
```
nntr_causallm /path/to/model_dir
```
Or with an explicit prompt:
```
nntr_causallm /path/to/model_dir "What is in this image?"
```

The binary will:
1. Load `config.json` and `nntr_config.json` from the model dir
2. Resolve all filenames (tokenizer, weights, image) relative to the model dir
3. Run the vision encoder on `sample.png`
4. Merge image features with text embeddings
5. Generate up to `num_to_generate` tokens and print the caption

## 10. Key verified facts
- Image wrapper tokens: image_start=498, image=396, image_end=499
- Connector LayerNorm: 3072-dim, before linear projections
- projector_hidden_size: 2560
- Vocab size: 65536
- Vision: SigLIP2-NaFlex 86M, 12 layers, hidden 768, patch 16, image 256x256
- Downsample factor: 2 (pixel_unshuffle: 256 patches to 64 tokens, 768 to 3072 dim)
- LM: 16x hybrid (conv/full_attention), hidden 1024, intermediate 4458 (adjusted), GQA 16/8 heads
