# L8 CausalLM Resources

> **Layer 8.** The `Applications/CausalLM/res/` tree is the on-disk model
> contract. This page documents the file layout, the conversion scripts, and
> the weight-file assumptions that the runtime makes.

---

## 1. Responsibility

Explain how the shipped resources are laid out, how model weights are
generated, and how the runtime expects to find them.

---

## 2. Resource tree

Top-level resource folders:

- `deberta_v2/`
- `gemma3/`
- `gpt-oss/`
- `kalm-embedding/`
- `qwen2/`
- `qwen3/`
- `tiny-bert/`
- `vit/`

Each family folder contains one or more model directories with `config.json`,
`generation_config.json`, `nntr_config.json`, tokenizer files, and the weight
bin referenced by `nntr_config.json["model_file_name"]`.

---

## 3. Directory patterns

### 3.1 Qwen3

Typical layout:

```text
qwen3/
  qwen3-0.6b/
  qwen3-4b/
  qwen3-30b-a3b/
  qwen3-30b-a3b-slim-cached/
```

Files seen in this tree:
- `nntr_config.json`
- `weight_converter.py`
- `gguf_to_nntrainer.py`
- `README.md`

### 3.2 GPT-OSS

```text
gpt-oss/
  gpt-oss-20b/
```

Files:
- `nntr_config.json`
- `weight_converter.py`

### 3.3 Qwen2

```text
qwen2/
  qwen2-0.5b/
```

Files:
- `nntr_config.json`
- `weight_converter.py`

### 3.4 Gemma3

```text
gemma3/
  270m/
  function/
```

Files:
- `nntr_config.json`
- `weight_converter.py`

### 3.5 Tiny BERT

```text
tiny-bert/
```

Files:
- `config.json`
- `generation_config.json`
- `nntr_config.json`

### 3.6 Vision

```text
vit/
  timm_vit_base_patch16_siglip_224/
```

Files:
- `nntr_config.json`
- `weight_converter.py`

### 3.7 DeBERTa

```text
deberta_v2/
```

Files:
- `weight_converter.py`

### 3.8 KALM embedding

```text
kalm-embedding/
```

Files:
- `nntr_config.json`
- `weight_converter.py`

---

## 4. Weight formats

The runtime accepts two main weight source formats:

- `.bin` for nntrainer binary weights
- `.safetensors` for direct safetensors loading when present

`Transformer::formatFromExtension()` decides between them by file extension.

The important on-disk rule is that the runtime weight file name is taken from
`nntr_config.json["model_file_name"]`. The directory layout must match that
name exactly.

---

## 5. Conversion scripts

Scripts in `res/` create the shipping weight files and the matching
`nntr_config.json`.

Common scripts:

- `weight_converter.py`
- `convert_lm.py`
- `convert_connector.py`
- `convert_embedding.py`
- `convert_vision_hf.py`
- `gguf_to_nntrainer.py`

The conversion scripts usually do three things:

1. read HuggingFace or GGUF source weights,
2. repack them into nntrainer binary layout,
3. emit the matching runtime config.

---

## 6. Tensor / dtype assumptions

The resource files encode a specific dtype contract:

- `model_tensor_type` describes the weight/activation combo.
- `fc_layer_dtype` controls transformer FC / projection layers.
- `embedding_dtype` controls embedding tables.
- `lmhead_dtype` controls output projection weights.

Typical values used by the shipped configs:

- `FP32-FP32`
- `Q4_0-FP32`
- `FP32`
- `Q4_0`
- `Q6_K`

If a converter changes a file layout, the matching `nntr_config.json` must be
updated in the same tree.

---

## 7. Runtime contract

The runtime expects:

1. `config.json` to describe the architecture.
2. `generation_config.json` to provide generation settings when needed.
3. `nntr_config.json` to point to the weight file and runtime dtype settings.
4. Tokenizer files to exist when the model path needs them.
5. The weight file to exist with the name in `model_file_name`.

Some model families have hardcoded internal configurations in the API path, but
the file-based contract above is the default and should be documented first.

---

## 8. What to update when changing resources

- Update the family folder README if a file name changes.
- Update `nntr_config.json` if the runtime weight file name changes.
- Update the converter script and the docs together if the layout changes.
- Update the model index if a family gets a new directory or platform split.

