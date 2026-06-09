# L7 CausalLM Model Index

> **Layer 7.** Folder-by-folder index for `Applications/CausalLM/models/`.
> This page exists so a contributor can find the right implementation family
> before reading code.

---

## 1. Responsibility

Map each model family folder to the concrete classes, the shared base classes,
and the platform constraints that matter.

---

## 2. Shared model files

These are the files every model family builds on.

| File | Role |
|---|---|
| `transformer.{h,cpp}` | Shared runtime foundation: config parsing, tokenizer loading, graph construction, initialize/load/run flow, save/load helpers. |
| `causal_lm.{h,cpp}` | Decoder-only specialization: LM head, KV cache, generation loop, output decoding. |
| `sentence_transformer.{h,cpp}` | Embedding / sentence-style runtime path. |
| `performance_metrics.h` | Runtime metrics structure returned by the model paths. |
| `meson.build` | Root model build graph and platform-gated subdirectories. |

---

## 3. Family folders

### 3.1 `qwen2/`

- `qwen2_causallm.{h,cpp}`
- `qwen2_embedding.{h,cpp}`

Role:
- Qwen2 causal and embedding paths.

### 3.2 `qwen3/`

- `qwen3_causallm.{h,cpp}`
- `qwen3_embedding.{h,cpp}`

Role:
- Main Qwen3 family for causal and embedding models.

### 3.3 `qwen3_moe/`

- `qwen3_moe_causallm.{h,cpp}`
- `qwen_moe_layer.{h,cpp}`

Role:
- Qwen3 MoE path and its expert-routing layer implementation.

### 3.4 `qwen3_slim_moe/`

- `qwen3_slim_moe_causallm.{h,cpp}`
- `qwen_moe_layer_fsu.{h,cpp}`

Role:
- Slim MoE path with FSU-style on-the-fly expert loading.

### 3.5 `qwen3_cached_slim_moe/`

- `qwen3_cached_slim_moe_causallm.{h,cpp}`
- `qwen_moe_layer_cached.{h,cpp}`

Role:
- Cached slim MoE path, used on non-Windows builds.

### 3.6 `gpt_oss/`

- `gptoss_causallm.{h,cpp}`
- `gpt_oss_moe_layer.{h,cpp}`

Role:
- GPT-OSS causal and MoE support.

### 3.7 `gpt_oss_cached_slim/`

- `gptoss_cached_slim_causallm.{h,cpp}`
- `gpt_oss_moe_layer_cached.{h,cpp}`

Role:
- Cached slim GPT-OSS path, non-Windows only.

### 3.8 `gemma3/`

- `gemma3_causallm.{h,cpp}`
- `embedding_gemma.{h,cpp}`
- `function.{h,cpp}`

Role:
- Gemma3 causal, embedding, and function/chat-template helpers.

### 3.9 `deberta_v2/`

- `deberta_v2.{h,cpp}`

Role:
- DeBERTa-v2 masked-LM / embedding-style path.

### 3.10 `bert/`

- `bert_transformer.{h,cpp}`
- `multilingual_tinybert_16mb.{h,cpp}`

Role:
- TinyBERT / BERT-style embedding path.

This folder is not built on Windows and is also excluded from Android.

### 3.11 `timm_vit/`

- `timm_vit_transformer.{h,cpp}`

Role:
- Vision transformer path used by the multimodal stacks.

---

## 4. Platform split

### Windows

Built:
- `qwen2/`
- `qwen3/`
- `qwen3_moe/`
- `qwen3_slim_moe/`
- `gpt_oss/`
- `gemma3/`
- `deberta_v2/`
- `timm_vit/`

Not built:
- `qwen3_cached_slim_moe/`
- `gpt_oss_cached_slim/`
- `bert/`

### Non-Windows

Full model matrix is available, including the cached-slim and BERT paths.

### Android

Same high-level model groups as non-Windows, but some registrations are
guarded in `main.cpp` and `causal_lm_api.cpp` depending on the platform.

---

## 5. What to read first

1. `models/meson.build`
2. `main.cpp` or `api/causal_lm_api.cpp`
3. The family folder you care about
4. The shared base files if the change crosses families

