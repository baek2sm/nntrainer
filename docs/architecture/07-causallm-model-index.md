# L7 CausalLM Supported Models

> **Layer 7.** This page is the support matrix for `Applications/CausalLM/`.
> It focuses on the models you can actually load, the class families behind
> them, and the platform split. It intentionally does not split the tree by
> implementation folder.

---

## 1. Responsibility

Document the supported model list, the class family behind each model, and the
platform matrix that determines whether the model is available on Windows,
Android, or non-Windows builds.

---

## 2. Supported model list

### 2.1 Decoder-only causal models

| Model key | Runtime class | Notes |
|---|---|---|
| `LlamaForCausalLM` | `causallm::CausalLM` | Base decoder-only path. |
| `Qwen2ForCausalLM` | `causallm::Qwen2CausalLM` | Qwen2 causal path. |
| `Qwen3ForCausalLM` | `causallm::Qwen3CausalLM` | Main Qwen3 causal path. |
| `Qwen3MoeForCausalLM` | `causallm::Qwen3MoECausalLM` | Qwen3 MoE causal path. |
| `Qwen3SlimMoeForCausalLM` | `causallm::Qwen3SlimMoECausalLM` | Slim MoE path. |
| `Qwen3CachedSlimMoeForCausalLM` | `causallm::Qwen3CachedSlimMoECausalLM` | Cached slim MoE path. |
| `GptOssForCausalLM` | `causallm::GptOssForCausalLM` | GPT-OSS causal path. |
| `GptOssCachedSlimCausalLM` | `causallm::GptOssCachedSlimCausalLM` | Cached slim GPT-OSS path. |
| `Gemma3ForCausalLM` | `causallm::Gemma3CausalLM` | Gemma3 causal path. |

### 2.2 Embedding / encoder-style models

| Model key | Runtime class | Notes |
|---|---|---|
| `Qwen2Embedding` | `causallm::Qwen2Embedding` | Qwen2 embedding path. |
| `Qwen3Embedding` | `causallm::Qwen3Embedding` | Qwen3 embedding path. |
| `EmbeddingGemma` | `causallm::EmbeddingGemma` | Gemma3 embedding path. |
| `DebertaV2` | `causallm::DebertaV2` | DeBERTa-v2 masked-LM / embedding path. |
| `MultilingualTinyBert` | `causallm::MultilingualTinyBert` | TinyBERT / BERT embedding path. |
| `TimmViT` | `causallm::TimmViTTransformer` | Vision transformer path. |

---

## 3. Support matrix

### Windows

Available:
- `LlamaForCausalLM`
- `Qwen2ForCausalLM`
- `Qwen2Embedding`
- `Qwen3ForCausalLM`
- `Qwen3MoeForCausalLM`
- `Qwen3SlimMoeForCausalLM`
- `Qwen3Embedding`
- `GptOssForCausalLM`
- `Gemma3ForCausalLM`
- `EmbeddingGemma`
- `DebertaV2`
- `TimmViT`

Not available:
- `Qwen3CachedSlimMoeForCausalLM`
- `GptOssCachedSlimCausalLM`
- `MultilingualTinyBert`

### Non-Windows

Available:
- full list above, including the cached-slim and BERT family paths

### Android

Available:
- same broad model families as non-Windows, but platform guards in
  `main.cpp`, `causal_lm_api.cpp`, and `models/meson.build` may exclude some
  variants

---

## 4. Registration sources

The support list comes from three places:

1. `Applications/CausalLM/main.cpp` for the CLI binary
2. `Applications/CausalLM/api/causal_lm_api.cpp` for the C API path
3. `Applications/CausalLM/models/meson.build` for what actually gets built

When those three disagree, the platform build graph wins.

---

## 5. What a reader should learn here

- Which model names are valid entry points.
- Which C++ class implements each model.
- Which models are intentionally missing on Windows.
- Which model families are embedding vs causal.
- Which files to inspect first when a model fails to load.
- Which families are built from the shared `createLayer()`-based transformer
  skeleton and which ones override attention or FFN assembly.
