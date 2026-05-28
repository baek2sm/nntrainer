# quick.ai C ABI

This directory contains the in-process C ABI for packaging CausalLM as one
runtime DLL: `quick_ai.dll`.

## ABI

The public header is `quick_ai.h`. It exposes only C basic types, an
opaque `quick_ai_ctx *`, UTF-8 `const char *`, and callback function pointers.
All ABI functions are `noexcept` on C++ builds. Failures return a negative
`quick_ai_status`; call `quick_ai_last_error(ctx)` for the UTF-8 diagnostic.

`quick_ai_generate()` accepts one OpenAI chat-completions compatible JSON request:
`model`, `messages`, `temperature`, `top_p`, `top_k`, `max_tokens`, `stop`, and
`stream`. The ABI always streams through `quick_ai_token_cb`; `stream` is accepted
for compatibility. Unsupported OpenAI fields such as `frequency_penalty`,
`presence_penalty`, `logit_bias`, `seed`, and `logprobs` are ignored.

`quick_ai_embed()` accepts one OpenAI embeddings compatible JSON request:
`{"model":"...","input":"text"}`. `input` may also be an array containing
exactly one string; larger batches are rejected by this bounded ABI. On success,
`out_count` is `1`, and `out_embedding` points to `out_dim` contiguous `float`
values allocated inside the DLL. Release that buffer with `quick_ai_free()`.

`quick_ai_model_info()` returns JSON with the package id, path, architecture,
model type/kind when inferable, and model dimensions: `max_seq_len`,
`init_seq_len`, `num_to_generate`, `batch_size`, and `hidden_dim`.

`quick_ai_get_performance_metrics()` copies the current context metrics into
`quick_ai_performance_metrics`. These values describe the last generation run.
Measure embedding latency around `quick_ai_embed()` on the host side.

## Threading

Set inference threads with `quick_ai_init_options.num_threads`. Do not use
`NNTR_NUM_THREADS` or other environment variables for this ABI. A single
`quick_ai_ctx` serializes `quick_ai_generate()` and `quick_ai_embed()` calls; create
separate contexts for independent model instances. `quick_ai_cancel()` may be called
from another thread while generation is running.

## Windows Build

Use a static-library build and static MSVC CRT when producing the required
single DLL:

```powershell
meson setup build-causallm-abi `
  -Dplatform=windows `
  -Denable-transformer=true `
  -Denable-causallm-single-dll-abi=true `
  -Denable-test=false `
  -Ddefault_library=static `
  -Db_vscrt=mt
ninja -C build-causallm-abi quick_ai quick_ai_sample
```

Verify that the DLL does not depend on nntrainer plugin DLLs, third-party DLLs,
or MSVC runtime DLLs:

```powershell
dumpbin /dependents build-causallm-abi\Applications\CausalLM\quick_ai.dll
```

Only Windows system DLLs should be listed.

## Smoke Test

The sample dynamically loads the DLL without modifying `PATH`:

```powershell
build-causallm-abi\Applications\CausalLM\quick_ai_sample.exe `
  build-causallm-abi\Applications\CausalLM\quick_ai.dll `
  C:\nntr_weights\qwen3-0.6b-full-q40 `
  "Write one short sentence about local inference."
```

See `MODEL_PACKAGE.md` for the required model directory layout.
