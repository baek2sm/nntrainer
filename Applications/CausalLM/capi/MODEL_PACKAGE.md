# quick.ai Model Package

The single-DLL C ABI loads one model directory passed to
`quick_ai_ctx_create(model_dir_utf8, ...)`. The runtime must not rely on the current
working directory, `PATH`, or environment variables for model discovery.

Required files:

- `config.json`: HuggingFace-style model architecture metadata.
- `generation_config.json`: generation defaults such as EOS/BOS tokens,
  temperature, top-k/top-p, and `do_sample`.
- `nntr_config.json`: NNTrainer runtime settings. `model_file_name` and
  `tokenizer_file` may be relative to the model directory or absolute paths.
- `tokenizer.json`: tokenizer data referenced by `nntr_config.json`.
- `tokenizer_config.json`: optional chat template; when present, chat messages
  are formatted through the template.
- `*.bin` or `*.safetensors`: NNTrainer model weights referenced by
  `nntr_config.json`.

To replace the model, place the replacement files in a new directory, update
`nntr_config.json` so `model_file_name` and `tokenizer_file` point to files in
that directory, then pass the new absolute directory path to
`quick_ai_ctx_create()`.

For CausalLM packages, the static graph length is fixed by
`nntr_config.json`. `max_seq_len` is the hard prompt-plus-generation context
window, `init_seq_len` is the maximum prefill input shape, and
`num_to_generate` is the package's maximum generation budget. The ABI rejects
requests whose tokenized prompt cannot fit with the requested `max_tokens`.

The tested Windows package shape is:

```text
package/
  quick_ai.dll
  quick_ai.h
  model/
    config.json
    generation_config.json
    nntr_config.json
    tokenizer.json
    tokenizer_config.json
    nntr_qwen3_0.6b_full_q40.bin
```
