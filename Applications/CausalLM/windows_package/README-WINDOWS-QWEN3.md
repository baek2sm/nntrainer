# NNTrainer Qwen3 0.6B Windows Package

This folder contains a Windows build of NNTrainer CausalLM, Qwen3 0.6B
model files, and helper scripts for command-line inference and an
OpenAI-compatible local REST server.

## Contents

- `bin/nntr_causallm.exe`: command-line CausalLM runner.
- `bin/nntr_causallm_openai_server.exe`: OpenAI-compatible REST server.
- `bin/**/*.dll`: NNTrainer CausalLM custom layer plugins and runtime DLLs.
- `model/`: Qwen3 0.6B config, tokenizer, and NNTrainer weight files.
- `run_cli.ps1`: smoke-test CLI inference.
- `run_openai_server.ps1`: start the local REST server.

## CLI Smoke Test

```powershell
powershell -ExecutionPolicy Bypass -File .\run_cli.ps1
```

The output should contain a normal completion sentence such as
`Paris, and the capital of Italy is Rome.`

## OpenAI-Compatible Server

Start the server:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_openai_server.ps1 -Port 8000
```

Query it from another terminal:

```powershell
$body = @{
  model = "qwen3-0.6b"
  messages = @(@{ role = "user"; content = "Say exactly: Hello from NNTrainer." })
} | ConvertTo-Json -Depth 4

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/v1/chat/completions `
  -ContentType application/json `
  -Body $body
```

Supported endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`

Chat completions support OpenAI-style SSE streaming with `stream = $true`:

```powershell
$body = @{
  model = "qwen3-0.6b"
  stream = $true
  max_tokens = 32
  messages = @(@{ role = "user"; content = "Say exactly: Hello from NNTrainer." })
} | ConvertTo-Json -Depth 4

Invoke-WebRequest -Method Post `
  -Uri http://127.0.0.1:8000/v1/chat/completions `
  -ContentType application/json `
  -Body $body
```

The stream emits `chat.completion.chunk` frames followed by `data: [DONE]`.
The server disables Qwen thinking mode by default and strips
`<think>...</think>` and special end markers from the returned OpenAI
`content`/`text` field.

## Runtime Notes

This build targets x64 Windows and expects the Microsoft Visual C++ 2015-2022
runtime to be installed. The package includes the common MSVC runtime DLLs in
`bin/`; if Windows still reports a missing runtime DLL, install the official
Microsoft Visual C++ Redistributable for x64.
