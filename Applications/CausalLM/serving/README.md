# Native CausalLM serving

Build and run through the existing C++ application:

```sh
meson setup --reconfigure build -Denable-transformer=true -Denable-causallm-server=true
ninja -C build Applications/CausalLM/nntr_causallm
build/Applications/CausalLM/nntr_causallm /path/to/model --serve \
  --host 127.0.0.1 --port 8080 --model-id local-model \
  --max-sessions 32 --queue-capacity 32
```

The HTTP server, request queue, model execution, sessions and KV management all
run inside this process. There is no Python runtime or subprocess protocol.
Server support is optional at build time. The server defaults to CPU, paged KV
and prefix reuse when those environment variables are unset. Explicitly disabling
prefix reuse is supported; selecting a non-CPU or legacy KV backend is rejected.
The model must use batch size 1. Model loading completes before the socket binds.

## Architecture

`HttpServer` parses HTTP/JSON with cpp-httplib, using a bounded pool of HTTP
workers. `Service` owns a bounded FIFO of tasks and a single inference worker.
Template rendering and model/cache operations execute on that worker; an HTTP
thread never calls the model. Session metadata has a short mutex-protected
control path, so create/list/delete do not wait for inference. Native cache
release is deferred to the model worker. Queue saturation returns 503. At most
six HTTP workers wait on inference or serve streams, leaving two for controls.

Logical sessions own native `KVCacheManager` session IDs. The native default
session stays empty so any conversation can be deselected and destroyed.
Sessions share the model weights and graph; native paged storage owns their KV
pages. The first request settles the prefill plane before allocating storage.

Every completion contains the **full prompt or complete message history**.
Chat requests use the model's existing `ChatTemplate`; no generic role delimiters
are substituted if the model has no template. Each run starts at logical
position zero. The existing token-prefix cache compares actual token IDs and
reuses matching committed pages, including across sessions. An edited history
therefore cannot blindly append to an unrelated cache. Usage reports the native
cache's reused token count. Without prefix caching, the full prompt is recomputed.

### DRAM and Flash tiers

Enable the filesystem tier with explicit byte budgets. Select a directory on
the SSD/flash device you want to use (a tmpfs such as `/dev/shm` uses RAM).

```sh
NNTR_ENGINE=cpu NNTR_CPU_KV_LEGACY=0 \
NNTR_KV_PAGED=1 NNTR_KV_PREFIX_CACHE=1 \
NNTR_KV_PAGE_TOKENS=16 \
NNTR_KV_DRAM_BUDGET_BYTES=268435456 \
NNTR_KV_FLASH_DIR=/path/on/ssd/nntr-kv \
NNTR_KV_FLASH_BUDGET_BYTES=2147483648 \
NNTR_KV_RESTORE_RESERVE_BYTES=16777216 \
NNTR_KV_CPU_TRACE=1 \
build-kv-cpu/Applications/CausalLM/nntr_causallm \
  build-kv-cpu/models/qwen3-0.6b-fp32 --serve \
  --host 127.0.0.1 --port 8080 --model-id qwen3-0.6b \
  --max-sessions 32 --queue-capacity 32
```

The example limits the KV page pool to 256 MiB and Flash files to 2 GiB, with
a target of 16 MiB free in the pool after idle session pages are released.
It uses 16-token pages so short conversations also publish reusable blocks.
The default page width is tuned for attention throughput (512 tokens on x86,
128 elsewhere); histories shorter than one page cannot spill full blocks.
The DRAM limit covers rounded KV allocations, including idle pool pages. It
does not include model weights, graph workspace, cache metadata or filesystem
page cache. The Flash limit covers logical file sizes including headers;
filesystem allocation units and directory metadata are additional overhead.

Unused resident LRU blocks move to Flash instead of being discarded. Exact
token-prefix matches restore complete K/V blocks into the same bounded pool;
the loader reclaims space before allocating a block. Cold LRU leaves are
deleted when Flash fills, preserving parent-chain reachability. Files are
validated by header, length and checksum before publishing restored pages.
Capacity misses, missing/corrupt files and write failures fall back to
recomputation or ordinary cache eviction. A request's active working set must
still fit in DRAM; active session/view pages cannot be swapped out.

In Flash mode, successful serving requests release their native session page
references immediately, so idle sessions do not pin the entire DRAM budget.
Logical session IDs and the `/sessions` API retain their existing lifetime
and LRU capacity rules. Every next request must still include the complete
history. Full token pages can be restored across session eviction or session
deletion; partial tail pages are recomputed. `max_sessions` is a logical
session limit, independent of the cache byte budgets.

`NNTR_KV_PREFIX_CACHE_BLOCKS` limits resident indexed blocks. Set it to `0` to
use only the byte budgets. `NNTR_KV_RESTORE_RESERVE_BYTES=0` defaults to one
full K/V block across all owner layers. The reserve is best effort while
pages are pinned, and the DRAM budget must fit a block plus the reserve.
`NNTR_KV_FLASH_BUDGET_BYTES=0` disables the Flash tier; enabling it requires
paged CPU prefix caching and a finite DRAM budget.

`NNTR_KV_CPU_TRACE=1` prints `resident_blocks`, `flash_blocks`, `flash_bytes`,
`flash_writes`, `flash_hits` and `flash_errors` at prefix lookup. Cold transfers
are synchronous on the inference worker; control APIs remain on their
separate metadata path. Files use private per-cache subdirectories and are
removed at normal destruction. This is a process-local performance cache:
server restarts do not recover sessions or reuse old files. An unclean process
exit can leave its private directory for the operator to remove.

### Android build and validation

After building/installing the Android engine with `build_android.sh`, enable
the server in the standalone application graph as well:

```sh
meson configure builddir_app/cpu -Denable-causallm-server=true
ninja -C builddir_app/cpu nntr_causallm
```

Deploy `nntr_causallm`, `libcausallm.so`, `libnntrainer.so`,
`libccapi-nntrainer.so` and the matching NDK `libc++_shared.so` together. Set
`LD_LIBRARY_PATH` to that runtime directory and `TMPDIR` to a writable device
directory. Use a writable path such as `/data/local/tmp/.../flash` for
`NNTR_KV_FLASH_DIR`. The serving/session/API options are the same as Ubuntu.

The existing NDK test harness now includes `unittest_serving` alongside
`unittest_kv_cache_core`. Set `NNTR_SERVING_FIXTURE_DIR` to the deployed
`test/unittest/models/causallm_reference/qwen2_tiny` directory. The full Qwen3
HTTP test accepts `NNTR_SERVING_QWEN3_DIR` for a converted model directory;
it is separate from the small fixture tests.

Verified on Ubuntu x86-64 and a Galaxy S23 Ultra (SM-S918N, Android 14,
arm64-v8a), using NDK 27.2.12479018 / android-29:

| Check | Ubuntu | Android device |
| --- | --- | --- |
| KV core, including Flash I/O failures and mixed encodings | 58 passed | 58 passed |
| Serving, HTTP/SSE, session controls and native Flash reuse | 18 passed | 18 passed |
| Qwen3 0.6B FP32, two live logical sessions, A/B/A/B | passed | passed |
| Reused tokens on the second A and B requests | 24 / 24 | 24 / 24 |
| Flash loads observed / I/O errors at final lookup | 12 / 0 | 12 / 0 |
| Session listing during inference (one observation) | 0.85 ms | 8.09 ms |

Both full-model runs used `--max-sessions 2`, 8-token pages, 64 MiB DRAM,
128 MiB Flash and a 62 MiB reserve target to force demotion. A/B generation
after restoration matched each original request; SSE text also matched, and
the tested outputs agreed across platforms. These latency observations are
functional checks, not a performance benchmark. The 64 MiB test budget fits
these short requests; longer active contexts require more DRAM.

Results, XML and the HTTP validation driver are under the ignored
`build-kv-cpu/validation/tiered-cache/` and
`build-kv-android/validation/tiered-cache/` directories. The combined report is
`build-kv-cpu/validation/tiered-cache/validation.json`.

This separation follows llama.cpp's HTTP routes, request queue, inference context
and sequence cache design:
https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README-dev.md
The current CausalLM execution API runs a complete generation per task; requests
execute in FIFO order rather than continuous batching.

## OpenAI-compatible endpoints

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `GET /health`

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"user","content":"Hello"}],"max_tokens":64,"temperature":0,"stream":false}'
```

Chat Completions clients can point their `base_url` at
`http://127.0.0.1:8080/v1` and use the advertised model ID. Responses include
`id`, `object`, `created`, `model`, `choices`, `finish_reason`, and token `usage`.
Errors use the OpenAI `error` object shape. The optional extra `session_id`
response field identifies the cache session.

Supported inputs: `model`, text `messages` (system/developer/user/assistant),
string `prompt` for Completions, `max_tokens` or `max_completion_tokens`,
`temperature`, `top_p`, `seed`, `n=1`, boolean `stream`, and `user`.
The local `chat_template_kwargs` extension accepts boolean `enable_thinking`
for model templates such as Qwen3; clients can pass it in their extra body fields.
Unsupported parameters are rejected rather than silently ignored. This is text
Chat Completions/Completions compatibility; Responses, tools, multimodal content,
stop sequences, penalties and logprobs are not implemented.
The request body limit is 1 MiB. Input exceeding the context or prefill window
is rejected instead of truncated. An inference failure invalidates and releases
its session; validation errors preserve an existing session.

## Sessions and reclamation

Standard OpenAI clients need no session extension. Their requests allocate
reclaimable cache sessions, and the native prefix cache may reuse a matching
history. The least recently used reclaimable session is released when logical
session capacity is reached.

For explicit retention:

```sh
curl -X POST http://127.0.0.1:8080/sessions
# {"id":"session-1","object":"session"}
```

Send that ID in the optional `session_id` completion field. Continue sending the
**complete history**, including assistant messages. Explicit sessions are pinned
until deleted or process exit. If every session is pinned, creation returns 503.
`GET /sessions` lists sessions with `state: idle` or `generating`.
`DELETE /sessions/{id}` removes an idle session immediately and schedules its
KV release on the model worker. Deleting a generating session returns 409
`session_busy`; it never blocks behind that generation. This is a
local extension, not an OpenAI session endpoint. Native shared prefix pages can
remain resident according to the prefix cache's own bounded eviction policy.
Queue shutdown drains accepted work, then releases sessions on the model worker.

## Verification

```sh
ninja -C build Applications/CausalLM/unittest_serving
build/Applications/CausalLM/unittest_serving
```

The C++ tests cover request/response schemas, full chat histories, invalid fields,
queue saturation, session capacity/LRU, failure cleanup, native HTTP requests,
and actual tiny-Qwen2 paged cache execution across interleaved and edited
histories. The HTTP test requires permission to bind a localhost socket.

The opt-in full-size integration test `Serving.Qwen3_06B_NativeHttp` loads
`<builddir>/models/qwen3-0.6b-fp32` when available. Set
`NNTR_SERVING_QWEN3_DIR` to select another converted Qwen3 0.6B directory. It
tests real HTTP chat completions, two interleaved conversations, history replay,
cache reuse, context rejection, streaming, cancellation and session deletion. It writes
its responses to `/tmp/nntrainer-qwen3-serving-results.json`.

## Streaming

Both completion endpoints accept `stream:true`. Chat responses use
`chat.completion.chunk` with an initial assistant role delta, content deltas,
a final `finish_reason`, and `data: [DONE]`. Text completions use `text` chunks.
`stream_options:{"include_usage":true}` adds a final usage chunk with empty
`choices` before `[DONE]`; other chunks then carry `usage:null`.

```sh
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"user","content":"Hello"}],"stream":true,"stream_options":{"include_usage":true},"max_tokens":64}'
```

Every request owns a bounded token mailbox (128 events, 256 KiB). Model callbacks
never block on client reads. Overflow cancels that generation. Disconnects are
checked while queued and during SSE writes/heartbeats; cancellation is observed
by the model at its next decoded-output boundary (prefill is not interrupted).
An active cancelled generation's session, including a pinned session, is deleted
on the inference worker. A cancellation before execution leaves an existing
session intact. Other sessions and requests are unaffected; recreate a deleted
pinned session before retrying. Completed streams retain their session normally.

Validation errors before generation starts retain HTTP status/error JSON. Once
SSE starts, inference failures are delivered as an SSE OpenAI-shaped error then
`[DONE]`, without a successful finish chunk. EOS marker text is held across token
callbacks and removed before emission, while preserving UTF-8 boundaries.
Socket writes have a 30-second timeout. Shutdown still drains accepted tasks;
stream queues bound memory even if a reader disappears.

Streaming response format reference:
https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events

Bodyless requests such as `curl -X POST http://127.0.0.1:8080/sessions`
require no Content-Length header. Requests without body framing are treated as
empty according to RFC 9112 section 6.3, rather than waiting for connection EOF.
Ambiguous or unsupported framing receives 400. Controls bypass the inference
queue; an exhausted logical session capacity still returns 503 promptly.
