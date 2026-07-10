# YOLOv11 detection (nntrainer Quick_AI app)

YOLOv11 (v11s / v11m) detection inference, run through the standard Quick_AI
application the same way the LLM (Qwen3 / Gemma3 / …) and TimmViT models are:
the model is a Factory-registered, `nntr_config.json`-driven config class
(`Yolov11Transformer`, architecture key `YOLOv11ForDetection`), so the plain
`nntr_quick_ai <model_dir>` binary runs it and `nntr_quantize <model_dir>`
quantizes it — no `YOLO_*` environment variables and no standalone binary.

The only custom op is the C2PSA spatial multi-head attention (`psa_attention`
layer, `c2psa_layer.{h,cpp}`); the rest of the graph (backbone, C3k2, SPPF,
detect head) is assembled from nntrainer's standard layers by
`yolov11_graph.h`. Post-processing (DFL decode + dist2bbox + NMS) lives in the
header-only `yolo_postprocess.h`. These four live alongside the model class in
this directory, the way gemma3 colocates `function.cpp` and gpt_oss colocates
`gpt_oss_moe_layer.cpp`.

## 1. Convert weights (PyTorch FP32 → nntrainer FP32, once)

The converter does FP32 conversion only — quantization is done by nntrainer's
own cpp quantizer (step 3), never in Python. Needs a Python env with
`ultralytics` + `torch`:

```bash
python res/YOLOv11/weight_converter.py \
  --weights /path/to/v11s_1024rect_best.pt \
  --out <model_dir>/v11s_fp32.safetensors
```

`--weights` may be `--image` mode; see `res/YOLOv11/run_pytorch.py` /
`res/YOLOv11/extract_reference.py` for generating the PyTorch reference `.bin`s
this path verifies against. The converter folds Conv+BatchNorm at convert time
(inference-only), so the nntrainer graph has no BatchNorm layers.

## 2. Run FP32 inference (config-driven)

A model directory is `config.json` + `nntr_config.json` (+ weights + ref bins).
See `res/yolov11/v11s-1024/` for the v11s, 1024×1024, nc=2 FP32 sample. With
`yolo_verify: true`, raw logits are compared against `ref_p{3,4,5}.bin`:

```bash
nntr_quick_ai res/yolov11/v11s-1024
```

Threads are set the standard way — the core `NNTR_NUM_THREADS` env var (read by
`nntrainer/utils/thread_manager.h`), not an app flag:

```bash
NNTR_NUM_THREADS=8 nntr_quick_ai res/yolov11/v11s-1024
```

## 3. Quantize (Q4_0 / Q8_0, cpp quantizer)

```bash
# Q4_0 (w4a16) — weight repack is ISA-specific: --isa X86 for an x86 file,
# --isa ARM for an Android file.
nntr_quantize res/yolov11/v11s-1024 --conv_dtype Q4_0 --isa ARM \
  --output_format safetensors --output <out_dir>

# Q8_0 (w8a16) — plain block_q8_0, ISA-independent.
nntr_quantize res/yolov11/v11s-1024 --conv_dtype Q8_0 --isa ARM \
  --output_format safetensors --output <out_dir>
```

`conv_dtype` selects which convs the cpp `Conv2DLayer::save` quantizes; the
eligible convs (block-aligned, group=1) are surfaced by
`Yolov11Transformer::getQuantizableLayerNames()`. The quantizer writes a ready
`nntr_config.json` (with `conv_dtype` set) into `<out_dir>`; run the result
with `nntr_quick_ai <out_dir>` (set `model_tensor_type: "FP32-FP16"` for FP16
activations).

## Build

The model class and graph are part of the Quick_AI app build
(`models/YOLOv11/meson.build`); the standard app build produces
`nntr_quick_ai`/`nntr_quantize` with YOLOv11 included. For Android,
`build_android.sh` + `install_android.sh` deploy the same binaries.

## Notes

- The C2PSA attention is implemented with nntrainer Tensor ops (`dotBatched`,
  `ActiFunc` softmax, in-place scale) over MemoryPool scratch tensors — no
  per-forward dynamic allocation and no direct kernel call, so FP16
  activations stay FP16 through the attention matmuls (the FP16 GEMM backend),
  with softmax accumulating in FP32 internally as the framework softmax does.
- Scale divisor is `1/sqrt(kd) = 1/sqrt(32)` (the per-head key dim).
- The default graph is built for v11m widths unless `yolo_variant: "v11s"` is
  set in `nntr_config.json`; loading v11s weights without it mismatches the
  detect head and SIGSEGVs at load.
