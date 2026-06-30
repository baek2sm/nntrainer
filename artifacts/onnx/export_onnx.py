#!/usr/bin/env python3
"""Export the exact nntrainer YOLOv11m (.pt, nc=1, 832x832) to FP32 + static INT8 ONNX
and verify on x86 onnxruntime. Writes all artifacts under artifacts/onnx/."""
import os, sys, warnings, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PT = "/home/seungbaek/projects/screenaivttprobe-v2.1.0-s02-pytorch/screenaivttprobe-v2.1.0-s02-pytorch/detection/v11m_832rect_best.pt"
RES = "/home/seungbaek/projects/nntrainer/Applications/CausalLM/models/YOLOv11/res"
INPUT_832 = os.path.join(RES, "input_832.bin")
INPUT_CAT = os.path.join(RES, "input_cat.bin")
FP32 = os.path.join(HERE, "yolov11m_832_fp32.onnx")
INT8 = os.path.join(HERE, "yolov11m_832_int8.onnx")
PREP = os.path.join(HERE, "yolov11m_832_fp32_prep.onnx")
IMGSZ = 832

WARN_LOG = []
_orig_showwarning = warnings.showwarning
def _capture(message, category, filename, lineno, file=None, line=None):
    WARN_LOG.append(f"{category.__name__}: {message}")
    _orig_showwarning(message, category, filename, lineno, file, line)
warnings.showwarning = _capture

def load_bin(path):
    a = np.fromfile(path, dtype="<f4")
    assert a.size == 1*3*IMGSZ*IMGSZ, f"{path} size {a.size}"
    return a.reshape(1, 3, IMGSZ, IMGSZ)

# ---------- Step 1: FP32 export ----------
print("="*70); print("STEP 1: FP32 ONNX export"); print("="*70)
from ultralytics import YOLO
model = YOLO(PT)
print(f"loaded: {PT}")
print(f"model nc = {model.model.nc if hasattr(model.model,'nc') else 'n/a'}")
out_path = model.export(format="onnx", imgsz=IMGSZ, opset=17, simplify=True,
                        dynamic=False, batch=1, device="cpu")
print(f"ultralytics exported to: {out_path}")
import shutil
if os.path.abspath(out_path) != os.path.abspath(FP32):
    shutil.copy(out_path, FP32)
print(f"FP32 onnx saved: {FP32}")

import onnx
m = onnx.load(FP32)
def io_info(graph_io):
    info = []
    for t in graph_io:
        dims = [d.dim_value if d.dim_value>0 else (d.dim_param or "?") for d in t.type.tensor_type.shape.dim]
        info.append((t.name, dims))
    return info
IN_INFO = io_info(m.graph.input)
OUT_INFO = io_info(m.graph.output)
print("FP32 inputs :", IN_INFO)
print("FP32 outputs:", OUT_INFO)
INPUT_NAME = IN_INFO[0][0]

# ---------- Step 2: static INT8 quantization ----------
print("="*70); print("STEP 2: static INT8 quantization (QDQ, S8S8, per-channel)"); print("="*70)
from onnxruntime.quantization import (quantize_static, CalibrationDataReader,
                                      QuantFormat, QuantType, CalibrationMethod)
from onnxruntime.quantization.shape_inference import quant_pre_process

quant_pre_process(FP32, PREP, skip_symbolic_shape=False)
print(f"quant_pre_process -> {PREP}")

class Reader(CalibrationDataReader):
    def __init__(self, name, files):
        self.name = name
        self.data = [load_bin(f).astype(np.float32) for f in files]
        self.i = 0
    def get_next(self):
        if self.i >= len(self.data):
            return None
        d = {self.name: self.data[self.i]}
        self.i += 1
        return d

reader = Reader(INPUT_NAME, [INPUT_832, INPUT_CAT])
quantize_static(
    PREP, INT8, reader,
    quant_format=QuantFormat.QDQ,
    per_channel=True,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    calibrate_method=CalibrationMethod.MinMax,
    reduce_range=False,
)
print(f"INT8 onnx saved: {INT8}")

# ---------- Step 3: verify on x86 onnxruntime ----------
print("="*70); print("STEP 3: verify on x86 onnxruntime (CPU EP, ORT_ENABLE_ALL)"); print("="*70)
import onnxruntime as ort
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

x = load_bin(INPUT_832).astype(np.float32)

def run(path, tag):
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    outs = sess.run(None, {inp: x})
    print(f"\n[{tag}] input name='{inp}'  n_outputs={len(outs)}")
    for j, o in enumerate(outs):
        oname = sess.get_outputs()[j].name
        print(f"  out[{j}] name='{oname}' shape={o.shape} dtype={o.dtype}")
        print(f"    min={o.min():.6f} max={o.max():.6f} mean={o.mean():.6f}")
        print(f"    first10={np.asarray(o).ravel()[:10]}")
    return outs

outs_fp32 = run(FP32, "FP32")
outs_int8 = run(INT8, "INT8")

print("\n--- FP32 vs INT8 output diff ---")
maxdiff = 0.0
assert len(outs_fp32) == len(outs_int8)
for j in range(len(outs_fp32)):
    a = np.asarray(outs_fp32[j]).astype(np.float64)
    b = np.asarray(outs_int8[j]).astype(np.float64)
    if a.shape == b.shape:
        d = float(np.max(np.abs(a-b)))
        maxdiff = max(maxdiff, d)
        print(f"  out[{j}] max-abs-diff = {d:.6f}")
    else:
        print(f"  out[{j}] shape mismatch {a.shape} vs {b.shape}")
print(f"OVERALL max-abs-diff = {maxdiff:.6f}")

# ---------- Step 4: summary ----------
print("="*70); print("STEP 4: summary"); print("="*70)
def sz(p): return os.path.getsize(p)
def mb(p): return sz(p)/1e6
summary = []
summary.append("# YOLOv11m ONNX export notes\n")
summary.append(f"- Source .pt: `{PT}` (nc={getattr(model.model,'nc','?')}, imgsz={IMGSZ})\n")
summary.append(f"- ultralytics 8.4.70, onnxruntime {ort.__version__}, onnx {onnx.__version__}\n")
summary.append(f"\n## Files\n")
summary.append(f"- FP32: `{FP32}` ({mb(FP32):.2f} MB)\n")
summary.append(f"- INT8: `{INT8}` ({mb(INT8):.2f} MB)\n")
summary.append(f"- (intermediate prep: `{PREP}` ({mb(PREP):.2f} MB))\n")
summary.append(f"\n## I/O (from FP32 graph)\n")
summary.append(f"- input name: `{INPUT_NAME}`\n")
for n,d in IN_INFO: summary.append(f"  - in  `{n}` {d}\n")
for n,d in OUT_INFO: summary.append(f"  - out `{n}` {d}\n")
summary.append(f"\n## INT8 quantization config\n")
summary.append("- QuantFormat.QDQ, per_channel=True, activation=QInt8, weight=QInt8, MinMax, reduce_range=False (symmetric S8S8 -> MlasConvSym indirect conv on ARM)\n")
summary.append("- calibration: input_832.bin + input_cat.bin (raw f32 [1,3,832,832])\n")
summary.append(f"\n## Verification (input_832.bin, x86 CPU EP)\n")
summary.append(f"- n_outputs={len(outs_fp32)}\n")
for j in range(len(outs_fp32)):
    o=outs_fp32[j]; summary.append(f"  - FP32 out[{j}] shape={o.shape} min={o.min():.4f} max={o.max():.4f} mean={o.mean():.4f}\n")
for j in range(len(outs_int8)):
    o=outs_int8[j]; summary.append(f"  - INT8 out[{j}] shape={o.shape} min={o.min():.4f} max={o.max():.4f} mean={o.mean():.4f}\n")
summary.append(f"- FP32 vs INT8 OVERALL max-abs-diff = {maxdiff:.6f}\n")
summary.append(f"\n## Warnings ({len(WARN_LOG)})\n")
for w in WARN_LOG[:40]: summary.append(f"- {w}\n")
notes = os.path.join(HERE, "EXPORT_NOTES.md")
with open(notes, "w") as f: f.writelines(summary)
print("".join(summary))
print(f"\nwrote {notes}")
print("DONE")
