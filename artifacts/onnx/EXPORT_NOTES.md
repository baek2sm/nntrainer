# YOLOv11m ONNX export notes
- Source .pt: `/home/seungbaek/projects/screenaivttprobe-v2.1.0-s02-pytorch/screenaivttprobe-v2.1.0-s02-pytorch/detection/v11m_832rect_best.pt` (nc=1, imgsz=832)
- ultralytics 8.4.70, onnxruntime 1.23.2, onnx 1.22.0

## Files
- FP32: `/home/seungbaek/projects/nntrainer/artifacts/onnx/yolov11m_832_fp32.onnx` (80.54 MB)
- INT8: `/home/seungbaek/projects/nntrainer/artifacts/onnx/yolov11m_832_int8.onnx` (21.04 MB)
- (intermediate prep: `/home/seungbaek/projects/nntrainer/artifacts/onnx/yolov11m_832_fp32_prep.onnx` (80.52 MB))

## I/O (from FP32 graph)
- input name: `images`
  - in  `images` [1, 3, 832, 832]
  - out `output0` [1, 5, 14196]

## INT8 quantization config
- QuantFormat.QDQ, per_channel=True, activation=QInt8, weight=QInt8, MinMax, reduce_range=False (symmetric S8S8 -> MlasConvSym indirect conv on ARM)
- calibration: input_832.bin + input_cat.bin (raw f32 [1,3,832,832])

## Verification (input_832.bin, x86 CPU EP)
- n_outputs=1
  - FP32 out[0] shape=(1, 5, 14196) min=-2.7229 max=865.4641 mean=227.6022
  - INT8 out[0] shape=(1, 5, 14196) min=-6.8977 max=865.6673 mean=226.7582
- FP32 vs INT8 OVERALL max-abs-diff = 71.203156

## Warnings (0)
