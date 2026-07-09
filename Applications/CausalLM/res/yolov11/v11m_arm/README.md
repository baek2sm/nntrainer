# YOLOv11m Android (w8a16, Q8_0+FP16)

832×832, nc=1, NHWC. ~340-380 ms / ~250 MB on S25U (SD8-Elite, 8 threads).

## Run

```bash
# Push configs + weights to device
adb push config.json nntr_config.json yolov11m_q8_0_w8a16.safetensors /data/local/tmp/yolov11m/

# Run
adb shell "cd /data/local/tmp/yolov11m && \
  LD_LIBRARY_PATH=/data/local/tmp/yolov11m \
  NNTR_NUM_THREADS=8 \
  /data/local/tmp/yolov11m/nntrainer_causallm ."
```

## Notes

- `conv_dtype: "Q8_0"` requires `input_format: "NHWC"`.
- Weights must be repacked with `repack_q8_0` (q8_0x4 layout).
