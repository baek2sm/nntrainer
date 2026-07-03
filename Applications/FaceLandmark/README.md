# Face Landmark Application

This is an nntrainer inference application for a PFLD-style face-landmark model.
It loads a TorchScript model, converts the weights into nntrainer binary format,
and verifies numerical parity between PyTorch and nntrainer.

## Model

The network expects a single-channel `128x128` grayscale image and outputs
`364` (13 points x 2 x 14) values.

- Conv + PReLU stem
- Mixed depthwise residual blocks with squeeze-and-excite modules
- Multi-scale tail pooling and convolution branches
- Fully-connected output layer

A custom nntrainer pluggable `prelu` layer is provided because the official
layer set does not include per-channel PReLU.

## Build

The application is built when `enable-ccapi=true`:

```bash
meson setup build -Denable-ccapi=true
ninja -C build nntrainer_face_landmark
```

The build produces:

- `build/Applications/FaceLandmark/nntrainer_face_landmark`
- `build/Applications/FaceLandmark/libprelu_layer.so`

## Convert weights

```bash
python3 Applications/FaceLandmark/convert_weights.py \
    /path/to/face_landmark.pt \
    Applications/FaceLandmark/face_landmark_nntrainer.bin
```

The converter fuses Conv+BN pairs into a single conv filter+bias and writes the
weights in the exact order required by `MODEL_FORMAT_BIN`.

## Run inference

```bash
./build/Applications/FaceLandmark/nntrainer_face_landmark \
    Applications/FaceLandmark/face_landmark_nntrainer.bin \
    [input.raw]
```

If `input.raw` is omitted, a zero input is used. The input file must contain
`128 * 128` float32 values in NCHW order.

## Verify numerical parity

```bash
python3 Applications/FaceLandmark/verify_parity.py \
    /path/to/face_landmark.pt \
    build/Applications/FaceLandmark/nntrainer_face_landmark \
    Applications/FaceLandmark/face_landmark_nntrainer.bin
```

The expected max absolute difference against PyTorch is below `1e-5` for typical
normalized inputs.

## Files

| File | Description |
|---|---|
| `jni/main.cpp` | nntrainer graph builder and inference executable |
| `prelu_layer.cpp/h` | Custom per-channel PReLU pluggable layer |
| `convert_weights.py` | PyTorch -> nntrainer binary weight converter |
| `verify_parity.py` | Numerical parity verification script |
| `meson.build` | Build rules for the application and PReLU plugin |
