# SimpleConv2D Example

This example demonstrates converting weights from PyTorch to NNTrainer for a simple Conv2D model with only one layer.

## Purpose
- Show the complete workflow: PyTorch → Weight Conversion → NNTrainer Inference
- Verify that both frameworks produce identical outputs
- Easy reference for users to understand weight conversion process

## Model Architecture
```
Input: [1, 3, 32, 32] (NCHW format, all values = 1.0)
  ↓
Conv2D:
  - In channels: 3
  - Out channels: 3
  - Kernel size: 3×3
  - Stride: 1
  - Padding: 1 (same)
  - Bias: Enabled
  ↓
Output: [1, 3, 32, 32] (NCHW format)
```

## How to Run

### Option 1: Quick Test (No PyTorch Required)

A pre-generated `conv2d_weights.bin` file is included for immediate testing:

```bash
cd build  # From nntrainer root
ninja
./Applications/SimpleConv2D/jni/nntrainer_simple_conv2d
```

Expected output:
```
==========================================
SimpleConv2D: NNTrainer Inference
==========================================

[Step 1] Creating model...
Model created successfully

[Step 2] Loading weights from ../PyTorch/conv2d_weights.bin
Weights loaded successfully

[Step 3] Creating all-ones input...
Input shape: [1, 3, 32, 32]

[Step 4] Running inference...
Output shape: [1, 3, 32, 32]

[Step 5] Comparison values:
nntrainer_input[0:5] = [1.000000, 1.000000, 1.000000, 1.000000, 1.000000]
nntrainer_output[0:5] = [156.000000, 225.000000, 225.000000, 225.000000, 225.000000]
```

**Note**: With the test weights, `output[0] = 156.0` is the expected value for the convolution at position (0,0,0,0).

### Option 2: Full Workflow with PyTorch

#### 1. Generate weights with PyTorch
```bash
cd Applications/SimpleConv2D/PyTorch
pip install torch numpy
python3 main.py
```

This will:
- Create the Conv2D model
- Run inference on all-ones input
- Save weights as `conv2d_weights.bin`
- Print key values for comparison

#### 2. Run NNTrainer inference
From the build directory:
```bash
cd build/Applications/SimpleConv2D/jni
./nntrainer_simple_conv2d
```

Expected output:
```
==========================================
SimpleConv2D: NNTrainer Inference
==========================================

[Step 1] Creating model...
Model created successfully

[Step 2] Loading weights from ../PyTorch/conv2d_weights.bin
Weights loaded successfully

[Step 3] Creating all-ones input...
Input shape: [1, 3, 32, 32]

[Step 4] Running inference...
Output shape: [1, 3, 32, 32]

[Step 5] Comparison values:
nntrainer_input[0:5] = [1.000000, 1.000000, 1.000000, 1.000000, 1.000000]
nntrainer_output[0:5] = [...]

==========================================
Inference completed successfully!
Compare these values with PyTorch output.
==========================================
```

### 3. Verify Results

**Quick Test Verification**:
The pre-generated weights use a simple pattern for easy verification:
- `weight[out_c, in_c, h, w] = out_c*100 + in_c*10 + h + w`
- `bias[out_c] = out_c * 1000`

For `output[0,0,0,0]` (first output value):
- Input: All ones (1.0)
- Convolution sum across 3 input channels: 12 + 52 + 92 = 156
- With bias[0] = 0: **Total = 156**
- ✓ Matches NNTrainer output!

**With PyTorch Workflow**:
Compare the values printed by PyTorch script:
```python
pytorch_input[0:5] = [1.0, 1.0, 1.0, 1.0, 1.0]
pytorch_output[0:5] = [value1, value2, value3, value4, value5]
conv1_weight[0,0,0,:] = [w1, w2, w3]
conv1_bias = [b1, b2, b3]
```

With NNTrainer output:
```cpp
nntrainer_input[0:5] = [1.000000, 1.000000, 1.000000, 1.000000, 1.000000]
nntrainer_output[0:5] = [value1, value2, value3, value4, value5]
```

**Expected Result**: The output values should be identical (within floating-point precision tolerance < 1e-6).

## Files Description

### `PyTorch/main.py`
Single script that:
1. Creates a Conv2D model (3→3 channels, 3×3 kernel)
2. Initializes input with all ones
3. Runs inference
4. Saves weights in NNTrainer binary format
5. Prints all key values for easy comparison

Key features:
- Follows the pattern of `Applications/LLaMA/PyTorch/weights_converter.py`
- Uses `model.state_dict()` to access parameters
- Defines `save_weight()` function for direct numpy array saving
- **No permute needed for Conv2D** - shape (out, in, h, w) matches NNTrainer
- Verifies file size matches expected (336 bytes)
- Outputs formatted comparison values

The weight saving pattern used:
```python
def save_weight(weight):
    """Save weight tensor to binary file"""
    np.array(weight, dtype=dtype).tofile(file)

# In save_conv2d_for_nntrainer():
save_weight(params['conv1.weight'])  # Shape: (3, 3, 3, 3)
save_weight(params['conv1.bias'])    # Shape: (3,)
```

### `jni/main.cpp`
C++ program that:
1. Creates the same Conv2D model programmatically
2. Loads weights from PyTorch-generated `.bin` file
3. Creates all-ones input
4. Runs inference
5. Prints comparison values

Key features:
- Uses `ml::train::createLayer()` for programmatic model creation
- Loads binary weights with `model->load(...)`
- Uses NCHW format consistently

### Weight Format
The binary file contains (in order):
1. Conv2D weight tensor: shape `(out_channels, in_channels, kernel_h, kernel_w)` = `(3, 3, 3, 3)`
2. Conv2D bias tensor: shape `(out_channels,)` = `(3,)`

Total size: (3×3×3×3 + 3) × 4 bytes (float32) = 336 bytes

**Saving Pattern (following Applications/LLaMA/PyTorch/weights_converter.py)**:
```python
def save_weight(weight):
    np.array(weight, dtype=dtype).tofile(file)

# Access parameters via state_dict
params = model.state_dict()
save_weight(params['conv1.weight'])  # No permute needed!
save_weight(params['conv1.bias'])
```

**Key Point**: For Conv2D layers, **no permute is needed** because PyTorch uses the same shape (out, in, h, w) as NNTrainer expects. This differs from Linear layers which require permute(1, 0).

## Key Points

- Both PyTorch and NNTrainer use **NCHW** format for tensors
- For Conv2D, parameters are saved in order: **weight, bias**
- torchconverter provides automatic parameter reordering and saving
- The model uses default PyTorch initialization, which NNTrainer loads correctly
- Using "same" padding ensures output dimensions match input dimensions

## Troubleshooting

### Error: "Cannot open weight file" or "file not found"
**Solution**:
- The build system automatically copies `PyTorch/` directory to the build folder
- Ensure you've rebuilt after creating new weight files:
  ```bash
  cd build
  ninja
  ```
- If running from source directly, the executable looks for `../PyTorch/conv2d_weights.bin`
- Verify the file exists: `ls build/Applications/SimpleConv2D/PyTorch/conv2d_weights.bin`

### Error: "File size mismatch"
**Solution:**
- The expected file size is 336 bytes
- If different, verify the model architecture matches exactly
- Check that you're using the same parameters in both PyTorch and NNTrainer

### Output values don't match
**Solution:**
- Verify that both frameworks use the exact same parameters:
  - Kernel size: 3×3
  - Stride: 1
  - Padding: same
  - Bias: enabled
- Check that input is all ones in both cases
- Note: Floating-point precision might cause tiny differences (< 1e-6)

## Reference

This example shows the minimal workflow for:
1. Training/configuring a model in PyTorch
2. Converting weights to NNTrainer format using `torchconverter`
3. Loading and running inference in NNTrainer

For more complex models, refer to:
- `Applications/VGG` - Complete training example with weight conversion
- `Applications/YOLOv3` - Complex network with custom layers and programmatic model creation
- `tools/pyutils/torchconverter.py` - Weight conversion utilities documentation

## Understanding torchconverter

The `torchconverter` module provides automatic parameter translation:

```python
from torchconverter import params_translated

# Automatically translates parameters in correct order
for name, param in params_translated(model):
    # Save to binary file
    np.array(param.detach().cpu(), dtype=np.float32).tofile(f)
```

For Conv2D, it handles:
- Weight tensor shape: `(out, in, h, w)`
- Bias tensor shape: `(out,)`
- Correct ordering for NNTrainer

No manual reordering needed for standard layers!
