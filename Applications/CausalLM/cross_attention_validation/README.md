# Cross-Attention Validation App

This folder provides a standalone validation flow for the
`fully_connected x3 -> mha_core -> fully_connected` graph in CausalLM.

## Files

- `main.cpp`: nntrainer executable source
- `model_reference.py`: PyTorch reference model + input/expected export
- `weights_converter.py`: PyTorch checkpoint to nntrainer `.bin` converter

## End-to-End Flow

1. Build CausalLM (contains `nntr_cross_attention_validation`).

```
meson setup builddir -Denable-transformer=true
ninja -C builddir nntr_cross_attention_validation
```

2. Generate reference artifacts.

```
python3 Applications/CausalLM/cross_attention_validation/model_reference.py \
  --output_dir /tmp/cross_attention_validation
```

3. Convert PyTorch weights to nntrainer format.

```
python3 Applications/CausalLM/cross_attention_validation/weights_converter.py \
  --input_checkpoint /tmp/cross_attention_validation/torch_weights.pt \
  --output_weight /tmp/cross_attention_validation/nntr_weights.bin
```

4. Run nntrainer validation.

```
./builddir/Applications/CausalLM/cross_attention_validation/\
nntr_cross_attention_validation \
  --decoder_input /tmp/cross_attention_validation/decoder_input.bin \
  --encoder_input /tmp/cross_attention_validation/encoder_input.bin \
  --expected_output /tmp/cross_attention_validation/expected_output.bin \
  --weight /tmp/cross_attention_validation/nntr_weights.bin
```

If `mha_core` cross-attention property is not included, run with
`--disable_cross_property` for compatibility checks.
