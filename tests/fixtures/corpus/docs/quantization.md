# Quantization

vLLM supports several quantization methods for the KV cache.

## FP8 KV cache

Set `--kv-cache-dtype` to `fp8` to store the KV cache in 8-bit floating point.
Two formats are available: `fp8_e4m3` offers more precision, and `fp8_e5m2`
offers more range.

Scaling factors are read from the checkpoint when present.
