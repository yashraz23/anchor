# Engine Arguments

Engine arguments control the behaviour of the vLLM engine.

## Batch size and context length

You can reduce memory usage by limiting the context length of the model
(`max_model_len`) and the maximum batch size (`max_num_seqs`).

```bash
vllm serve meta-llama/Llama-3.1-8B \
  --max-model-len 4096 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.9
```

Raise `--gpu-memory-utilization` when the server has the card to itself.

## Eager mode

Pass `--enforce-eager` to turn off the torch.compile integration and run
entirely in eager mode. This also disables CUDA graphs, which reduces memory
use at the cost of throughput.
