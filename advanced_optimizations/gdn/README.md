# Gated DeltaNet, step by step

Read and run the files in this order:

1. `../gpt_mha.py` — regular softmax attention and its growing KV history.
2. `linear_attention.py` — removes softmax, proves that quadratic causal
   attention can be reassociated into recurrent `S @ q`, and demonstrates
   `prefill(4) + decode(1) + decode(1)`.
3. `gdn.py` — replaces additive writes with Delta Rule, then adds dynamic
   `alpha` decay and `beta` write strength.
4. `gdn_explained.html` — the same path with small matrices and a mapping to
   GigaChat/TensorRT-LLM chunkwise prefill.

Run from the repository root:

```bash
python3 advanced_optimizations/gdn/linear_attention.py
python3 advanced_optimizations/gdn/gdn.py
```

Both scripts are self-contained and use generated tensors; training data and a
GPU are not required.
