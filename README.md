

### train model

```
cd tinyGPT_train
python3 ./shakespeare.py --train
```

### inference

Main point of this project is to memorize KV-cache structure for decode stage. After we trained the model:

```
python3 ./shakespeare.py --load ./tinygpt_model.pt --tokens 2000
```

Results into:
```
Generation took 16.9763 seconds
```

vs (using KV-cache):

```
python3 ./shakespeare.py --load ./tinygpt_model.pt --tokens 2000 --kv
```
Results into:
```
Generation took 15.8209 seconds
```

why this is happening? this is because our model is not attention-bound (small batch, lots of unfused kernels, etc). To address this, we do the following:

1) use Flash Attention and BF16
2) use ROPE instead of pos encoding
3) use torch compiler (fuses more kernels)
4) use batch size > 1, e.g. 64 (more compute per kernel)
5) fuse QKV

This features are dropped into optTinyGPT.py file:

```
python3 ./shakespeare.py --load ./tinygpt_model.pt --tokens 255 --bf16 --batch_size 128 --opt
Generation took 23.7522 seconds
```

```
python3 ./shakespeare.py --load ./tinygpt_model.pt --tokens 255 --bf16 --batch_size 128 --opt --kv
Generation took 1.8248 seconds
```

### advanced optimizations

Small, self-contained implementations live in `advanced_optimizations/`.
`advanced_optimizations/README.md` describes the underlying LLM operations
(math, shapes, toy examples) for DeepSeek and GigaChat.

```
advanced_optimizations/
  mha_base.py   base model: multi-head attention, everything else starts from it
  README.md     LLM operations reference
  mha/          bigram, single-head and MHA + FFN + LayerNorm variants
  dsa/          DeepSeek Sparse Attention (DSA indexer, MISA)
  moe/          Mixture of Experts: sigmoid router, top-k, shared expert
  gdn/          Gated DeltaNet / linear attention
```

Run the scripts from their own folder (they read `input.txt` by a relative path):

```bash
cd advanced_optimizations/moe && python3 moe.py
```

The Gated DeltaNet path starts from the same Q/K/V operations as
`mha_base.py`, derives recurrent linear attention, and then adds Delta Rule,
alpha decay and beta write strength:

```bash
python3 advanced_optimizations/gdn/linear_attention.py
python3 advanced_optimizations/gdn/gdn.py
```

Open `advanced_optimizations/gdn/gdn_explained.html` for the illustrated
prefill/decode walkthrough and the mapping to GigaChat/TensorRT-LLM.
