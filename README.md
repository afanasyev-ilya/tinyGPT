

### train model

```
cd final
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