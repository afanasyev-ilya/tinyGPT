

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
# Generation took 16.9763 seconds

vs (using KV-cache):

```
python3 ./shakespeare.py --load ./tinygpt_model.pt --tokens 2000 --kv
```
Results into :
# Generation took 15.8209 seconds

TODO why acceleration is poor