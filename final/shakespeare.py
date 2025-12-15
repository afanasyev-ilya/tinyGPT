from tinyGPT import TinyGPTModel
from tinyGPT import block_size, batch_size
import torch
import torch.nn as nn
import argparse
import os
import time
import nvtx


default_max_iters = 1000
eval_interval = 300
learning_rate = 3e-4
eval_iters = 500


torch.manual_seed(1337)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# data loading
def get_batch(data):
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            data = train_data if split == 'train' else val_data
            X, Y = get_batch(data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



def train(model, train_data, val_data, max_iters):
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch(train_data)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return model


def load_shakespeare_dataset(path):
    # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data, vocab_size, decode, encode


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="TinyGPT Model: Train or Load")
    parser.add_argument('--load', type=str, help="Path to a saved model to load. If omitted, a new TinyGPT model will be created.")
    parser.add_argument('--train', action='store_true', help="Train the model.")
    parser.add_argument('--iters', type=int, default=default_max_iters, help="Number of training iterations (default: 5000).")
    parser.add_argument('--kv', action='store_true', help="Use kv-cache.")
    parser.add_argument('--tokens', type=int, default=(block_size-1), help="How much output tokens to generate, default 200.")
    args = parser.parse_args()

    print("Using ", device)
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is not available.")

    train_data, val_data, vocab_size, decode, encode = load_shakespeare_dataset('../input.txt')

    use_cache = args.kv
    if args.train:
        use_cache = False # not sure if it is save to train with KV-cache turned on
    model = TinyGPTModel(vocab_size, args.kv)

    # Load or create a new model based on --load argument
    if args.load:
        if os.path.exists(args.load):
            print(f"Loading model from {args.load}...")
            model.load_state_dict(torch.load(args.load))
            model.eval()
        else:
            print(f"Error: The specified model file '{args.load}' does not exist.")
            exit(1)
    else:
        print("No model loaded. Creating a new TinyGPT model...")

    # always put model to device
    model = model.to(device)

    # Check if training is requested
    if args.train:
        print("Training the model for", str(args.iters), " iters...")
        model.train()
        model = train(model, train_data, val_data, args.iters)
        model_path = args.load if args.load else "tinygpt_model.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    else:
        if not args.load:
            print("Error: Model was created from scratch and not trained.")
            exit(1)

    # generate from the model
    model.eval()

    user_prompt = encode("")
    user_context = torch.LongTensor(user_prompt).to(device)
    user_context = torch.unsqueeze(user_context, 0)

    empty_context = torch.zeros((1, 1), dtype=torch.long, device=device)
    user_context = empty_context

    start_time = time.time()
    # Only measure the generation step; do not include decode()
    tokens = model.generate(user_context, max_new_tokens=args.tokens)[0].tolist()
    end_time = time.time()

    print(decode(tokens))
    print(f"Generation took {end_time - start_time:.4f} seconds")

    print("---------------- HEAT done ------------------ \n")

    if device == "cuda":
        torch.cuda.synchronize()

    start_time = time.time()
    # Only measure the generation step; do not include decode()
    with nvtx.annotate("sh_infer"):
        tokens = model.generate(user_context, max_new_tokens=args.tokens)[0].tolist()
    if device == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    print(decode(tokens))
    print(f"Generation took {end_time - start_time:.4f} seconds")


main()
