from tinyGPT import TinyGPTModel
from tinyGPT import block_size, batch_size
import torch
import torch.nn as nn
import argparse
import os
import time
import nvtx
from contextlib import nullcontext


default_max_iters = 1000
eval_interval = 300
learning_rate = 3e-4
eval_iters = 500


torch.manual_seed(1337)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# data loading
def get_batch(data):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


def make_autocast_ctx(amp_dtype):
    # Only CUDA autocast is relevant for your setup
    if amp_dtype is None or device != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, amp_dtype=None):
    out = {}
    model.eval()
    autocast_ctx = make_autocast_ctx(amp_dtype)

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            data = train_data if split == 'train' else val_data
            X, Y = get_batch(data)
            with autocast_ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out


def train(model, train_data, val_data, max_iters, amp_dtype=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    autocast_ctx = make_autocast_ctx(amp_dtype)

    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, amp_dtype=amp_dtype)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch(train_data)

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            logits, loss = model(xb, yb)

        loss.backward()
        optimizer.step()

    return model


def load_shakespeare_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data, vocab_size, decode, encode


def main():
    parser = argparse.ArgumentParser(description="TinyGPT Model: Train or Load")
    parser.add_argument('--load', type=str, help="Path to a saved model to load.")
    parser.add_argument('--train', action='store_true', help="Train the model.")
    parser.add_argument('--iters', type=int, default=default_max_iters, help="Number of training iterations.")
    parser.add_argument('--kv', action='store_true', help="Use kv-cache.")
    parser.add_argument('--tokens', type=int, default=(block_size-1), help="How many output tokens to generate.")
    parser.add_argument('--bf16', action='store_true', help="Use bf16 autocast on CUDA (Ampere+).")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for generation (default 1")
    args = parser.parse_args()

    print("Using", device)
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is not available.")

    # Decide AMP dtype
    amp_dtype = None
    if args.bf16:
        if device == "cuda" and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            print("[AMP] Using bf16 autocast.")
        else:
            print("[WARN] bf16 requested but not supported (or not on CUDA). Using fp32.")

    train_data, val_data, vocab_size, decode, encode = load_shakespeare_dataset('../input.txt')

    use_cache = args.kv
    if args.train:
        use_cache = False  # safest: don't train with kv-cache

    model = TinyGPTModel(vocab_size, use_cache)
    if args.load:
        if os.path.exists(args.load):
            print(f"Loading model from {args.load}...")
            model.load_state_dict(torch.load(args.load, map_location=device))
            model.eval()
        else:
            print(f"Error: The specified model file '{args.load}' does not exist.")
            exit(1)
    else:
        print("No model loaded. Creating a new TinyGPT model...")

    model = model.to(device)

    if args.train:
        print("Training the model for", str(args.iters), "iters...")
        model.train()
        model = train(model, train_data, val_data, args.iters, amp_dtype=amp_dtype)
        model_path = args.load if args.load else "tinygpt_model.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    else:
        if not args.load:
            print("Error: Model was created from scratch and not trained.")
            exit(1)

    model = torch.compile(model, mode="reduce-overhead")
    model.eval()

    user_prompt = encode("")
    user_context = torch.LongTensor(user_prompt).to(device)
    user_context = torch.unsqueeze(user_context, 0)

    empty_context = torch.zeros((args.batch_size, 1), dtype=torch.long, device=device)
    user_context = empty_context

    autocast_ctx = make_autocast_ctx(amp_dtype)

    # Warmup / first timing (optionally synchronize for accurate CUDA timing)
    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()
    with autocast_ctx:
        tokens = model.generate(user_context, max_new_tokens=args.tokens)[0].tolist()
    if device == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    print(decode(tokens))
    print(f"Generation took {end_time - start_time:.4f} seconds")

    print("---------------- HEAT done ------------------\n")

    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()
    torch.cuda.nvtx.range_push("sh_infer")
    tokens = model.generate(user_context, max_new_tokens=args.tokens)[0].tolist()
    torch.cuda.nvtx.range_pop()
    if device == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    print(decode(tokens))
    print(f"Generation took {end_time - start_time:.4f} seconds")


main()
