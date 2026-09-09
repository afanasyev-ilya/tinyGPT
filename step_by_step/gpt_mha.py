import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('../input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # input of size (batch, time-step, hidden_dim)
        # output of size (batch, time-step, head_size)
        B, T, hidden_dim = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # Q и K определяют, насколько каждый токен заинтересован в каждом другом
        # токене; V хранит данные токена, которые затем попадут в результат.
        wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)

        # Запрещаем смотреть в будущее. Например, для трех токенов:
        # [[score_00,      -inf,      -inf],
        #  [score_10,  score_11,      -inf],
        #  [score_20,  score_21,  score_22]]
        wei = wei.masked_fill(self.tril[0:T, 0:T] == 0, float('-inf'))  # (B, T, T)

        # Softmax нормирует каждую строку: разрешенные веса неотрицательны и в
        # сумме дают 1, а позиции в будущем получают вес 0.
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        # Например, нормированные веса для трех токенов могут выглядеть так:
        # [[1.00,  0.00,  0.00],
        #  [0.25,  0.75,  0.00],
        #  [0.10,  0.60,  0.30]]

        # Для каждого токена смешиваем V-векторы доступных токенов с этими весами.
        # Например, для токенов [«Франция», «столица», «Париж»]:
        # out(«Париж») = 0.10*V(«Франция») + 0.60*V(«столица») + 0.30*V(«Париж»)
        out = wei @ v
        # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)

        # У отдельных координат head_size нет заранее заданной человеческой
        # интерпретации. Это обученные признаки: синтаксические и семантические
        # связи, связи сущностей с объектами и позиционная информация.
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return out


# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = MultiHeadAttention(4, n_embd // 4)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B, T, hidden_dim)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, hidden_dim)
        x = tok_emb + pos_emb  # (B, T, hidden_dim)
        x = self.sa_heads(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, vocab_size_current = logits.shape
            logits = logits.view(B*T, vocab_size_current)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]

            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, vocab_size)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
