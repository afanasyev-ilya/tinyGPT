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
H_index = 4
index_dim = 8
K = 4
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

        # DSA-indexer устроен как Multi-Query Attention: у index_q есть
        # H_index голов, а index_k — один общий вектор для исторического токена.
        # На decode новый query имеет T_query=1, тогда как index_k-cache содержит
        # всю историю. Отдельный index_k для каждой головы увеличил бы этот
        # длинный кеш в H_index раз; основной attention KV-cache хранится отдельно.
        self.index_key = nn.Linear(n_embd, index_dim, bias=False)
        self.index_query = nn.Linear(n_embd, H_index * index_dim, bias=False)
        self.index_weight = nn.Linear(n_embd, H_index, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    # Ниже показан только inference-путь DSA. Для обучения сначала получают
    # dense-модель, затем отдельно учат indexer предсказывать распределение
    # важных позиций dense attention. После warm-up включают Top-K и продолжают
    # адаптировать модель и indexer с отдельными loss-функциями.
    def forward(self, x):
        # input of size (batch, time-step, hidden_dim)
        # output of size (batch, time-step, head_size)
        B, T, hidden_dim = x.shape

        q = self.query(x)  # (B, T, head_size)
        k = self.key(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # index_q — «что текущая query-позиция ищет?» Каждая голова описывает
        # свой способ поиска. index_k — «какую информацию предлагает и по каким
        # признакам можно найти этот исторический токен?» Все головы index_q
        # сравниваются с одним общим index_k каждого токена.
        index_k = self.index_key(x)  # (B, T, index_dim)
        index_q = self.index_query(x)  # (B, T, H_index * index_dim)
        index_q = index_q.view(B, T, H_index, index_dim)  # (B, T, H_index, index_dim)

        # Для каждой query-позиции это H_index скаляров, определяющих, как
        # смешивать результаты голов. Это не attention probabilities: здесь нет
        # softmax, поэтому веса не обязаны давать в сумме 1 и могут быть знаковыми.
        index_weights = self.index_weight(x)  # (B, T, H_index)

        # Считаем indexer-скоры каждого запроса со всеми ключами.
        # COMPLEXITY: для каждой из T query-позиций и T key-позиций выполняется
        # dot product длины index_dim в каждой из H_index голов:
        # O(T² * H_index * index_dim), если не учитывать размер batch.
        # Мини-пример: один элемент batch, рассматриваем query-позицию 1 при T=3.
        # query_heads = [[1, 0],    # head 0
        #                [0, 1]]    # head 1
        # # (H_index=2, index_dim=2)
        # history_keys = [[ 2,  1], # key position 0
        #                 [-1,  3], # key position 1
        #                 [ 1, -2]] # key position 2
        # # (T_key=3, index_dim=2)
        # scores_by_head = query_heads @ history_keys.T
        # # (H_index, index_dim) @ (index_dim, T_key)
        # # -> (H_index, T_key)
        # scores_by_head = [[2, -1,  1],  # head 0 против всех history keys
        #                   [1,  3, -2]]  # head 1 против всех history keys

        # einsum здесь не просто переставляет оси, а делает dot product:
        # 'bqhd,bld->bqhl'
        #  b — batch, q — query-позиция, h — indexer head,
        #  l — позиция исторического key, d — координата index_dim.
        # Буква d есть в обоих входах, но отсутствует в результате, поэтому
        # произведения суммируются по d:
        # per_head_scores[b, q, h, l] = sum_d(index_q[b, q, h, d]
        #                                      * index_k[b, l, d])
        per_head_scores = torch.einsum(
            'bqhd,bld->bqhl',
            index_q,
            index_k,
        )  # (B, T_query, H_index, T_key)

        # ReLU(x) = max(0, x): отрицательное совпадение одной головы становится
        # нулем и само не отменяет полезный сигнал другой. index_weights затем
        # смешивает головы отдельно для каждой query-позиции. В примере выше:
        # ReLU(scores_by_head) = [[2, 0, 1], [1, 3, 0]]
        # weights_for_query = [0.5, 2.0]
        # scores_for_query = 0.5*[2, 0, 1] + 2.0*[1, 3, 0]
        #                  = [3.0, 6.0, 0.5]
        index_scores = (
            F.relu(per_head_scores) * index_weights.unsqueeze(-1)
        ).sum(dim=2)  # (B, T_query, T_key)

        # Оставляем прошлое и текущий токен, затем выбираем Top-K позиций.
        index_scores = index_scores.masked_fill(
            self.tril[0:T, 0:T] == 0,
            float('-inf'),
        )  # (B, T_query, T_key)
        top_indices = torch.topk(
            index_scores,
            k=min(K, T),
            dim=-1,
        ).indices  # (B, T_query, K)

        # Продолжение примера для query-позиции 1 и условного K=2:
        # scores_for_query = [3.0, 6.0, 0.5]
        # allowed_by_causal_mask = [True, True, False]
        # masked_scores     = [3.0, 6.0, -inf]
        # top_indices       = [1, 0]
        # Позиция 2 отброшена как будущая, а из разрешенных позиций 0 и 1
        # torch.topk возвращает индексы в порядке убывания score.

        # Top-K выбирается отдельно для каждой query-строки: у разных queries
        # могут быть разные наборы key/value-позиций, а не один набор на весь attention.
        #                    key positions
        # query 0            ✓
        # query 1            · ✓
        # query 2            ✓ · ✓
        # query 3            · ✓ · ✓

        # Собираем настоящие K и V только в выбранных позициях.
        gather_indices = top_indices.unsqueeze(-1).expand(
            -1, -1, -1, k.size(-1)
        )  # (B, T_query, K, head_size)
        selected_k = torch.gather(
            k.unsqueeze(1).expand(-1, T, -1, -1),
            dim=2,
            index=gather_indices,
        )  # (B, T_query, K, head_size)
        selected_v = torch.gather(
            v.unsqueeze(1).expand(-1, T, -1, -1),
            dim=2,
            index=gather_indices,
        )  # (B, T_query, K, head_size)

        # Было: dense attention считал скоры и агрегацию по всем T токенам.
        # wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5  # (B, T, T)
        # wei = wei.masked_fill(self.tril[0:T, 0:T] == 0, float('-inf'))
        # wei = F.softmax(wei, dim=-1)  # (B, T, T)
        # out = wei @ v  # (B, T, head_size)

        # Считаем sparse attention только по выбранным K токенам.
        # COMPLEXITY: каждая из T query-позиций взаимодействует с K выбранными
        # key/value-позициями: O(T * K * head_dim), где K=2048 в GigaChat 4.0.
        # Квадратичность основного attention исчезает, но остается в indexer
        # einsum выше: O(T² * H_index * index_dim).
        wei = (
            q.unsqueeze(2) * selected_k
        ).sum(dim=-1) * self.head_size ** -0.5  # (B, T_query, K)
        selected_is_future = top_indices > torch.arange(T, device=x.device).view(1, T, 1)
        wei = wei.masked_fill(selected_is_future, float('-inf'))  # (B, T_query, K)
        wei = F.softmax(wei, dim=-1)  # (B, T_query, K)
        out = (
            wei.unsqueeze(-1) * selected_v
        ).sum(dim=2)  # (B, T_query, head_size)
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
