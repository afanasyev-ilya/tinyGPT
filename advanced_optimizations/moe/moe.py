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

# MoE hyperparameters.
# В GigaChat 4.0 (для сравнения): hidden = 7168, n_experts = 256, top_k = 8,
# expert_hidden (moe_intermediate_size) = 2048, 1 shared expert,
# routed_scaling_factor = 2.5. Здесь всё маленькое, чтобы было видно руками.
n_experts = 4               # E: сколько routed-экспертов в слое
top_k = 2                   # k: через сколько экспертов проходит каждый токен
expert_hidden = 2 * n_embd  # I: внутренний размер одного эксперта
routed_scaling_factor = 2.5 # множитель для суммы весов routed-экспертов
bias_update_speed = 1e-3    # gamma: шаг подстройки bias для балансировки
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('../../input.txt', 'r', encoding='utf-8') as f:
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
        wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[0:T, 0:T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        out = wei @ v  # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return out


# ============================================================================
# Mixture of Experts
# ============================================================================
#
# В обычном трансформере после attention стоит FeedForward (см.
# ../mha/mha_ff_head_mix_plus_layernorm.py):
#     x = x + ffwd(ln2(x)),   ffwd: Linear(C, 4C) -> ReLU -> Linear(4C, C)
#
# В MoE вместо одного FeedForward стоит много маленьких FeedForward
# ("экспертов"), и каждый токен проходит только через top_k из них:
#     h = ln2(x)
#     MoE(h) = shared(h) + sum_{j=1..k} w_j * expert_{e_j}(h)
#     x = x + MoE(h)
# Роутер для каждого токена решает, какие k экспертов взять (номера e_j)
# и с какими весами (w_j). Так у модели много параметров (все эксперты),
# но на один токен считается только k экспертов.
#
# Сквозной toy-пример ниже: C = 4, E = 4 эксперта, top_k = 2,
# routed_scaling_factor = 2.5, два токена. Его же можно запустить:
# toy_router_example() в конце файла печатает все промежуточные тензоры.


class Expert(nn.Module):
    """Один эксперт = FeedForward со SwiGLU (как в DeepSeek / GigaChat).

    Отличие от FeedForward из tinyGPT: вместо ReLU стоит SwiGLU,
    silu(g) * u. Первая матрица выдаёт сразу и g, и u.
    """

    def __init__(self, n_embd, expert_hidden):
        super().__init__()
        # Обе матрицы обучаются обычным backprop'ом, как Linear в FeedForward.
        # W1: (C) -> (2*I), склеенные gate- и up-проекции.
        # W2: (I) -> (C), down-проекция.
        # GigaChat 4.0: W1 [4096, 7168], W2 [7168, 2048].
        self.w1 = nn.Linear(n_embd, 2 * expert_hidden, bias=False)
        self.w2 = nn.Linear(expert_hidden, n_embd, bias=False)

    def forward(self, h):
        # h: (N, C), только те токены, которые выбрали этого эксперта
        gu = self.w1(h)               # (N, 2*I)
        g, u = gu.chunk(2, dim=-1)    # (N, I), (N, I)
        a = F.silu(g) * u             # (N, I); silu(g) = g * sigmoid(g)
        return self.w2(a)             # (N, C)


class Router(nn.Module):
    """DeepSeek-V3-style роутер: sigmoid scores + correction bias + top-k.

    Что обучается и как:
      * W_gate (self.gate.weight) — ОБУЧАЕМЫЙ параметр, градиенты как у
        любого Linear. Он решает, какому эксперту токен "подходит".
      * e_score_correction_bias — НЕ параметр (register_buffer, градиентов
        нет). Во время обучения его двигают простым правилом после каждого
        шага (update_bias): перегруженному эксперту bias уменьшают,
        недогруженному увеличивают. Это балансировка нагрузки без aux-loss.
        На inference bias заморожен: он хранит не текущую загрузку, а
        выученную на обучении поправку "этого эксперта модель склонна
        выбирать слишком часто / слишком редко". Поэтому он влияет только на
        ВЫБОР экспертов, но не на их веса.
    """

    def __init__(self, n_embd, n_experts, top_k, routed_scaling_factor):
        super().__init__()
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        # W_gate: [E, C]. Каждая строка — "образ" эксперта в пространстве h.
        # GigaChat 4.0: [256, 7168] bf16.
        self.gate = nn.Linear(n_embd, n_experts, bias=False)
        # bias: [E], стартует с нуля, градиентов нет.
        self.register_buffer('e_score_correction_bias', torch.zeros(n_experts))
        # Сколько токенов выбрало каждого эксперта на последнем forward
        # (для update_bias).
        self.register_buffer('last_expert_load', torch.zeros(n_experts))

    def forward(self, h):
        # h: (N, C), N = B*T токенов. В decode N = число генерируемых токенов.
        #
        # toy: два токена
        #   h = [[1.0,  0.0, -1.0,  2.0],     # токен 0
        #        [2.4, -3.0,  1.0, -1.8]]     # токен 1
        #   W_gate = [[1.0, 0, 0, 0.5],       # эксперт 0
        #             [0,   1, 1, 0  ],       # эксперт 1
        #             [0.5, 0, 0, 0  ],       # эксперт 2
        #             [0,   0,-1, 0  ]]       # эксперт 3

        # Шаг 1. Gate GEMM: оценка "насколько токен похож на эксперта".
        # logit[t, e] = <h[t], W_gate[e]>; скалярное произведение, как q @ k
        # в attention, только с E фиксированными обученными векторами.
        #   токен 0: e0 = 1*1 + 2*0.5 = 2.0, e1 = 0 + (-1) = -1.0,
        #            e2 = 0.5*1 = 0.5,       e3 = -(-1) = 1.0
        #   logits = [[ 2.0, -1.0, 0.5,  1.0],
        #             [ 1.5, -2.0, 1.2, -1.0]]
        logits = self.gate(h)                      # (N, E)

        # Шаг 2. Sigmoid: каждый эксперт оценивается сам по себе, от 0 до 1.
        #   sigmoid(z) = 1 / (1 + exp(-z))
        #   z =  2.0: exp(-2.0) = 0.135 -> 1 / 1.135 = 0.881
        #   z = -1.0: exp( 1.0) = 2.718 -> 1 / 3.718 = 0.269
        #   z =  0.5: exp(-0.5) = 0.607 -> 1 / 1.607 = 0.622
        #   z =  1.0: exp(-1.0) = 0.368 -> 1 / 1.368 = 0.731
        #   s = [[0.881, 0.269, 0.622, 0.731],
        #        [0.818, 0.119, 0.769, 0.269]]
        # Почему не softmax (как в Mixtral): softmax(2.0, -1.0, 0.5, 1.0) =
        # [0.609, 0.030, 0.136, 0.224]. Эксперты делят между собой единицу,
        # и рост одного логита уменьшает вероятность всех остальных. Sigmoid
        # так не связывает экспертов. В GPU-ядрах TRT-LLM то же самое
        # считают как 0.5 * tanh(0.5 * z) + 0.5.
        s = torch.sigmoid(logits.float())          # (N, E)

        # Шаг 3. Bias: поправка только для выбора.
        #   пусть bias = [-0.3, 0.0, 0.2, 0.0]
        #   s + bias = [[0.581, 0.269, 0.822, 0.731],
        #               [0.518, 0.119, 0.969, 0.269]]
        biased = s + self.e_score_correction_bias  # (N, E)

        # Шаг 4. Top-k по s + bias.
        #   токен 0: idx = [2, 3]. У эксперта 0 самый большой s (0.881),
        #            но bias -0.3 его вытеснил: он перегружен.
        #   токен 1: idx = [2, 0]
        # topk не дифференцируем: градиент через "выбор" не течёт.
        _, idx = torch.topk(biased, k=self.top_k, dim=-1)  # (N, k)

        # Шаг 5. Веса — из s БЕЗ bias, нормированные по выбранным.
        #   токен 0: s[2] = 0.622, s[3] = 0.731, сумма = 1.354
        #            w = [0.622, 0.731] / 1.354 * 2.5 = [1.150, 1.350]
        #   токен 1: s[2] = 0.769, s[0] = 0.818, сумма = 1.586
        #            w = [0.769, 0.818] / 1.586 * 2.5 = [1.211, 1.289]
        # Деление на сумму (norm_topk_prob) делает так, что веса до умножения
        # всегда дают в сумме 1. routed_scaling_factor масштабирует вклад
        # routed-экспертов относительно shared.
        # Через gather(s) градиент течёт в W_gate: если выбранный эксперт
        # помог уменьшить loss, его logit для таких токенов растёт.
        w = torch.gather(s, -1, idx)               # (N, k)
        w = w / (w.sum(-1, keepdim=True) + 1e-20) * self.routed_scaling_factor

        # Нагрузка для балансировки: сколько раз выбрали каждого эксперта.
        #   toy: idx = [[2, 3], [2, 0]] -> load = [1, 0, 2, 1]
        with torch.no_grad():
            self.last_expert_load = torch.bincount(
                idx.flatten(), minlength=self.last_expert_load.numel()
            ).float()
        return idx, w.type_as(h)

    @torch.no_grad()
    def update_bias(self, gamma):
        """Балансировка без aux-loss (DeepSeek-V3): вызывается после шага.

        toy: load = [100, 20, 90, 46], среднее = 64
             перегружены e0, e2 -> bias -= gamma
             недогружены e1, e3 -> bias += gamma
        Через много шагов bias сойдётся к значениям, при которых эксперты
        выбираются примерно поровну. Дальше его просто сохраняют вместе с
        весами (в чекпоинте GigaChat это e_score_correction_bias).
        """
        load = self.last_expert_load
        self.e_score_correction_bias += gamma * torch.sign(load.mean() - load)


class MoE(nn.Module):
    """Замена FeedForward: shared expert + top_k из n_experts routed-экспертов."""

    def __init__(self, n_embd, n_experts, top_k, expert_hidden, routed_scaling_factor):
        super().__init__()
        self.router = Router(n_embd, n_experts, top_k, routed_scaling_factor)
        self.experts = nn.ModuleList(
            [Expert(n_embd, expert_hidden) for _ in range(n_experts)]
        )
        # Shared expert: через него проходят ВСЕ токены, без роутера.
        # Он забирает общие для всех токенов признаки, чтобы routed-эксперты
        # могли специализироваться. В GigaChat 4.0 он один, размер 2048.
        self.shared_expert = Expert(n_embd, expert_hidden)

    def forward(self, h):
        B, T, C = h.shape
        h = h.view(B * T, C)       # (N, C): дальше токены независимы

        # Роутинг: для каждого токена k номеров экспертов и k весов.
        #   toy: idx = [[2, 3],          w = [[1.150, 1.350],
        #               [2, 0]]               [1.211, 1.289]]
        idx, w = self.router(h)    # (N, k), (N, k)

        # Shared expert считается для всех токенов.
        out = self.shared_expert(h)   # (N, C)

        # Dispatch: раскидываем токены по экспертам.
        #   toy (два токена):
        #     expert 0: токен 1 (вес 1.289)
        #     expert 1: никто — его веса в этот раз даже не читаются
        #     expert 2: токены 0 (1.150) и 1 (1.211) — мини-батч из 2 строк
        #     expert 3: токен 0 (1.350)
        # Каждый эксперт считает только свои строки, результат умножается на
        # вес и прибавляется в строку токена (combine):
        #   out[0] += 1.150 * expert_2(h[0]) + 1.350 * expert_3(h[0])
        #   out[1] += 1.211 * expert_2(h[1]) + 1.289 * expert_0(h[1])
        # В TRT-LLM / MegaMoE dispatch и combine идут между GPU (эксперты
        # разложены по рангам), а здесь — простой цикл.
        for e, expert in enumerate(self.experts):
            token_ids, slot = (idx == e).nonzero(as_tuple=True)  # (n_e,), (n_e,)
            if token_ids.numel() == 0:
                continue
            y = expert(h[token_ids])                      # (n_e, C)
            y = y * w[token_ids, slot].unsqueeze(-1)      # (n_e, C) * (n_e, 1)
            out = out.index_add(0, token_ids, y)          # combine

        return out.view(B, T, C)


class Block(nn.Module):
    """Transformer block: attention и MoE, оба с pre-norm и residual."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        # БЫЛО в mha_base.py: self.sa_heads = MultiHeadAttention(4, n_embd // 4)
        self.sa = MultiHeadAttention(n_head, head_size)
        # СТАЛО: после attention стоит MoE (вместо FeedForward).
        self.moe = MoE(n_embd, n_experts, top_k, expert_hidden, routed_scaling_factor)
        # В GigaChat тут RMSNorm, для tinyGPT разницы нет.
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        # h = norm2(x) — тот самый вектор, который получает роутер.
        h = self.ln2(x)        # (B, T, C)
        x = x + self.moe(h)    # (B, T, C)
        return x


# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # БЫЛО в mha_base.py: self.sa_heads = MultiHeadAttention(4, n_embd // 4)
        # СТАЛО: один Block = attention + MoE.
        self.block = Block(n_embd, n_head=4)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B, T, hidden_dim)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, hidden_dim)
        x = tok_emb + pos_emb  # (B, T, hidden_dim)
        x = self.block(x)
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


@torch.no_grad()
def toy_router_example():
    """Печатает сквозной toy-пример из комментариев Router.forward."""
    router = Router(n_embd=4, n_experts=4, top_k=2, routed_scaling_factor=2.5)
    router.gate.weight.copy_(torch.tensor([[1.0, 0, 0, 0.5],
                                           [0, 1, 1, 0],
                                           [0.5, 0, 0, 0],
                                           [0, 0, -1, 0]]))
    router.e_score_correction_bias.copy_(torch.tensor([-0.3, 0.0, 0.2, 0.0]))
    h = torch.tensor([[1.0, 0.0, -1.0, 2.0],
                      [2.4, -3.0, 1.0, -1.8]])
    logits = router.gate(h)
    s = torch.sigmoid(logits)
    idx, w = router(h)
    print("toy router example")
    print("  logits   =", logits.tolist())
    print("  sigmoid  =", [[round(v, 3) for v in r] for r in s.tolist()])
    print("  s + bias =", [[round(v, 3) for v in r] for r in (s + router.e_score_correction_bias).tolist()])
    print("  idx      =", idx.tolist())
    print("  w        =", [[round(v, 3) for v in r] for r in w.tolist()])
    print("  load     =", router.last_expert_load.tolist())


toy_router_example()

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
# В оптимизатор попадают W_gate, W1/W2 всех экспертов, attention и т.д.
# e_score_correction_bias — buffer, в model.parameters() его нет.
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
routers = [mod for mod in model.modules() if isinstance(mod, Router)]

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        load = routers[0].last_expert_load.tolist()
        bias = [round(b, 3) for b in routers[0].e_score_correction_bias.tolist()]
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, "
              f"expert load {load}, bias {bias}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Балансировка: после шага подвинуть bias по нагрузке этого батча.
    for router in routers:
        router.update_bias(bias_update_speed)

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
