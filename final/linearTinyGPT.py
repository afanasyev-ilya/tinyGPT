import torch
import torch.nn as nn
from torch.nn import functional as F

from tinyGPT import n_embd, n_heads, n_blocks, block_size, dropout, FeedFoward


class LinearAttentionHead(nn.Module):
    def __init__(self, head_size, use_cache=True, eps=1e-6):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.use_cache = use_cache
        self.eps = eps
        self.cache_S = None
        self.cache_Z = None

    def _phi(self, x):
        # Positive feature map for linear attention
        return F.elu(x) + 1.0

    def forward_no_cache(self, x):
        # Training path: parallel causal linear attention via prefix sums
        k = self._phi(self.key(x))
        q = self._phi(self.query(x))
        v = self.value(x)

        # S_t = sum_{i<=t} k_i v_i^T
        kv = torch.einsum('btd,btm->btdm', k, v)
        S = torch.cumsum(kv, dim=1)

        # Z_t = sum_{i<=t} k_i
        Z = torch.cumsum(k, dim=1)

        # out_t = (q_t^T S_t) / (q_t^T Z_t)
        numerator = torch.einsum('btd,btdm->btm', q, S)
        denominator = torch.einsum('btd,btd->bt', q, Z).unsqueeze(-1)
        out = numerator / (denominator + self.eps)
        out = self.dropout(out)
        return out

    def forward_with_cache(self, x):
        # Decode path: recurrent update (RNN-like), typically with T=1
        k = self._phi(self.key(x))
        q = self._phi(self.query(x))
        v = self.value(x)

        B, T, D = k.shape
        M = v.size(-1)
        device = x.device
        dtype = x.dtype

        if self.cache_S is None:
            self.cache_S = torch.zeros(B, D, M, device=device, dtype=dtype)
            self.cache_Z = torch.zeros(B, D, device=device, dtype=dtype)

        outputs = []
        for t in range(T):
            k_t = k[:, t, :]
            q_t = q[:, t, :]
            v_t = v[:, t, :]

            self.cache_S = self.cache_S + torch.einsum('bd,bm->bdm', k_t, v_t)
            self.cache_Z = self.cache_Z + k_t

            numerator = torch.einsum('bd,bdm->bm', q_t, self.cache_S)
            denominator = torch.einsum('bd,bd->b', q_t, self.cache_Z).unsqueeze(-1)
            out_t = numerator / (denominator + self.eps)
            outputs.append(out_t.unsqueeze(1))

        out = torch.cat(outputs, dim=1)
        out = self.dropout(out)
        return out

    def forward(self, x):
        if self.use_cache:
            return self.forward_with_cache(x)
        return self.forward_no_cache(x)

    def reset_cache(self):
        self.cache_S = None
        self.cache_Z = None


class MultiHeadLinearAttention(nn.Module):
    def __init__(self, num_heads, head_size, use_cache):
        super().__init__()
        self.heads = nn.ModuleList([LinearAttentionHead(head_size, use_cache=use_cache) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

    def reset_cache(self):
        for head in self.heads:
            head.reset_cache()


class LinearAttentionBlock(nn.Module):
    def __init__(self, n_embd, use_cache, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadLinearAttention(n_head, head_size, use_cache)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

    def reset_cache(self):
        self.sa.reset_cache()


class TinyGPTLinearAttentionModel(nn.Module):
    """
    TinyGPT variant with causal linear attention.
    API is intentionally kept similar to TinyGPTModel for educational side-by-side use.
    """
    def __init__(self, vocab_size, use_cache=False):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[LinearAttentionBlock(n_embd, use_cache, n_head=n_heads) for _ in range(n_blocks)]
        )
        self.ln_fin = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_cache = use_cache

    def forward(self, idx, targets=None, pos_offset=0):
        B, T = idx.shape
        device = idx.device

        tok_emb = self.token_embedding_table(idx)
        pos_ids = torch.arange(pos_offset, pos_offset + T, device=device)
        pos_ids = torch.clamp(pos_ids, max=block_size - 1)
        pos_emb = self.position_embedding_table(pos_ids)
        x = tok_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_fin(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def reset_cache(self):
        for block in self.blocks:
            block.reset_cache()

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.reset_cache()

        if not getattr(self, 'use_cache', False):
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -block_size:]
                logits, _ = self(idx_cond, pos_offset=0)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            return idx

        idx_cond = idx[:, -block_size:]
        logits, _ = self(idx_cond, pos_offset=0)

        for _ in range(max_new_tokens):
            logits_last = logits[:, -1, :]
            probs = F.softmax(logits_last, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            pos = min(idx.size(1) - 1, block_size - 1)
            logits, _ = self(idx_next, pos_offset=pos)

        self.reset_cache()
        return idx
