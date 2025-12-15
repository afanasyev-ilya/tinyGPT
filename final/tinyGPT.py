import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters (keep yours)
batch_size = 64
block_size = 256
n_embd = 384
n_blocks = 6
n_heads = 6
dropout = 0.2


# ---------------- RoPE helpers ----------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def cos_sin(self, positions: torch.Tensor, device, dtype):
        # positions: (T,)
        freqs = torch.einsum("t,d->td", positions.to(device=device, dtype=self.inv_freq.dtype), self.inv_freq)  # (T, D/2)
        return freqs.cos().to(dtype=dtype), freqs.sin().to(dtype=dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x:   (B, H, T, D)
    cos: (T, D/2)
    sin: (T, D/2)
    RoPE with even/odd interleaving (LLaMA/NeoX-style).
    """
    # split even/odd
    x_even = x[..., 0::2]  # (B,H,T,D/2)
    x_odd  = x[..., 1::2]  # (B,H,T,D/2)

    cos = cos[None, None, :, :]  # (1,1,T,D/2)
    sin = sin[None, None, :, :]

    out_even = x_even * cos - x_odd * sin
    out_odd  = x_even * sin + x_odd * cos

    out = torch.empty_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out


# ---------------- Fused QKV + Flash (SDPA) + KV cache ----------------
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, n_embd: int, use_cache: bool):
        super().__init__()
        assert n_embd % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = n_embd // num_heads
        self.use_cache = use_cache

        # fused QKV
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=True)
        self.dropout = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(self.head_dim)

        # KV cache: (B, H, L, D)
        self.k_cache = None
        self.v_cache = None
        self.cache_index = 0      # how many tokens currently stored
        self.pos_base = 0         # absolute position of cache index 0 (for sliding window)

    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None
        self.cache_index = 0
        self.pos_base = 0

    def _maybe_init_cache(self, B, device, dtype):
        if self.k_cache is None:
            self.k_cache = torch.empty(B, self.num_heads, block_size, self.head_dim, device=device, dtype=dtype)
            self.v_cache = torch.empty(B, self.num_heads, block_size, self.head_dim, device=device, dtype=dtype)
            self.cache_index = 0
            self.pos_base = 0

    def _slide_if_needed(self, T: int):
        """
        Sliding window for KV-cache when exceeding block_size.
        This is now feasible/correct with RoPE because positions are absolute (pos_base grows).
        """
        if self.cache_index + T <= block_size:
            return

        overflow = self.cache_index + T - block_size
        # keep the last (block_size - T) old tokens, then append T new tokens
        keep = block_size - T
        if keep > 0:
            self.k_cache[:, :, :keep, :] = self.k_cache[:, :, overflow:self.cache_index, :].clone()
            self.v_cache[:, :, :keep, :] = self.v_cache[:, :, overflow:self.cache_index, :].clone()
        self.cache_index = max(0, keep)
        self.pos_base += overflow

    def forward(self, x: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        """
        x: (B, T, C)
        pos_offset: used only when use_cache=False (training/no-cache inference)
        """
        B, T, C = x.shape
        device = x.device
        dtype = x.dtype

        qkv = self.qkv(x)  # (B, T, 3C)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, T, D)

        if self.use_cache:
            self._maybe_init_cache(B, device, dtype)
            self._slide_if_needed(T)

            start = self.cache_index
            end = start + T

            # absolute positions for these new tokens
            positions = torch.arange(self.pos_base + start, self.pos_base + end, device=device)
        else:
            # no-cache path (e.g., training): positions come from pos_offset
            positions = torch.arange(pos_offset, pos_offset + T, device=device)

        cos, sin = self.rope.cos_sin(positions, device=device, dtype=dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.use_cache:
            # write K/V to cache
            self.k_cache[:, :, start:end, :] = k
            self.v_cache[:, :, start:end, :] = v
            self.cache_index = end

            k_full = self.k_cache[:, :, :self.cache_index, :]  # (B,H,L,D)
            v_full = self.v_cache[:, :, :self.cache_index, :]

            # Causality handling:
            # - decode step (T=1): no future keys exist in k_full -> no mask needed
            # - prefill (T>1): need causal mask w.r.t. the chunk
            if T == 1:
                attn_mask = None
                is_causal = False
            else:
                # If this is the very first prefill from empty cache, we can use is_causal=True (Flash-friendly).
                if start == 0:
                    attn_mask = None
                    is_causal = True
                else:
                    # General chunked case: build an explicit (T, L) additive mask (may disable flash).
                    L = k_full.size(2)
                    qpos = torch.arange(self.pos_base + start, self.pos_base + end, device=device).view(T, 1)  # (T,1)
                    kpos = torch.arange(self.pos_base, self.pos_base + L, device=device).view(1, L)            # (1,L)
                    allowed = (kpos <= qpos)  # (T,L)
                    attn_mask = torch.zeros((T, L), device=device, dtype=dtype)
                    attn_mask = attn_mask.masked_fill(~allowed, float("-inf"))
                    is_causal = False

            dropout_p = dropout if self.training else 0.0
            y = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
        else:
            # Training/no-cache: standard causal attention (Flash-friendly)
            dropout_p = dropout if self.training else 0.0
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True)

        # (B,H,T,D) -> (B,T,C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.dropout(y)
        return y


class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ff(x)


class Block(nn.Module):
    def __init__(self, n_embd, block_idx, use_cache, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd, use_cache)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, pos_offset=0):
        x = x + self.sa(self.ln1(x), pos_offset=pos_offset)
        x = x + self.ffwd(self.ln2(x))
        return x

    def reset_cache(self):
        self.sa.reset_cache()


class TinyGPTModel(nn.Module):
    def __init__(self, vocab_size, use_cache=False):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        self.blocks = nn.ModuleList(
            [Block(n_embd, block_idx, use_cache, n_head=n_heads) for block_idx in range(n_blocks)]
        )
        self.ln_fin = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.use_cache = use_cache

    def forward(self, idx, targets=None, pos_offset=0):
        B, T = idx.shape
        x = self.token_embedding_table(idx)  # (B,T,C)

        # propagate pos_offset (only relevant in no-cache; cache uses internal positions)
        cur_pos = pos_offset
        for blk in self.blocks:
            x = blk(x, pos_offset=cur_pos)
        x = self.ln_fin(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def reset_cache(self):
        for blk in self.blocks:
            blk.reset_cache()

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.reset_cache()

        if not self.use_cache:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -block_size:]
                logits, _ = self(idx_cond, pos_offset=0)
                probs = F.softmax(logits[:, -1, :], dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            return idx

        # cache path: prefill once, then decode token-by-token
        idx_cond = idx[:, -block_size:]
        logits, _ = self(idx_cond, pos_offset=0)

        for _ in range(max_new_tokens):
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            logits, _ = self(idx_next, pos_offset=0)  # pos handled internally in cache
        return idx

