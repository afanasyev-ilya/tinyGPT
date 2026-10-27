import torch
import torch.nn as nn
from torch.nn import functional as F

# Минимальные размеры, нужные только для иллюстрации DSA-head.
n_embd = 32
block_size = 8
H_index = 4
index_dim = 8
K = 4


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

        # В scores_by_head выше есть отрицательные значения -1 и -2.
        # ReLU(x) = max(0, x) превращает именно их в нули, поэтому отрицательное
        # совпадение одной головы само не отменяет полезный сигнал другой.
        # index_weights затем смешивает головы для каждой query-позиции:
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
