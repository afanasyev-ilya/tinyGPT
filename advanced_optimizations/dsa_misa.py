import torch
import torch.nn as nn
from torch.nn import functional as F

# Минимальные размеры, нужные только для иллюстрации MISA-head.
n_embd = 32
block_size = 8

# DSA-indexer имеет H_index разных query-голов и один общий index_k.
H_index = 4
index_dim = 8
K = 4

# MISA выбирает отдельно для каждой query только h голов из H_index.
# Тяжелый token-level scan индексатора становится в H_index / h раз меньше:
# вместо 4 голов по всем токенам проходят только 2.
h = 2

# Это НЕ общий block_size модели выше. misa_block_size говорит, сколько
# соседних index_k временно усредняется в один ключ для дешевого router.
# При T=8 получаются ceil(T / misa_block_size) = 4 блока:
#   tokens [0,1] -> block 0
#   tokens [2,3] -> block 1
#   tokens [4,5] -> block 2
#   tokens [6,7] -> block 3
# Блоки нужны только для выбора голов. Выбранные h голов затем снова смотрят
# на все T отдельных токенов, а не на эти четыре блока.
misa_block_size = 2


def causal_mean_pool(index_k):
    """Строим маленькую block-level историю для MISA router.

    index_k:          (B, T, index_dim)
    pooled_index_k:   (B, T_query, T_blocks, index_dim)
    valid_blocks:     (T_query, T_blocks)

    Реальное ядро не материализует такой большой тензор. Здесь мы явно
    строим свои causal-блоки для каждой query, чтобы было видно, откуда они
    берутся и чтобы в среднее не попадали будущие токены.
    """
    B, T, D = index_k.shape
    T_blocks = (T + misa_block_size - 1) // misa_block_size

    pooled_index_k = index_k.new_zeros(B, T, T_blocks, D)
    valid_blocks = torch.zeros(T, T_blocks, dtype=torch.bool, device=index_k.device)

    for query_pos in range(T):
        # Query в позиции query_pos видит историю [0, query_pos].
        causal_end = query_pos + 1

        for block_id in range(T_blocks):
            block_start = block_id * misa_block_size
            block_end = min(block_start + misa_block_size, causal_end)

            if block_start < block_end:
                pooled_index_k[:, query_pos, block_id] = index_k[
                    :, block_start:block_end
                ].mean(dim=1)
                valid_blocks[query_pos, block_id] = True

    return pooled_index_k, valid_blocks


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Это тот же обученный DSA-indexer, что и в gpt_dsa.py. MISA не
        # добавляет новых обучаемых projections: она только решает, какие
        # H_index query-головы запускать по полной token-level истории.
        self.index_key = nn.Linear(n_embd, index_dim, bias=False)
        self.index_query = nn.Linear(n_embd, H_index * index_dim, bias=False)
        self.index_weight = nn.Linear(n_embd, H_index, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    # Это максимально простой one-stage MISA inference-путь. В настоящей
    # модели MISA подключается к уже обученному DSA-indexer без переобучения.
    def forward(self, x):
        # input of size (batch, time-step, hidden_dim)
        # output of size (batch, time-step, head_size)
        B, T, hidden_dim = x.shape

        # Настоящие Q/K/V основного attention. MISA их не меняет.
        q = self.query(x)  # (B, T, head_size)
        k = self.key(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # Тот же learned DSA-indexer, что и раньше.
        index_k = self.index_key(x)  # (B, T, index_dim)
        index_q = self.index_query(x)  # (B, T, H_index * index_dim)
        index_q = index_q.view(B, T, H_index, index_dim)  # (B, T, H_index, index_dim)
        index_weights = self.index_weight(x)  # (B, T, H_index)

        # -----------------------------------------------------------------
        # MISA STEP 1. Сжимаем token-history только для дешевого router.
        # -----------------------------------------------------------------
        # Численный пример для B=1, T=5, index_dim=1, misa_block_size=2:
        #   index_k = [10, 20, 30, 40, 50]
        #
        # Для каждой query берем только доступную ей causal history и
        # усредняем соседние ключи по два:
        #   query 0: [10]                 -> [10]
        #   query 1: [10, 20]             -> [15]
        #   query 2: [10, 20, 30]         -> [15, 30]
        #   query 3: [10, 20, 30, 40]     -> [15, 35]
        #   query 4: [10, 20, 30, 40, 50] -> [15, 35, 50]
        # Последний видимый блок может быть неполным. Невидимые будущие
        # блоки остаются нулевыми и затем отсекаются через valid_blocks.
        pooled_index_k, valid_blocks = causal_mean_pool(index_k)
        # pooled_index_k: (B, T_query, T_blocks, index_dim)
        # T_blocks = ceil(T / misa_block_size), то есть примерно
        # T / misa_block_size вместо T исходных token-level ключей.
        #
        # Функция возвращает два отдельных прямоугольных тензора:
        #
        # pooled_index_k[0, :, :, 0]:
        #   query 0: [10,  0,  0]
        #   query 1: [15,  0,  0]
        #   query 2: [15, 30,  0]
        #   query 3: [15, 35,  0]
        #   query 4: [15, 35, 50]
        #
        # valid_blocks:
        #   query 0: [True,  False, False]
        #   query 1: [True,  False, False]
        #   query 2: [True,  True,  False]
        #   query 3: [True,  True,  False]
        #   query 4: [True,  True,  True ]
        #
        # Значит [15, 35, 50] — строка pooled_index_k для query 4.
        # Нули в остальных строках — padding; valid_blocks отличает их от
        # настоящих pooled-ключей.

        # Все H_index голов сравниваем с короткой block-level историей.
        # 'bqhd,bqmd->bqhm':
        #  q — query position
        #  h — indexer head
        #  m — pooled history block
        #  d — index_dim, по которому выполняется dot product
        #
        # Продолжим тот же пример только для последней query. Пусть ее четыре
        # scalar query-головы и веса равны:
        #   index_q[query 4]       = [1, -1, 2, 0.5]
        #   index_weights[query 4] = [1,  1, 0.75, 1.5]
        #   pooled_index_k[query 4] = [15, 35, 50]
        routing_affinity = torch.einsum(
            'bqhd,bqmd->bqhm',
            index_q,
            pooled_index_k,
        )  # (B, T_query, H_index, T_blocks)
        # Для query 4 это дает по одной строке на голову:
        #   head 0:  1   * [15, 35, 50] = [ 15,  35,  50]
        #   head 1: -1   * [15, 35, 50] = [-15, -35, -50]
        #   head 2:  2   * [15, 35, 50] = [ 30,  70, 100]
        #   head 3:  0.5 * [15, 35, 50] = [7.5, 17.5, 25]

        routing_contribution = (
            F.relu(routing_affinity) * index_weights.unsqueeze(-1)
        )  # (B, T_query, H_index, T_blocks)
        # После ReLU и умножения каждой строки на вес ее головы:
        #   head 0: [   15,    35,   50]
        #   head 1: [    0,     0,    0]
        #   head 2: [ 22.5,  52.5,   75]
        #   head 3: [11.25, 26.25, 37.5]

        # Для ранних query часть будущих блоков еще не существует.
        routing_contribution = routing_contribution.masked_fill(
            ~valid_blocks.view(1, T, 1, -1),
            0,
        )

        # Одна оценка полезности на каждую indexer-head и каждую query.
        # Деление на число блоков превращает сумму в среднее; для Top-K голов
        # оно не меняет порядок, но лучше показывает смысл формулы MISA.
        valid_block_count = valid_blocks.sum(dim=-1).clamp_min(1).view(1, T, 1)
        head_scores = (
            routing_contribution.abs().sum(dim=-1) / valid_block_count
        )  # (B, T_query, H_index)
        # У query 4 валидны все три блока, поэтому средние scores равны:
        #   head_scores[query 4] = [33.33, 0, 50, 25]

        # -----------------------------------------------------------------
        # MISA STEP 2. Для КАЖДОЙ query отдельно выбираем h из H_index голов.
        # -----------------------------------------------------------------
        active_head_indices = torch.topk(
            head_scores,
            k=min(h, H_index),
            dim=-1,
            largest=True,
            sorted=False,
        ).indices  # (B, T_query, h)

        # При h=2 для query 4 выбирается множество голов {2, 0}. Порядок
        # внутри пары не гарантирован, потому что sorted=False.

        # Например, разные queries могут выбрать разные головы:
        # query 0 -> heads [1, 3]
        # query 1 -> heads [0, 3]
        # query 2 -> heads [0, 2]
        # Это не один глобальный набор h голов на весь запрос.

        # В нашем scalar-примере, если topk вернул [2, 0], gather оставит:
        #   active_index_q[query 4]       = [2, 1]
        #   active_index_weights[query 4] = [0.75, 1]
        #
        # Теперь отдельный 2D-пример показывает механику expand. Пусть
        # B=1, T_query=2, H_index=4, index_dim=2, h=2.
        # У каждой query есть четыре двухмерных index_q-вектора:
        #   query 0: [[  0,   1], [ 10,  11], [ 20,  21], [ 30,  31]]
        #   query 1: [[100, 101], [110, 111], [120, 121], [130, 131]]
        # Пусть router выбрал:
        #   active_head_indices = [[[3, 1], [0, 2]]]  # (1, 2, 2)
        # Тогда хотим получить:
        #   query 0: [[ 30,  31], [ 10,  11]]
        #   query 1: [[100, 101], [120, 121]]
        # То есть размер голов схлопывается H_index=4 -> h=2, а целый
        # index_dim-вектор каждой выбранной головы сохраняется.
        #
        # torch.gather требует, чтобы index имел столько же измерений,
        # сколько index_q. Поэтому:
        #   unsqueeze(-1): (B, T_query, h) -> (B, T_query, h, 1)
        # добавляет ось координат вектора, а
        #   expand(..., index_dim):        -> (B, T_query, h, index_dim)
        # повторяет номер головы для каждой координаты d. Для примера выше:
        #   query 0: [[3, 3], [1, 1]]
        #   query 1: [[0, 0], [2, 2]]
        # expand здесь создает логическое представление, а не копию данных.
        active_index_q = torch.gather(
            index_q,
            dim=2,
            index=active_head_indices.unsqueeze(-1).expand(
                -1, -1, -1, index_dim
            ),
        )  # (B, T_query, h, index_dim)

        # У index_weights нет последнего index_dim: на голову хранится одно
        # число. Поэтому active_head_indices уже имеет нужную форму, и
        # unsqueeze/expand не нужны. В примере получим:
        #   query 0: [weight_3, weight_1]
        #   query 1: [weight_0, weight_2]
        active_index_weights = torch.gather(
            index_weights,
            dim=2,
            index=active_head_indices,
        )  # (B, T_query, h)

        # -----------------------------------------------------------------
        # MISA STEP 3. Только выбранные h голов сканируют ВСЕ token-level K.
        # -----------------------------------------------------------------
        # Обычный DSA:
        #   (B, T_query, H_index, index_dim) x (B, T_key, index_dim)
        #   -> (B, T_query, H_index, T_key)
        #
        # MISA:
        #   (B, T_query, h, index_dim) x (B, T_key, index_dim)
        #   -> (B, T_query, h, T_key)
        #
        # Именно эта тяжелая часть сокращается примерно в H_index / h раз.
        active_per_head_scores = torch.einsum(
            'bqhd,bld->bqhl',
            active_index_q,
            index_k,
        )  # (B, T_query, h, T_key)
        # Продолжение scalar-примера для query 4 и голов [2, 0]:
        #   head 2: 2 * [10, 20, 30, 40, 50] = [20, 40, 60, 80, 100]
        #   head 0: 1 * [10, 20, 30, 40, 50] = [10, 20, 30, 40,  50]

        index_scores = (
            F.relu(active_per_head_scores)
            * active_index_weights.unsqueeze(-1)
        ).sum(dim=2)  # (B, T_query, T_key)
        # Применяем веса выбранных голов и складываем по h:
        #   0.75 * [20, 40, 60, 80, 100]
        #      1 * [10, 20, 30, 40,  50]
        #   --------------------------------
        #          [25, 50, 75, 100, 125] = index_scores[query 4]

        # Полный indexer compute уменьшается чуть меньше чем H_index / h,
        # потому что к сокращенному token scan добавился дешевый router:
        #
        # DSA:  O(T² * H_index * index_dim)
        # MISA: O(T² / misa_block_size * H_index * index_dim)  <- router
        #     + O(T² * h * index_dim)                           <- token scan

        # -----------------------------------------------------------------
        # Дальше обычный DSA: causal mask и Top-K отдельных токенов.
        # -----------------------------------------------------------------
        index_scores = index_scores.masked_fill(
            self.tril[0:T, 0:T] == 0,
            float('-inf'),
        )  # (B, T_query, T_key)

        # One-stage означает, что выбранные h голов сразу определяют
        # финальные K токенов. Повторного rerank всеми H_index головами нет.
        top_indices = torch.topk(
            index_scores,
            k=min(K, T),
            dim=-1,
        ).indices  # (B, T_query, K)
        # При K=4 query 4 выберет token indices [4, 3, 2, 1]. Именно эти
        # четыре строки настоящих attention K/V будут собраны ниже.

        # Блоки router здесь уже нигде не используются. Мы собираем настоящие
        # K/V отдельных токенов, выбранных из всей causal history.
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
        # Продолжение примера:
        #   selected_k[query 4] = [k_4, k_3, k_2, k_1]
        #   selected_v[query 4] = [v_4, v_3, v_2, v_1]
        # Здесь k_i/v_i — уже не scalar index_k из router, а настоящие
        # head_size-векторы основного attention для token i.

        # Основной sparse attention не знает про MISA и pooled-блоки.
        # Он получает обычные token indices и считает attention по K позициям.
        wei = (
            q.unsqueeze(2) * selected_k
        ).sum(dim=-1) * self.head_size ** -0.5  # (B, T_query, K)
        # Для query 4 получаются четыре обычных attention logits:
        #   [q_4 @ k_4, q_4 @ k_3, q_4 @ k_2, q_4 @ k_1] / sqrt(head_size)

        selected_is_future = top_indices > torch.arange(T, device=x.device).view(1, T, 1)
        wei = wei.masked_fill(selected_is_future, float('-inf'))  # (B, T_query, K)
        wei = F.softmax(wei, dim=-1)  # (B, T_query, K)

        out = (
            wei.unsqueeze(-1) * selected_v
        ).sum(dim=2)  # (B, T_query, head_size)
        # Если softmax дал [a_4, a_3, a_2, a_1], то:
        #   out[query 4] = a_4*v_4 + a_3*v_3 + a_2*v_2 + a_1*v_1
        return out
