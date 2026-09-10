"""Переход от attention из gpt_mha.py к рекуррентной linear attention.

Этот файл намеренно останавливается до Delta Rule, alpha и beta. Здесь показан
ровно один переход: после удаления softmax растущую KV-историю можно свернуть
в матрицу S фиксированного размера.

Было в обычном attention:
    wei = softmax(q @ k.T)
    out = wei @ v

В gpt_mha.py тензор x имеет форму (B, T, hidden_dim), а после projections
q, k, v имеют форму (B, T, head_dim). Ось T — это позиции разных токенов:
    q[:, 0] — query первого токена
    q[:, 1] — query второго токена
    ...
    q[:, i] — query токена на позиции i

Causal mask разрешает токену i смотреть только на токены 0, ..., i:
    i=0: token 0 видит [token 0]
    i=1: token 1 видит [token 0, token 1]
    i=2: token 2 видит [token 0, token 1, token 2]

Поэтому после удаления softmax для каждой позиции i берём свой префикс:
    o_i = (q_i @ K_<=i.T) @ V_<=i

Перегруппировали матричные умножения внутри этого префикса:
    o_i = q_i @ (K_<=i.T @ V_<=i)

Ниже state хранится транспонированно, поэтому та же формула записана как:
    S_i = S_(i-1) + outer(v_i, k_i)
    o_i = S_i @ q_i
"""

import torch
import torch.nn as nn


def quadratic_attention_without_softmax(q, k, v, scale):
    """Знакомый causal attention, только без softmax.

    q, k, v: (B, T, head_dim)
    out:     (B, T, head_dim)
    """
    _, T, _ = q.shape

    # Было в gpt_mha.py перед softmax:
    # (B, T, K) @ (B, K, T) -> (B, T, T).
    scores = q @ k.transpose(-2, -1) * scale

    # Без softmax запрещённые позиции зануляем, а не заполняем -inf.
    causal_mask = torch.tril(
        torch.ones(T, T, dtype=torch.bool, device=q.device)
    )
    scores = scores.masked_fill(~causal_mask, 0.0)

    # (B, T, T) @ (B, T, D) -> (B, T, D).
    return scores @ v


def linear_attention_step(state, q_i, k_i, v_i, scale):
    """Обновить state и получить output одной token-позиции."""

    # Входы этого шага уже не содержат ось времени T:
    # q_i, k_i, v_i: (B, head_dim), state: (B, head_dim, head_dim).
    # Во время обычного decode runtime вызывает этот шаг один раз для нового
    # токена и передаёт new_state в decode следующего токена.

    # Чтобы получить outer product, превращаем value в столбец, а key — в строку.
    # unsqueeze(-1) добавляет новую ось в конец:
    #   v_i:               (B, D)    например [[2, 3]]
    #   v_i.unsqueeze(-1): (B, D, 1)          [[[2], [3]]]
    # unsqueeze(-2) добавляет ось перед последней существующей осью:
    #   k_i:               (B, D)    например [[1, 0]]
    #   k_i.unsqueeze(-2): (B, 1, D)          [[[1, 0]]]
    # Теперь @ для каждого элемента batch перемножает (D, 1) @ (1, D):
    #   [[2], [3]] @ [[1, 0]] = [[2, 0], [3, 0]]
    # Полученная (B, D, D) матрица — новая запись outer(v_i, k_i).
    new_state = state + v_i.unsqueeze(-1) @ k_i.unsqueeze(-2)

    # Для чтения из state query тоже должен быть вектором-столбцом:
    #   q_i:               (B, D)    например [[1, 0]]
    #   q_i.unsqueeze(-1): (B, D, 1)          [[[1], [0]]]
    # torch.bmm — batch matrix multiplication; для каждого b он считает отдельно:
    #   new_state[b] @ q_i[b]
    # В toy-примере при state=0:
    #   [[2, 0], @ [[1], = [[2],
    #    [3, 0]]    [0]]    [3]]
    # Результат bmm имеет форму (B, D, 1). squeeze(-1) удаляет последнюю ось
    # размера 1 и возвращает обычный вектор (B, D): [[[2], [3]]] -> [[2, 3]].
    # scale=head_dim**-0.5 — один скаляр, которым умножается весь output.
    o_i = torch.bmm(new_state, q_i.unsqueeze(-1)).squeeze(-1) * scale
    return o_i, new_state


def recurrent_linear_attention(q, k, v, scale, initial_state=None):
    """Применить token-step ко всем позициям известной последовательности."""

    # q, k, v имеют форму (B, T, head_dim): по одной строке длины head_dim
    # для каждого из T токенов в каждом из B независимых элементов batch.
    # Небольшой пример с B=1, T=2, head_dim=2:
    #   q = [[[1, 0], [0, 1]]]
    #   k = [[[1, 0], [0, 1]]]
    #   v = [[[2, 3], [4, 5]]]
    # В нём строки с индексами 0 и 1 соответствуют двум разным token-позициям.

    # Эта строка только читает размеры тензора; никаких вычислений с q не делает.
    B, T, head_dim = q.shape

    # В начале первого prefill исторических записей нет, поэтому S_0 — нулевая
    # матрица для каждого элемента batch. На decode initial_state — это S_T,
    # сохранённый после prompt или предыдущего decode-шага.
    if initial_state is None:
        state = torch.zeros(
            B, head_dim, head_dim, dtype=q.dtype, device=q.device
        )
    else:
        state = initial_state

    # Здесь накопятся T отдельных outputs формы (B, head_dim).
    outputs = []

    # Это цикл по уже известным входным позициям, а не generation loop с выбором
    # новых token IDs. Prefill вызывает его с T=длине prompt, а decode — с T=1.
    for token_pos in range(T):
        # Синтаксис q[:, token_pos] означает:
        #   :          — взять все B элементов batch;
        #   token_pos  — взять одну строку на оси времени T.
        # Ось T после выбора одной строки исчезает:
        #   q:                 (B, T, D)
        #   q[:, token_pos]:   (B, D)
        # Для toy-примера выше:
        #   token_pos=0: q[:, 0] = [[1, 0]], k[:, 0] = [[1, 0]],
        #                v[:, 0] = [[2, 3]]
        #   token_pos=1: q[:, 1] = [[0, 1]], k[:, 1] = [[0, 1]],
        #                v[:, 1] = [[4, 5]]
        o_i, state = linear_attention_step(
            state,
            q[:, token_pos],
            k[:, token_pos],
            v[:, token_pos],
            scale,
        )

        # При начальном state=0 toy-пример проходит так:
        #   token_pos=0:
        #     S_1 = [[2, 0], [3, 0]]
        #     o_1 = S_1 @ [1, 0] = [2, 3]
        #   token_pos=1:
        #     S_2 = S_1 + [[0, 4], [0, 5]] = [[2, 4], [3, 5]]
        #     o_2 = S_2 @ [0, 1] = [4, 5]
        outputs.append(o_i)

    # outputs — список из T тензоров (B, D). stack вставляет обратно ось времени
    # в позицию dim=1: T * (B, D) -> (B, T, D).
    # Наружу возвращается только последний state S_T для продолжения decode.
    return torch.stack(outputs, dim=1), state


class AdditiveLinearAttentionHead(nn.Module):
    """Одна tinyGPT-style голова с рекуррентной матрицей вместо KV-cache."""

    def __init__(self, hidden_dim, head_dim):
        super().__init__()
        self.key = nn.Linear(hidden_dim, head_dim, bias=False)
        self.query = nn.Linear(hidden_dim, head_dim, bias=False)
        self.value = nn.Linear(hidden_dim, head_dim, bias=False)
        self.scale = head_dim**-0.5

    def project_qkv(self, x):
        return self.query(x), self.key(x), self.value(x)

    def forward(self, x, initial_state=None):
        q, k, v = self.project_qkv(x)
        return recurrent_linear_attention(
            q, k, v, self.scale, initial_state=initial_state
        )

    def decode_step(self, x_i, state):
        """Обработать hidden-вектор одного уже выбранного нового токена."""
        q_i = self.query(x_i)  # (B, head_dim)
        k_i = self.key(x_i)  # (B, head_dim)
        v_i = self.value(x_i)  # (B, head_dim)
        return linear_attention_step(state, q_i, k_i, v_i, self.scale)


def demo():
    torch.manual_seed(1337)
    B, T, hidden_dim = 2, 6, 8
    head_dim = 4
    x = torch.randn(B, T, hidden_dim)
    head = AdditiveLinearAttentionHead(hidden_dim, head_dim)
    q, k, v = head.project_qkv(x)

    # Было: строим полную causal-матрицу scores размера T x T.
    quadratic_out = quadratic_attention_without_softmax(
        q, k, v, head.scale
    )

    # Стало: по одной сворачиваем пары (k_i, v_i) в state фиксированного размера.
    recurrent_out, full_state = recurrent_linear_attention(
        q, k, v, head.scale
    )
    torch.testing.assert_close(recurrent_out, quadratic_out)

    # Prefill обрабатывает все четыре уже известных prompt-токена.
    prefill_out, state = head(x[:, :4])

    # Настоящий generation loop живёт выше attention: модель выбирает следующий
    # token, строит его hidden-вектор x_i и вызывает каждый layer ровно один раз.
    # Здесь x[:, token_pos] изображает уже полученный hidden-вектор нового токена.
    decode_outputs = []
    for token_pos in range(4, T):
        o_i, state = head.decode_step(x[:, token_pos], state)
        decode_outputs.append(o_i.unsqueeze(1))

    streaming_out = torch.cat([prefill_out, *decode_outputs], dim=1)
    torch.testing.assert_close(streaming_out, recurrent_out)
    torch.testing.assert_close(state, full_state)

    print("PASS: quadratic no-softmax attention == recurrent S @ q")
    print("PASS: prefill(4) + decode(1) + decode(1) == one full pass")
    print("q/k/v shapes:", q.shape, k.shape, v.shape)
    print("cached state shape:", full_state.shape, "= (B, head_dim, head_dim)")
    print(
        "per-sequence cache elements after 6 tokens:",
        f"KV={2 * T * head_dim}, recurrent S={head_dim**2}",
    )
    print(
        "Without softmax, each causal prefix can be folded into its S_i. "
        "One global Q @ (K.T @ V) would leak future tokens."
    )


if __name__ == "__main__":
    demo()
