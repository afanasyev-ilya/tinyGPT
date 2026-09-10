"""Минимальная самодостаточная Gated DeltaNet-голова в стиле tinyGPT.

Сначала прочитайте linear_attention.py. Здесь additive-запись заменяется на
Gated Delta Rule, используемый в GDN:

    S = alpha * S
    old = S @ k
    correction = beta * (v - old)
    S = S + outer(correction, k)
    o = scale * S @ q

Цикл показывает семантику. В production TRT-LLM prefill считается chunkwise,
а decode — fused recurrent kernel, а не отдельным Python-шагом на каждый токен.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


def l2_normalize_like_trtllm(x, eps=1e-6):
    """TRT-LLM нормирует Q и K как x / (sqrt(sum(x*x)) + eps)."""
    return x / (torch.linalg.vector_norm(x.float(), dim=-1, keepdim=True) + eps)


def gated_delta_step(state, q_i, k_i, v_i, alpha_i, beta_i, scale):
    """Один GDN token-step: decay, delta update и чтение output."""

    # Формы совпадают с linear_attention_step из linear_attention.py:
    # q_i, k_i, v_i: (B, head_dim), state: (B, head_dim, head_dim).
    # alpha_i и beta_i имеют форму (B): по одному gate на текущий токен
    # для каждого элемента batch в этой учебной single-head версии.
    q_i = q_i.float()
    k_i = k_i.float()
    v_i = v_i.float()

    # 1. Alpha состаривает сразу весь state. Две добавленные оси превращают
    # (B) в (B, 1, 1), чтобы один alpha_i[b] умножил всю матрицу state[b].
    decayed_state = alpha_i[:, None, None] * state.float()

    # 2. Читаем, что состаренный state уже возвращает по текущему адресу k_i.
    # unsqueeze/bmm/squeeze работают точно как в linear_attention_step.
    old_value = torch.bmm(
        decayed_state,
        k_i.unsqueeze(-1),
    ).squeeze(-1)  # (B, head_dim)

    # 3. Beta задаёт, какую долю ошибки между желаемым v_i и old_value записать.
    # beta_i[:, None]: (B) -> (B, 1), один scalar применяется ко всем channels.
    correction = beta_i[:, None] * (v_i - old_value)  # (B, head_dim)

    # 4. В additive linear attention мы всегда прибавляли outer(v_i, k_i).
    # GDN вместо этого прибавляет только correction — ошибку текущей записи.
    new_state = (
        decayed_state
        + correction.unsqueeze(-1) @ k_i.unsqueeze(-2)
    )  # (B, head_dim, head_dim)

    # 5. Читаем output текущей позиции. Это output GDN-head, не следующий token.
    o_i = torch.bmm(
        new_state,
        q_i.unsqueeze(-1),
    ).squeeze(-1) * scale  # (B, head_dim)
    return o_i, new_state


def gated_delta_recurrent(
    q,
    k,
    v,
    alpha,
    beta,
    scale,
    initial_state=None,
):
    """Применить один GDN token-step ко всем известным позициям."""

    # q, k, v: (B, T, head_dim); alpha, beta: (B, T).
    # Как и recurrent_linear_attention, этот wrapper обрабатывает T известных
    # позиций. Prefill вызывает его с T=длине prompt, обычный decode — с T=1.
    B, T, head_dim = q.shape
    if initial_state is None:
        state = torch.zeros(
            B, head_dim, head_dim, dtype=torch.float32, device=q.device
        )
    else:
        state = initial_state.float()

    outputs = []
    for token_pos in range(T):
        # Срезы q/k/v[:, token_pos] совпадают с базовой linear attention.
        # Дополнительно берём alpha и beta именно текущей token-позиции.
        o_i, state = gated_delta_step(
            state,
            q[:, token_pos],
            k[:, token_pos],
            v[:, token_pos],
            alpha[:, token_pos],
            beta[:, token_pos],
            scale,
        )
        outputs.append(o_i)

    # Возвращаем все causal outputs и только финальный state для decode.
    return torch.stack(outputs, dim=1), state


class GatedDeltaHead(nn.Module):
    """Одна учебная GDN-голова — аналог Head из gpt_mha.py."""

    def __init__(self, hidden_dim, head_dim):
        super().__init__()
        self.key = nn.Linear(hidden_dim, head_dim, bias=False)
        self.query = nn.Linear(hidden_dim, head_dim, bias=False)
        self.value = nn.Linear(hidden_dim, head_dim, bias=False)

        # Здесь одна голова, поэтому для токена получаем по одному raw a и b.
        # Настоящий multi-head GigaChat получает такую пару для каждой value-head.
        self.a_projection = nn.Linear(hidden_dim, 1, bias=False)
        self.b_projection = nn.Linear(hidden_dim, 1, bias=False)

        # После обучения это фиксированные параметры конкретных layer/head.
        # Сами alpha и beta всё равно заново зависят от каждого входного токена.
        self.A_log = nn.Parameter(torch.zeros(()))
        self.dt_bias = nn.Parameter(torch.zeros(()))
        self.scale = head_dim**-0.5

    def project_inputs(self, x):
        q = l2_normalize_like_trtllm(self.query(x))
        k = l2_normalize_like_trtllm(self.key(x))
        v = self.value(x)

        a = self.a_projection(x).squeeze(-1)
        b = self.b_projection(x).squeeze(-1)

        # TRT-LLM calls log(alpha) "g" in the Triton API.
        log_alpha = -torch.exp(self.A_log.float()) * F.softplus(
            a.float() + self.dt_bias.float()
        )
        alpha = torch.exp(log_alpha)  # always in (0, 1]
        beta = torch.sigmoid(b.float())  # always in (0, 1)
        return q, k, v, alpha, beta

    def forward(self, x, initial_state=None):
        q, k, v, alpha, beta = self.project_inputs(x)
        output, final_state = gated_delta_recurrent(
            q,
            k,
            v,
            alpha,
            beta,
            self.scale,
            initial_state=initial_state,
        )
        return output, final_state, alpha, beta

    def decode_step(self, x_i, state):
        """Обработать hidden-вектор одного уже выбранного нового токена."""
        q_i, k_i, v_i, alpha_i, beta_i = self.project_inputs(x_i)
        output, new_state = gated_delta_step(
            state,
            q_i,
            k_i,
            v_i,
            alpha_i,
            beta_i,
            self.scale,
        )
        return output, new_state, alpha_i, beta_i


def numerical_alpha_beta_demo():
    """Две независимые записи [10, 20] из gdn_explained.html."""
    state = torch.diag(torch.tensor([10.0, 20.0])).unsqueeze(0)
    q_i = k_i = torch.tensor([[1.0, 0.0]])
    v_i = torch.tensor([[4.0, 0.0]])

    def update(alpha_value, beta_value):
        _, final_state = gated_delta_step(
            state,
            q_i,
            k_i,
            v_i,
            alpha_i=torch.tensor([alpha_value]),
            beta_i=torch.tensor([beta_value]),
            scale=1.0,
        )
        return final_state.diagonal(dim1=-2, dim2=-1).flatten().tolist()

    print("S=[10,20], current key selects 10, new value=4")
    print("alpha=1.0 beta=0.0 ->", update(1.0, 0.0))
    print("alpha=1.0 beta=0.5 ->", update(1.0, 0.5))
    print("alpha=1.0 beta=1.0 ->", update(1.0, 1.0))
    print("alpha=0.5 beta=0.0 ->", update(0.5, 0.0))
    print("alpha=0.5 beta=1.0 ->", update(0.5, 1.0))


def streaming_demo():
    torch.manual_seed(1337)
    B, T, hidden_dim = 2, 6, 8
    head = GatedDeltaHead(hidden_dim, head_dim=4)
    x = torch.randn(B, T, hidden_dim)

    # Полный проход задаёт ожидаемый рекуррентный результат.
    full_output, full_state, alpha, beta = head(x)

    # Prefill обрабатывает четыре известных prompt-токена.
    prefill_output, state, _, _ = head(x[:, :4])

    # Как в linear_attention.py, generation loop находится выше attention и
    # переносит cached state между вызовами одного decode_step.
    decode_outputs = []
    for token_pos in range(4, T):
        o_i, state, _, _ = head.decode_step(x[:, token_pos], state)
        decode_outputs.append(o_i.unsqueeze(1))

    streaming_output = torch.cat([prefill_output, *decode_outputs], dim=1)

    torch.testing.assert_close(streaming_output, full_output)
    torch.testing.assert_close(state, full_state)
    print("PASS: GDN prefill(4) + decode(1) + decode(1) == one full pass")
    print("output shape:", full_output.shape)
    print("cached state shape:", full_state.shape)
    print(
        "data-dependent alpha range:",
        (float(alpha.min().detach()), float(alpha.max().detach())),
    )
    print(
        "data-dependent beta range:",
        (float(beta.min().detach()), float(beta.max().detach())),
    )
    print(
        "Production omissions: multi-head/GVA layout, causal Conv1d, "
        "chunkwise WY prefill, fused decode, request slots, TP/DP and quantization."
    )


if __name__ == "__main__":
    numerical_alpha_beta_demo()
    print()
    streaming_demo()
