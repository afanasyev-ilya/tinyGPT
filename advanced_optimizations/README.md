# Операции LLM на примере GigaChat и DeepSeek

Математика, размерности, небольшие численные примеры и учебные реализации на C++17 и Python/PyTorch. Рассматриваются конкретные открытые архитектуры: **DeepSeek V3**, **DeepSeek V3.2** и **GigaChat 3.5 432B-A28B**.

Названия моделей в таблице ведут к их архитектуре ниже, «Формула и код» — к соответствующей операции. Составные блоки вынесены в отдельную таблицу сразу после базовых операций.

<a name="operations"></a>

## Таблица базовых операций

| Операция | Где используется | Модели | Описание |
|---|---|---|---|
| Linear / GEMM / dot product | Проекции [attention](#block-attention), [MLA](#block-mla), [FFN](#block-swiglu), [MoE](#block-moe), LM head | [DeepSeek V3](#model-deepseek-v3), [DeepSeek V3.2](#model-deepseek-v32), [GigaChat 3.5](#model-gigachat35) | [Формула и код](#op-linear) |
| Embedding lookup | Вход модели: token ID → hidden vector | [DeepSeek V3](#model-deepseek-v3), [DeepSeek V3.2](#model-deepseek-v32), [GigaChat 3.5](#model-gigachat35) | [Формула и код](#op-embedding) |
| Сложение | [Residual](#op-residual), bias, correction scores router | [DeepSeek V3](#model-deepseek-v3), [DeepSeek V3.2](#model-deepseek-v32), [GigaChat 3.5](#model-gigachat35) | [Формула и код](#op-add) |
| Поэлементное умножение | [SwiGLU](#block-swiglu), нормы, gates, веса экспертов | [DeepSeek V3](#model-deepseek-v3), [DeepSeek V3.2](#model-deepseek-v32), [GigaChat 3.5](#model-gigachat35) | [Формула и код](#op-mul) |
| Sum / mean / reduction | Нормировки, attention weighted sum, combine экспертов | [DeepSeek V3](#model-deepseek-v3), [DeepSeek V3.2](#model-deepseek-v32), [GigaChat 3.5](#model-gigachat35) | [Формула и код](#op-reduce) |
| **Softmax** | [Attention / MLA](#block-mla): logits → веса values; после LM head — вероятности токенов | [DeepSeek V3](#model-deepseek-v3), [DeepSeek V3.2](#model-deepseek-v32), [GigaChat 3.5](#model-gigachat35) | [Формула и код](#op-softmax) |
| **Sigmoid** | [MoE router](#block-moe); у GigaChat также β и выходные gates [GDN](#block-gdn), gate MLA и [GatedNorm](#block-gatednorm) | [DeepSeek V3](#model-deepseek-v3), [DeepSeek V3.2](#model-deepseek-v32), [GigaChat 3.5](#model-gigachat35) | [Формула и код](#op-sigmoid) |
| **SiLU / Swish** | [SwiGLU](#block-swiglu) dense FFN и экспертов; у GigaChat также GatedNorm и Q/K/V после GDN conv | [DeepSeek V3](#model-deepseek-v3), [DeepSeek V3.2](#model-deepseek-v32), [GigaChat 3.5](#model-gigachat35) | [Формула и код](#op-silu) |
| ReLU | Положительная часть QK dot product в [DSA indexer](#block-dsa) | [DeepSeek V3.2](#model-deepseek-v32) | [Формула и код](#op-relu) |
| Softplus | Параметризация затухания памяти [GDN](#block-gdn) | [GigaChat 3.5](#model-gigachat35) | [Формула и код](#op-softplus) |
| Exp / log / sqrt / rsqrt | Softmax, активации, нормировки; у GigaChat также decay GDN | [DeepSeek V3](#model-deepseek-v3), [DeepSeek V3.2](#model-deepseek-v32), [GigaChat 3.5](#model-gigachat35) | [Формула и код](#op-elementary) |
| LayerNorm | Key-проекция [DSA indexer](#block-dsa) | [DeepSeek V3.2](#model-deepseek-v32) | [Формула и код](#op-layernorm) |
| RMSNorm | Основные и latent-нормы DeepSeek; основа GatedNorm и выходной нормы GDN в GigaChat | [DeepSeek V3](#model-deepseek-v3), [DeepSeek V3.2](#model-deepseek-v32), [GigaChat 3.5](#model-gigachat35) | [Формула и код](#op-rmsnorm) |
| Zero-centered RMSNorm | Scale γ = 1 + w в [GatedNorm](#block-gatednorm) и выходной норме [GDN](#block-gdn) | [GigaChat 3.5](#model-gigachat35) | [Формула и код](#op-zero-centered) |
| L2-нормировка Q/K | Подготовка query/key [GDN](#block-gdn) | [GigaChat 3.5](#model-gigachat35) | [Формула и код](#op-l2norm) |
| RoPE | Позиционные Q/K [MLA](#block-mla); в V3.2 также [DSA](#block-dsa) | [DeepSeek V3](#model-deepseek-v3), [DeepSeek V3.2](#model-deepseek-v32), [GigaChat 3.5](#model-gigachat35) | [Формула и код](#op-rope) |
| Маскирование | Causal [attention](#block-attention), допустимые группы MoE; в V3.2 также DSA selection mask | [DeepSeek V3](#model-deepseek-v3), [DeepSeek V3.2](#model-deepseek-v32), [GigaChat 3.5](#model-gigachat35) | [Формула и код](#op-mask) |
| Top-k экспертов | [MoE](#block-moe): выбор routed experts; в DeepSeek также отбор групп | [DeepSeek V3](#model-deepseek-v3), [DeepSeek V3.2](#model-deepseek-v32), [GigaChat 3.5](#model-gigachat35) | [Формула и код](#op-topk) |
| Top-k позиций | [DSA](#block-dsa): выбор до 2048 токенов истории | [DeepSeek V3.2](#model-deepseek-v32) | [Формула и код](#op-topk) |
| Gather / index select | Embedding, веса выбранных экспертов; в DSA — выбранные KV-записи | [DeepSeek V3](#model-deepseek-v3), [DeepSeek V3.2](#model-deepseek-v32), [GigaChat 3.5](#model-gigachat35) | [Формула и код](#op-gather) |
| Scatter-add / indexed sum | [MoE combine](#block-moe): сложение expert outputs по исходным token IDs | [DeepSeek V3](#model-deepseek-v3), [DeepSeek V3.2](#model-deepseek-v32), [GigaChat 3.5](#model-gigachat35) | [Формула и код](#op-scatter) |
| Reshape / transpose / concat | Раскладка heads, соединение частей Q/K и выходов heads | [DeepSeek V3](#model-deepseek-v3), [DeepSeek V3.2](#model-deepseek-v32), [GigaChat 3.5](#model-gigachat35) | [Формула и код](#op-layout) |
| Clamp / clipping активаций | Ограничение gate и up перед [SwiGLU](#block-swiglu) | [GigaChat 3.5](#model-gigachat35) | [Формула и код](#op-clamp) |
| Causal depthwise Conv1d | Короткое смешивание Q/K/V перед ядром [GDN](#block-gdn) | [GigaChat 3.5](#model-gigachat35) | [Формула и код](#op-conv) |
| Outer product / rank-one update | Корректировка матрицы состояния [GDN](#block-gdn) | [GigaChat 3.5](#model-gigachat35) | [Формула и код](#op-outer) |
| Нормированный Hadamard | [DSA indexer](#block-dsa): вращение Q/K перед FP8 | [DeepSeek V3.2](#model-deepseek-v32) | [Формула и код](#op-hadamard) |
| FP8 quantize / dequantize | Представление весов/активаций в FP8-вариантах; дополнительно DSA indexer | [DeepSeek V3](#model-deepseek-v3), [DeepSeek V3.2](#model-deepseek-v32), [GigaChat 3.5](#model-gigachat35) | [Формула и код](#op-fp8) |

**Область таблицы.** «Используется» означает математическое присутствие операции, а не отдельный CUDA kernel: несколько операций могут быть fused. FP8 относится к соответствующему checkpoint/backend. Softmax после LM head — отдельное преобразование logits при генерации, а не часть Linear. DSA и GDN принадлежат указанным версиям; наличие операции не переносится автоматически на всё семейство моделей. Источники сведены в [конце документа](#sources).

### Составные блоки

| Блок | Из каких операций состоит | Модели | Описание |
|---|---|---|---|
| Attention | Q/K/V Linear, QKᵀ, scale, mask, softmax, weighted sum, output Linear | Основа MLA у всех рассматриваемых моделей | [Attention](#block-attention) |
| MLA | Сжатие Q/KV, нормы, отдельный RoPE, attention, output projection | [DeepSeek V3](#model-deepseek-v3), [V3.2](#model-deepseek-v32), [GigaChat 3.5](#model-gigachat35) | [MLA](#block-mla) |
| DSA | Index projections, LayerNorm, RoPE, Hadamard/FP8, ReLU, weighted sum, token top-k, выбранные позиции MLA | [DeepSeek V3.2](#model-deepseek-v32) | [DSA](#block-dsa) |
| FFN / SwiGLU | Две входные Linear, SiLU, поэлементное произведение, выходная Linear | [DeepSeek V3](#model-deepseek-v3), [V3.2](#model-deepseek-v32), [GigaChat 3.5](#model-gigachat35) | [SwiGLU](#block-swiglu) |
| MoE | Router Linear, sigmoid, correction, top-k, выбранные SwiGLU experts, shared expert, взвешенная сумма | [DeepSeek V3](#model-deepseek-v3), [V3.2](#model-deepseek-v32), [GigaChat 3.5](#model-gigachat35) | [MoE](#block-moe) |
| GatedNorm | Zero-centered RMSNorm, маленький MLP с SiLU, gate 2·sigmoid, произведение | [GigaChat 3.5](#model-gigachat35) | [GatedNorm](#block-gatednorm) |
| Gated DeltaNet | Проекции, causal conv, SiLU, L2 Q/K, decay, delta update, output norm/gate | [GigaChat 3.5](#model-gigachat35) | [GDN](#block-gdn) |
| LM head | Linear из hidden state в vocabulary logits | Все три модели | [LM head](#op-lm-head) |

Навигация: [архитектуры](#models) · [обозначения и запуск](#notation) · [словарь PyTorch](#python-glossary) · [чтение профиля](#profiling) · [источники](#sources).

<a name="models"></a>

## Архитектуры моделей

Входной путь всех трёх моделей: token IDs → [embedding](#op-embedding) → decoder-блоки → финальная нормировка → [LM head](#op-lm-head). MoE расположен в FFN-части decoder-блока.

<a name="model-deepseek-v3"></a>

### DeepSeek V3

Базовая связка — **MLA + FFN/MoE**. Нормировка перед каждой ветвью — RMSNorm, выход ветви прибавляется к residual:

```math
h=x+\mathrm{MLA}(N_1(x)),\qquad y=h+\mathrm{FFN/MoE}(N_2(h)).
```

$x,h,y$ — hidden states одинаковой ширины, $N_1,N_2$ — разные обученные RMSNorm. Первые три FFN dense, последующие используют MoE: top-8 из 256 routed experts и один shared expert.

- [Softmax](#op-softmax) нормирует основные attention logits MLA.
- [Sigmoid](#op-sigmoid) даёт affinity scores MoE router. Correction bias влияет на выбор, а веса смешивания берутся из исходных affinity scores.
- [SiLU](#op-silu) входит в SwiGLU dense FFN и каждого эксперта.
- [RMSNorm](#op-rmsnorm) применяется также к сжатым Q/KV; [RoPE](#op-rope) — к позиционной части Q/K.

DSA и GDN не входят в рассматриваемый V3. Источник: [DeepSeek V3 Technical Report](https://arxiv.org/html/2412.19437v2).

<a name="model-deepseek-v32"></a>

### DeepSeek V3.2

V3.2 сохраняет основу V3 и добавляет **DSA внутрь MLA**. Отдельный lightning indexer выбирает до 2048 позиций истории; основной MLA затем вычисляет attention по выбранным позициям.

Путь индексатора: query из сжатой query-ветви MLA; key через Linear и [LayerNorm](#op-layernorm); частичный RoPE; общий Hadamard и FP8; QK dot products → [ReLU](#op-relu) → взвешенная сумма по heads → [top-k позиций](#op-topk).

После выбора используются **отдельные MLA logits и softmax**. Index scores не становятся attention weights. Веса heads индексатора — сырые выходы Linear; они не проходят sigmoid или softmax. [Sigmoid](#op-sigmoid) остаётся в MoE router, [SiLU](#op-silu) — в FFN и экспертах. Это два разных отбора: MoE top-k выбирает экспертов, а DSA top-k — позиции контекста.

Конфигурация V3.2: hidden size 7168, 61 слой; 256 routed experts, top-8, один shared expert; отбор 4 из 8 групп перед expert top-k. Источники: [конфигурация](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json), [отчёт V3.2](https://arxiv.org/html/2512.02556v1), [код индексатора и MLA](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py). Карточка V3.2 направляет к inference-коду V3.2-Exp с той же архитектурой.

<a name="model-gigachat35"></a>

### GigaChat 3.5

Открытый GigaChat 3.5 432B-A28B содержит **40 слоёв: 30 GDN и 10 MLA**. Первые три FFN dense; следующие используют MoE с top-8 из 256 routed experts и одним shared expert. Групповой фильтр router тривиален: одна группа.

В режиме `pre_post`:

```math
h=x+GN_2(A(GN_1(x))),\qquad y=h+GN_4(F(GN_3(h))).
```

$A$ — MLA либо GDN, $F$ — dense FFN либо MoE; $GN_1,\ldots,GN_4$ — разные GatedNorm. Выходная нормировка ветви выполняется **до residual add**.

Здесь три разных механизма gating:

| Место | Gate |
|---|---|
| [GatedNorm](#block-gatednorm) | Нормированный вход $n$ умножается на $2\sigma(W_{down}\mathrm{SiLU}(W_{up}n))$ |
| [Выход GDN](#block-gdn) | Каждый head после zero-centered RMSNorm умножается на $2\sigma(z)$ |
| [Выход MLA](#block-mla) | Склеенные head outputs умножаются на $\sigma(W_{gate}a)$ перед $W_O$, без множителя 2 |

$a=GN_1(x)$ — вход attention-ветви после pre-нормировки; из него также вычисляется отдельная проекция $z$ в GDN. $W_{gate},W_{up},W_{down},W_O$ — обученные матрицы. Zero-centered — параметризация scale $\gamma=1+w$, не вычитание среднего.

[Softmax](#op-softmax) используется в MLA. В recurrent GDN attention softmax нет. [Sigmoid](#op-sigmoid) присутствует в MoE router и gates, [SiLU](#op-silu) — в SwiGLU, gate-MLP GatedNorm и после GDN-свёртки Q/K/V. [Softplus](#op-softplus) участвует в decay GDN. В SwiGLU gate ограничивается сверху значением 10, up — диапазоном [−10,10].

Источники: [конфигурация](https://huggingface.co/ai-sage/GigaChat3.5-432B-A28B/blob/main/config.json), [реализация](https://huggingface.co/ai-sage/GigaChat3.5-432B-A28B/blob/main/modeling_gigachat3_5.py). Эти детали относятся к указанному checkpoint; другие версии GigaChat могут отличаться.

<a name="notation"></a>

## Обозначения и соглашения

| Символ | Значение |
|---|---|
| $t,j$ | Позиция текущего токена и позиция токена контекста |
| $B,T,S$ | Batch size, число query-токенов и число доступных key/value-токенов |
| $D,M$ | Размер hidden state и промежуточная ширина FFN |
| $H,d_k,d_v$ | Число attention heads, ширина key/query и value одного head |
| $x_t\in\mathbb R^D$ | Вектор признаков одного токена |
| $W\in\mathbb R^{D_{out}\times D_{in}}$ | Обученная матрица Linear; $b$ — её bias |
| $\gamma,\eta$ | Обученные scale и shift нормировки; shift здесь обозначаем $\eta$, чтобы не смешивать с GDN gate $\beta$ |
| $\varepsilon>0$ | Стабилизатор знаменателя |
| $\odot,\; ^\top$ | Поэлементное умножение и транспонирование |
| $\sigma$ | Sigmoid; $\exp$ и $\log$ — экспонента и натуральный логарифм |
| $\mathrm{Concat}$ | Склейка векторов по feature-оси |

В формулах отдельные векторы — **столбцы**: $y=Wx+b$. В коде токены обычно хранятся строками: `X @ W.T + b`. Это одна операция при разном соглашении о хранении. В численных примерах запись `[a,b]` обозначает компоненты вектора, а не новое соглашение для формул.

### Как читать и запускать код

Документ самодостаточен: Python-фрагменты читаются и выполняются сверху вниз, функции определяются до использования. Для отдельного раздела нужны общий префикс и вызываемые в нём функции предыдущих разделов. Требуются Python и PyTorch 2.x; веса моделей и GPU не нужны. В исходных примерах использован PyTorch 2.9.1 CPU и float64 для читаемых чисел.

```python
import math
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=6, sci_mode=False)
```

`F` — сокращённое имя `torch.nn.functional`; `F.silu` применяет функцию без создания обучаемого слоя. Примеры показывают математику. Они не воспроизводят все dtypes, fused kernels и раскладки production-реализаций. В DSA и MLA явно указано, какие входные проекции уже считаются выполненными.

**Общий префикс C++17:**

```cpp
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

using Vec = std::vector<double>;
using Mat = std::vector<Vec>;

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
```

C++ функции ниже используют этот префикс и функции предыдущих разделов. Их можно собрать в один `.cpp`, добавить `main()` с нужным примером и скомпилировать через `g++ -std=c++17 -O2 example.cpp -o example`.

<a name="primitive-ops"></a>

## 0 Поэлементные операции и раскладка тензоров

<a name="op-add"></a>

### Сложение

```math
y_i=a_i+b_i.
```

$a,b,y\in\mathbb R^D$. Пример: `[1,2]+[0.1,0.2]=[1.1,2.2]`. В residual складываются hidden states одинаковой формы; при добавлении bias он может распространяться на все токены через broadcasting. [C++ реализация residual add](#op-residual).

<a name="op-mul"></a>

### Поэлементное умножение

```math
y_i=a_i b_i,\qquad y=a\odot b.
```

Пример: `[1,2] * [3,4] = [3,8]`. Это не dot product, который для тех же векторов равен 11. Применения: SwiGLU, gates, масштабы нормировок, веса expert outputs.

```cpp
Vec elementwise_mul(const Vec& a, const Vec& b) {
    require(a.size() == b.size(), "mul: incompatible dimensions");
    Vec y(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) y[i] = a[i] * b[i];
    return y;
}
```

<a name="op-reduce"></a>

### Сумма и среднее

```math
s=\sum_{i=1}^{D}x_i,\qquad \mu=s/D.
```

Для `[1,2,3]`: сумма 6, среднее 2. Редукция уменьшает одну из осей; в нормировках обычно сворачивается ось признаков одного токена. `keepdim=True` сохраняет ось длины 1 для последующего broadcasting.

```cpp
double sum_values(const Vec& x) {
    return std::accumulate(x.begin(), x.end(), 0.0);
}

double mean_values(const Vec& x) {
    require(!x.empty(), "mean: empty vector");
    return sum_values(x) / x.size();
}
```

<a name="op-elementary"></a>

### Exp log sqrt и rsqrt

```math
\mathrm{exp}(x)=e^x,\quad \mathrm{log}(x)=\ln x,\quad
\mathrm{sqrt}(x)=\sqrt{x},\quad \mathrm{rsqrt}(x)=1/\sqrt{x}.
```

`log` требует положительного аргумента, `sqrt` — неотрицательного, `rsqrt` — положительного. Примеры: `exp(0)=1`, `log(e)=1`, `sqrt(4)=2`, `rsqrt(4)=0.5`. В коде C++ это `std::exp`, `std::log`, `std::sqrt`, `1.0/std::sqrt(x)`. Для tensor-входа PyTorch применяет функции поэлементно. Они входят в softmax, активации, нормировки и decay GDN.

<a name="op-layout"></a>

### Reshape transpose concat и broadcasting

Эти операции организуют данные по осям. Они не добавляют обучаемых параметров. Для матрицы $X$ транспонирование задаётся $Y_{ij}=X_{ji}$; reshape меняет разбиение того же числа элементов на оси; concat соединяет тензоры вдоль одной оси.

Пример: `[1,2,3,4]` → reshape(2,2) → `[[1,2],[3,4]]`; transpose → `[[1,3],[2,4]]`. Concat `[1,2]` и `[3,4]` даёт `[1,2,3,4]`. Broadcasting позволяет, например, прибавить `[10,20]` к каждой строке матрицы 2×2. В реализации transpose может быть view со strides, а последующий `.contiguous()` материализует нужную раскладку.

```python
layout_x = torch.tensor([1., 2., 3., 4.])
layout_matrix = layout_x.reshape(2, 2)
print(layout_matrix.T)  # [[1,3],[2,4]]
print(torch.cat((layout_x[:2], layout_x[2:]), dim=0))
```

`.reshape(2,2)` создаёт две оси; `.T` меняет местами оси 2D-матрицы; `[:2]` берёт первые два элемента, `[2:]` — остальные; `torch.cat` склеивает существующие оси, тогда как `torch.stack` создаёт новую. Подробные формы остальных операторов есть в [словаре](#python-glossary).

<a name="op-linear"></a>

## 1 Скалярное произведение Linear и GEMM

**Назначение:** смешать признаки и перейти к новому пространству. Linear используется для Q/K/V, router, FFN и LM head.

```math
a^\top b=\sum_{i=1}^{D}a_ib_i,\qquad y_o=\sum_{i=1}^{D_{in}}W_{oi}x_i+b_o.
```

$a,b\in\mathbb R^D$ — векторы; $i$ нумерует входные, $o$ — выходные признаки. Для $N$ токенов $X\in\mathbb R^{N\times D_{in}}$: $Y=XW^\top+\mathbf1_Nb^\top$. Здесь $\mathbf1_N$ — столбец единиц, реализующий добавление одного bias к каждой строке.

GEMM — матричное умножение, обычно записываемое $C\leftarrow\alpha AB+\beta C$. $\alpha,\beta$ здесь скалярные коэффициенты GEMM. Это вычислительный строительный блок Linear; произвольное GEMM не обязательно является слоем модели.

**Пример:** $[1,2]\cdot[3,4]=11$. При

```math
W=\begin{bmatrix}1&0\\0&1\\1&-1\end{bmatrix},\quad x=\begin{bmatrix}1\\2\end{bmatrix},\quad b=\begin{bmatrix}.5\\-.5\\2\end{bmatrix}
```

получаем $y=[1.5,1.5,1]$. Многие проекции этих моделей используют $b=0$; ненулевой bias здесь выбран для наглядности.

**C++17.**

```cpp
double dot(const Vec& a, const Vec& b) {
    require(a.size() == b.size(), "dot: incompatible dimensions");
    double result = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) result += a[i] * b[i];
    return result;
}

Vec linear(const Vec& x, const Mat& W, const Vec& b) {
    require(W.size() == b.size(), "linear: incompatible bias");
    Vec y(W.size());
    for (std::size_t o = 0; o < W.size(); ++o) y[o] = dot(W[o], x) + b[o];
    return y;
}
```

```python
x = torch.tensor([1., 2.])
W = torch.tensor([[1., 0.], [0., 1.], [1., -1.]])
b = torch.tensor([.5, -.5, 2.])
print(x @ W.T + b)
```

Вывод примера:

```text
tensor([1.500000, 1.500000, 1.000000])
```

<a name="op-embedding"></a>

## 2 Embedding и LM head

**Embedding** выбирает строку таблицы, соответствующую целочисленному token ID:

```math
x_t=E_{u_t,:},\qquad E\in\mathbb R^{V\times D},\quad u_t\in\{0,\ldots,V-1\}.
```

$V$ — размер словаря, $E$ — обученная таблица, $u_t$ — ID. Математически это то же, что умножить one-hot вектор на таблицу, но реализуют прямой lookup.

**Пример:** $E=[[10,11],[20,21],[30,31]]$, IDs `[2,0]` дают `[[30,31],[10,11]]`.

<a name="op-lm-head"></a>

### LM head

**LM head** преобразует конечный hidden state в оценки токенов словаря:

```math
\ell=W_{vocab}h,\quad W_{vocab}\in\mathbb R^{V\times D},\quad p_i=\mathrm{softmax}(\ell)_i.
```

$h$ — вектор после последней нормировки, $\ell$ — logits, $p_i$ — вероятность следующего token ID $i$. Например, $h=[1,2]$ и строки $W_{vocab}=[1,0],[0,1],[1,1]$ дают logits `[1,2,3]`. Сам Linear не выполняет softmax. Embedding и LM head могут иметь разные веса; у выбранных checkpoints `tie_word_embeddings=false`.

**C++17.**

```cpp
Mat embedding(const std::vector<std::size_t>& token_ids, const Mat& E) {
    Mat x;
    for (std::size_t id : token_ids) x.push_back(E.at(id));
    return x;
}
```

```python
E = torch.tensor([[10., 11.], [20., 21.], [30., 31.]])
ids = torch.tensor([2, 0])  # integer tensor, не float
print("Embedding:", E[ids])
W_vocab = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
print("LM logits:", torch.tensor([1., 2.]) @ W_vocab.T)
```

Вывод примера:

```text
Embedding: tensor([[30., 31.],
        [10., 11.]])
LM logits: tensor([1., 2., 3.])
```

<a name="activations"></a>

## 3 ReLU sigmoid SiLU и softplus

Все четыре функции применяются **к каждой координате независимо**, без смешивания токенов или признаков.

| Функция | Формула | Пример для $z=-1,0,2$ |
|---|---|---|
| ReLU | $\max(0,z)$ | $0,0,2$ |
| Sigmoid | $\sigma(z)=1/(1+e^{-z})$ | $0.268941,0.5,0.880797$ |
| SiLU | $z\sigma(z)$ | $-0.268941,0,1.761594$ |
| Softplus | $\log(1+e^z)$ | $0.313262,0.693147,2.126928$ |

$z$ — скалярный вход. Sigmoid даёт коэффициент от 0 до 1; SiLU — гладкая активация с небольшими отрицательными выходами; softplus даёт положительное число и удобна для параметризации скорости затухания.

<a name="op-relu"></a>

### ReLU

```math
\mathrm{ReLU}(z)=\max(0,z).
```

Отбрасывает отрицательную часть скалярного входа $z$. Пример: `[-1,0,2] → [0,0,2]`. В [DeepSeek V3.2](#model-deepseek-v32) применяется к QK dot products индексатора DSA. Основные FFN выбранных моделей используют SiLU внутри SwiGLU.

<a name="op-sigmoid"></a>

### Sigmoid

```math
\sigma(z)=\frac1{1+e^{-z}}.
```

Преобразует каждую координату отдельно в число от 0 до 1. Пример: `[-1,0,2] → [0.268941,0.5,0.880797]`. Сумма результатов не обязана равняться 1: в отличие от softmax, координаты не нормируются друг относительно друга.

В [MoE](#block-moe) обоих семейств sigmoid применяется к router logits. После top-k выбранные affinity scores отдельно нормируются для смешивания экспертов. В [GigaChat 3.5](#model-gigachat35) sigmoid также участвует в GatedNorm, β GDN, выходном gate GDN и gate MLA; это разные места применения одной скалярной функции.

<a name="op-silu"></a>

### SiLU

```math
\mathrm{SiLU}(z)=z\sigma(z).
```

Пример: `[-1,0,2] → [-0.268941,0,1.761594]`. Выход не ограничен диапазоном [0,1]; отрицательные входы не всегда превращаются в ноль. В [SwiGLU](#block-swiglu) SiLU применяется к gate-проекции до умножения на up-проекцию. В GigaChat дополнительно используется после GDN conv и в bottleneck MLP GatedNorm.

<a name="op-softplus"></a>

### Softplus

```math
\mathrm{softplus}(z)=\log(1+e^z).
```

Пример: `[-1,0,2] → [0.313262,0.693147,2.126928]`. Даёт положительное число; в [GDN](#block-gdn) из него строится скорость затухания. Устойчивая форма: $\max(z,0)+\log(1+e^{-|z|})$. Она избегает переполнения $e^z$ при больших положительных $z$.

Следующие C++ функции и Python-пример покрывают все четыре активации. В `sigmoid` две ветви избегают overflow; `log1p(v)` точно вычисляет $\log(1+v)$ при малом $v$.

**C++17.**

```cpp
double relu(double x) { return std::max(0.0, x); }

double sigmoid(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    const double e = std::exp(x);
    return e / (1.0 + e);
}

double silu(double x) { return x * sigmoid(x); }

double softplus(double x) {
    return std::max(x, 0.0) + std::log1p(std::exp(-std::abs(x)));
}
```

```python
z = torch.tensor([-1., 0., 2.])
print("ReLU:", torch.relu(z))
print("sigmoid:", torch.sigmoid(z))
print("SiLU:", F.silu(z))
print("softplus:", F.softplus(z))
```

Вывод примера:

```text
ReLU: tensor([0., 0., 2.])
sigmoid: tensor([0.268941, 0.500000, 0.880797])
SiLU: tensor([-0.268941,  0.000000,  1.761594])
softplus: tensor([0.313262, 0.693147, 2.126928])
```

<a name="normalizations"></a>

## 4 LayerNorm RMSNorm и L2 нормировка

Нормируем признаки **одного токена**, а не всю последовательность или batch.

<a name="op-layernorm"></a>

### LayerNorm

```math
\mu=\frac1D\sum_i x_i,\qquad v=\frac1D\sum_i(x_i-\mu)^2,\qquad y_i=\gamma_i\frac{x_i-\mu}{\sqrt{v+\varepsilon}}+\eta_i.
```

$\mu$ — среднее, $v$ — population variance с делителем $D$, $\gamma_i,\eta_i$ — обученные scale/shift. После обученного affine-преобразования выход уже не обязан иметь нулевое среднее и единичную дисперсию. Для $x=[1,3]$, $\gamma=[1,1]$, $\eta=0$, без $\varepsilon$: $\mu=2,v=1,y=[-1,1]$.

<a name="op-rmsnorm"></a>

### RMSNorm

```math
r=\sqrt{\frac1D\sum_i x_i^2+\varepsilon},\qquad y_i=\gamma_i x_i/r.
```

$r$ — root mean square; среднее не вычитается. Для $x=[3,4]$, $\gamma=1$: $r=\sqrt{12.5+\varepsilon}\approx\sqrt{12.5}$, $y\approx[0.848528,1.131371]$. Это основная нормировка DeepSeek V3/V3.2; LayerNorm в V3.2 находится в DSA key path.

<a name="op-l2norm"></a>

### L2 нормировка

```math
y=\frac{x}{\sqrt{\sum_i x_i^2+\varepsilon}}.
```

Это вариант GDN в GigaChat. Для $[3,4]$ получаем примерно $[0.6,0.8]$. Здесь **сумма**, в RMSNorm — **среднее**; без $\varepsilon$ и affine между ними множитель $\sqrt D$.

Отдельно существует вариант `F.normalize`: $x/\max(\|x\|_2,\varepsilon)$, где $\|x\|_2=\sqrt{\sum_i x_i^2}$. Они различаются для маленьких векторов: $x=[0.001,0]$, $\varepsilon=10^{-6}$ дают `[0.707107,0]` в GDN и `[1,0]` в clamp-варианте. Нулевой вход при положительном $\varepsilon$ остаётся нулевым.

Спецификации: [PyTorch LayerNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html), [RMSNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.RMSNorm.html).

**C++17.**

```cpp
Vec layernorm(const Vec& x, const Vec& gamma, const Vec& beta, double eps = 1e-5) {
    require(!x.empty() && gamma.size() == x.size() && beta.size() == x.size(),
            "layernorm: incompatible dimensions");
    require(eps > 0.0, "layernorm: eps must be positive");
    const double mean = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
    double variance = 0.0;
    for (double value : x) variance += (value - mean) * (value - mean);
    variance /= x.size();
    const double inv_std = 1.0 / std::sqrt(variance + eps);
    Vec y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        y[i] = gamma[i] * (x[i] - mean) * inv_std + beta[i];
    return y;
}

Vec rmsnorm(const Vec& x, const Vec& gamma, double eps = 1e-6) {
    require(!x.empty() && gamma.size() == x.size(), "rmsnorm: incompatible dimensions");
    require(eps > 0.0, "rmsnorm: eps must be positive");
    const double inv_rms = 1.0 / std::sqrt(dot(x, x) / x.size() + eps);
    Vec y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = gamma[i] * x[i] * inv_rms;
    return y;
}

Vec l2norm_gdn(const Vec& x, double eps = 1e-6) {
    require(eps > 0.0, "l2norm_gdn: eps must be positive");
    const double denominator = std::sqrt(dot(x, x) + eps);
    Vec y = x;
    for (double& value : y) value /= denominator;
    return y;
}
```

```python
def layer_norm(x, weight=None, bias=None, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    centered = x - mean
    # population variance: деление на D, а не D-1.
    variance = (centered * centered).mean(dim=-1, keepdim=True)
    y = centered / torch.sqrt(variance + eps)
    if weight is not None:
        y = y * weight
    return y if bias is None else y + bias

def rms_norm(x, weight=None, eps=1e-6):
    # dim=-1: нормируем признаки каждого токена отдельно.
    # keepdim=True: [..., D] -> [..., 1], чтобы сработал broadcasting.
    rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + eps)
    y = x / rms
    return y if weight is None else y * weight

def l2_norm_gdn(x, eps=1e-6):
    # Именно вариант GigaChat GDN: eps под корнем; НЕ F.normalize.
    return x / torch.sqrt((x * x).sum(dim=-1, keepdim=True) + eps)
```

```python
print("LN:", layer_norm(torch.tensor([1., 3.])))
print("RMS:", rms_norm(torch.tensor([3., 4.])))
print("L2:", l2_norm_gdn(torch.tensor([3., 4.])))
print("L2 small:", l2_norm_gdn(torch.tensor([.001, 0.])))
```

Вывод примера:

```text
LN: tensor([-0.999995,  0.999995])
RMS: tensor([0.848528, 1.131371])
L2: tensor([0.600000, 0.800000])
L2 small: tensor([0.707107, 0.000000])
```

<a name="op-zero-centered"></a>

## 5 Zero centered RMSNorm и GatedNorm

Zero-centered означает параметризацию **масштаба** $\gamma=1+w$: при обучаемом параметре $w=0$ эффективный scale равен 1. Среднее входа при этом не вычитается.

```math
n=(1+w)\odot\frac{x}{\sqrt{D^{-1}\sum_i x_i^2+\varepsilon}}.
```

<a name="block-gatednorm"></a>

### GatedNorm

В GigaChat 3.5 GatedNorm добавляет зависящий от входа поэлементный gate:

```math
g=2\sigma\bigl(W_{down}\mathrm{SiLU}(W_{up}n)\bigr),\qquad y=n\odot g.
```

$W_{up}\in\mathbb R^{R_g\times D}$, $W_{down}\in\mathbb R^{D\times R_g}$; $R_g$ — ширина маленького gate-MLP, в этой конфигурации 16. Gate вычисляется из **нормированного** $n$. Если его logits равны нулю, $2\sigma(0)=1$ и gate не меняет масштаб.

**Пример:** $x=[3,4]$, $w=0$ дают $n=[0.848528,1.131371]$. Выберем маленький bottleneck с $W_{up}=[1,0]$ и $W_{down}=[0,\log(3)/\mathrm{SiLU}(n_0)]^\top\approx[0,1.848928]^\top$, чтобы его logits были $[0,\log3]$. Тогда $g=[1,1.5]$, $y\approx[0.848528,1.697056]$.

В коде ниже точный второй коэффициент строится через `log(3)/SiLU(n[0])`; это учебный выбор весов. Реальные веса обучаются. Источник варианта: [GigaChat implementation](https://huggingface.co/ai-sage/GigaChat3.5-432B-A28B/blob/main/modeling_gigachat3_5.py).

**C++17.**

```cpp
Vec zero_centered_rmsnorm(const Vec& x, const Vec& w, double eps = 1e-6) {
    Vec gamma = w;
    for (double& value : gamma) value += 1.0;
    return rmsnorm(x, gamma, eps);
}
```

```python
def gated_norm(x, zero_weight, gate_up, gate_down, eps=1e-6):
    # zero-centered: эффективная gamma = 1 + zero_weight.
    n = rms_norm(x, 1.0 + zero_weight, eps)
    # A @ B: матричное произведение; .T: транспонирование 2D-матрицы.
    gate_logits = F.silu(n @ gate_up.T) @ gate_down.T
    return n * (2.0 * torch.sigmoid(gate_logits))
```

```python
x = torch.tensor([3., 4.])
n = rms_norm(x)
gate_up = torch.tensor([[1., 0.]])
gate_down = torch.tensor([[0.], [math.log(3) / float(F.silu(n[0]))]])
print("gate_down:", gate_down)
print("GatedNorm:", gated_norm(x, torch.zeros(2), gate_up, gate_down))
```

Вывод примера:

```text
gate_down: tensor([[0.000000],
        [1.848928]])
GatedNorm: tensor([0.848528, 1.697056])
```

<a name="op-softmax"></a>

## 6 Softmax и маска

```math
p_i=\frac{e^{z_i-m}}{\sum_j e^{z_j-m}},\qquad m=\max_j z_j.
```

$z_i$ — logits, $p_i\ge0$ — нормированные веса, $\sum_i p_i=1$. Вычитание одного и того же $m$ не меняет отношение экспонент, но предотвращает переполнение.

**Пример:** `[1,2,3]` и `[1000,1001,1002]` дают один результат `[0.090031,0.244728,0.665241]`. В attention softmax идёт по **позициям keys**, в LM head — по словарю. Запрещённые logits заменяют на $-\infty$: `[0,0,-inf]` даёт `[0.5,0.5,0]`. Полностью запрещённая строка не задаёт распределение; пример явно отклоняет её.

<a name="op-mask"></a>

### Маскирование

Для logits $L_{tj}$ задают $M_{tj}=0$ в доступных позициях и $M_{tj}=-\infty$ в запрещённых, затем считают softmax от $L+M$. В causal attention запрещены будущие позиции $j>t$. DSA дополнительно ограничивает набор выбранными позициями. Для MoE аналогично маскируются эксперты вне выбранных групп перед top-k.

`masked_fill(mask, -torch.inf)` заменяет запрещённые logits. Это отличается от умножения logits на ноль: нулевой logit всё ещё даёт ненулевую экспоненту. В [attention-примере](#block-attention) показана реализация и учёт абсолютных позиций decode.

**C++17.**

```cpp
Vec softmax(const Vec& x) {
    require(!x.empty(), "softmax: empty vector");
    const double m = *std::max_element(x.begin(), x.end());
    require(std::isfinite(m), "softmax: needs at least one finite logit and no +inf");
    Vec p(x.size());
    double denominator = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        p[i] = std::exp(x[i] - m);
        denominator += p[i];
    }
    for (double& value : p) value /= denominator;
    return p;
}
```

```python
def stable_softmax(x):
    maximum = x.amax(dim=-1, keepdim=True)
    if not torch.isfinite(maximum).all():
        raise ValueError("В каждой строке нужен хотя бы один конечный logit")
    numerator = torch.exp(x - maximum)
    return numerator / numerator.sum(dim=-1, keepdim=True)
```

```python
print(stable_softmax(torch.tensor([1000., 1001., 1002.])))
print(stable_softmax(torch.tensor([0., 0., -torch.inf])))
```

Вывод примера:

```text
tensor([0.090031, 0.244728, 0.665241])
tensor([0.500000, 0.500000, 0.000000])
```

<a name="op-topk"></a>

## 7 Top k gather и scatter add

Для вектора $s\in\mathbb R^N$ операция top-k возвращает $k$ наибольших **значений и их индексы**:

```math
I=\mathrm{TopKIndices}(s,k),\qquad v_r=s_{I_r},\quad r=0,\ldots,k-1.
```

Пример: `s=[0.2,0.7,0.1,0.6]`, `k=2` → values `[0.7,0.6]`, indices `[1,3]`. По умолчанию PyTorch сортирует выбранные values по убыванию; порядок индексов при равных значениях не гарантируется. [Спецификация topk](https://docs.pytorch.org/docs/2.14/generated/torch.topk.html).

<a name="op-gather"></a>

### Gather

**Gather:** чтение элементов по индексам, $y_r=x_{I_r}$. 

<a name="op-scatter"></a>

### Scatter-add

**Scatter-add:** накопление результатов обратно, $y_t=\sum_{r:I_r=t}c_r$, где $c_r$ — вклад маршрута $r$, $I_r$ — исходный токен. При повторных IDs нужна сумма; обычное присваивание перезапишет предыдущий вклад.

В DSA выбирают **позиции контекста**, в MoE — **экспертов**. Для MoE один токен может пойти к нескольким экспертам; их вклады затем складываются.

```python
scores = torch.tensor([[.2, .7, .1, .6], [9., 2., 8., 1.]])
values, ids = torch.topk(scores, k=2, dim=-1)
print("values:", values)
print("indices:", ids)
print("gather:", scores.gather(dim=1, index=ids))

token_ids = torch.tensor([0, 0, 1])
contributions = torch.tensor([[1., 0.], [0., 2.], [3., 4.]])
combined = torch.zeros(2, 2)
combined.index_add_(0, token_ids, contributions)
print("scatter-add:", combined)
```

Вывод примера:

```text
values: tensor([[0.700000, 0.600000],
        [9.000000, 8.000000]])
indices: tensor([[1, 3],
        [0, 2]])
gather: tensor([[0.700000, 0.600000],
        [9.000000, 8.000000]])
scatter-add: tensor([[1., 2.],
        [3., 4.]])
```

<a name="op-rope"></a>

## 8 RoPE

RoPE кодирует позицию вращением пар признаков query/key. Для пары $m$ при позиции $p$:

```math
\theta_m=b^{-2m/d_r},\quad \phi_{p,m}=p\theta_m,
```

```math
\begin{bmatrix}x'_{2m}\\x'_{2m+1}\end{bmatrix}=
\begin{bmatrix}\cos\phi&-\sin\phi\\\sin\phi&\cos\phi\end{bmatrix}
\begin{bmatrix}x_{2m}\\x_{2m+1}\end{bmatrix}.
```

$d_r$ — чётное число вращаемых признаков, $b>1$ — base частот, $m=0,\ldots,d_r/2-1$. При обозначении всего преобразования $R_p$ верно $(R_pq)^\top R_jk=q^\top R_{j-p}k$: в dot product появляется относительная позиция.

**Пример:** `[1,2]`, угол $\pi/2$ → `[-2,1]`. Это пример одной матрицы вращения; позиция токена в настоящей модели целая. В проверке ниже дробная позиция выбрана только для получения красивого угла.

В MLA вращают отдельную positional-часть, а content-часть оставляют без RoPE. Код ниже — базовый RoPE с соседними парами; YaRN меняет частоты и scaling, а layout пар должен соответствовать checkpoint. [Исходная статья RoFormer](https://arxiv.org/abs/2104.09864).

**C++17.**

```cpp
std::array<double, 2> rope_pair(double a, double b, double phi) {
    const double c = std::cos(phi), s = std::sin(phi);
    return {a * c - b * s, a * s + b * c};
}
```

```python
def rope(x, positions, base=10000.0):
    """Базовый RoPE без YaRN; x[T,H,d], positions[T], чётное d.

    Вращаем соседние пары (0,1), (2,3), ... . Другие layouts
    требуют перестановки признаков вместе с согласованными весами.
    """
    d = x.shape[-1]
    if d % 2:
        raise ValueError("RoPE dimension must be even")
    inverse_frequency = base ** (-torch.arange(0, d, 2, dtype=x.dtype,
                                              device=x.device) / d)
    # [T,1] * [1,d/2] -> [T,d/2]; None добавляет ось длины 1.
    angle = positions[:, None] * inverse_frequency[None, :]
    cosine = angle.cos()[:, None, :]  # [T,1,d/2], общие для heads
    sine = angle.sin()[:, None, :]
    even = x[..., 0::2]  # ... оставляет все предыдущие оси
    odd = x[..., 1::2]   # start:stop:step, шаг 2
    pairs = torch.stack((even * cosine - odd * sine,
                         even * sine + odd * cosine), dim=-1)
    return pairs.flatten(-2)  # [T,H,d/2,2] -> [T,H,d]
```

```python
x = torch.tensor([[[1., 2.]]])  # [T=1,H=1,d=2]
print(rope(x, torch.tensor([math.pi / 2])))
```

Вывод примера:

```text
tensor([[[-2.000000,  1.000000]]])
```

<a name="block-swiglu"></a>

## 9 FFN и SwiGLU

FFN независимо преобразует каждый токен. Обычная двухслойная форма:

```math
\mathrm{FFN}(x)=W_2\varphi(W_1x+b_1)+b_2.
```

$W_1\in\mathbb R^{M\times D}$ расширяет признаки, $W_2\in\mathbb R^{D\times M}$ возвращает ширину $D$, $\varphi$ — поэлементная активация.

В выбранных моделях dense FFN и отдельные MoE experts используют **SwiGLU**:

```math
a=W_gx,\quad u=W_ux,\quad h=\mathrm{SiLU}(a)\odot u,\quad y=W_dh.
```

$W_g,W_u\in\mathbb R^{M\times D}$, $W_d\in\mathbb R^{D\times M}$. Здесь три Linear. Gate SiLU не ограничен диапазоном `[0,1]`; это не sigmoid gate.

**Пример:** $x=[1,2]$, $W_g=W_u=I_2$, $W_d=\begin{bmatrix}1&1\\1&-1\end{bmatrix}$. Получаем $a=u=[1,2]$, $h=[0.731059,3.523188]$, $y=[4.254247,-2.792130]$. $I_2$ — единичная матрица 2×2.

<a name="op-clamp"></a>

### Ограничение активаций

Для GigaChat 3.5 используется ограничение $L=10$ **до** SiLU и произведения:

```math
a\leftarrow\min(a,L),\qquad u\leftarrow\mathrm{clip}(u,-L,L).
```

Gate ограничен только сверху, up — с обеих сторон. Для $a=[12,-12],u=[15,-15]$ получается $[10,-12]$ и $[10,-10]$. В нашей функции `limit=None` отключает ограничение; это учебный API. [Исходник GigaChat](https://huggingface.co/ai-sage/GigaChat3.5-432B-A28B/blob/main/modeling_gigachat3_5.py).

```python
def swiglu(x, w_gate, w_up, w_down, limit=None):
    """x[...,D], gate/up[M,D], down[D,M]. Без bias.

    limit=None: обычный SwiGLU; limit=10: clipping GigaChat 3.5.
    """
    gate = x @ w_gate.T
    up = x @ w_up.T
    if limit is not None:
        gate = gate.clamp(max=limit)  # только верхняя граница
        up = up.clamp(min=-limit, max=limit)
    hidden = F.silu(gate) * up        # * — поэлементное умножение
    return hidden @ w_down.T
```

```python
x = torch.tensor([1., 2.])
wg = torch.eye(2)
wd = torch.tensor([[1., 1.], [1., -1.]])
print("SwiGLU:", swiglu(x, wg, wg, wd))
print("clipped gate:", torch.tensor([12., -12.]).clamp(max=10))
print("clipped up:", torch.tensor([15., -15.]).clamp(-10, 10))
```

Вывод примера:

```text
SwiGLU: tensor([ 4.254247, -2.792130])
clipped gate: tensor([ 10., -12.])
clipped up: tensor([ 10., -10.])
```

<a name="block-attention"></a>

## 10 Causal self attention и multi head

Для одного head:

```math
Q=XW_Q^\top,\quad K=XW_K^\top,\quad V_a=XW_V^\top,
```

```math
L_{tj}=\frac{q_t^\top k_j}{\sqrt{d_k}}+M_{tj},\qquad
A_{tj}=\mathrm{softmax}_j(L_{tj}),\qquad o_t=\sum_j A_{tj}v_j.
```

$X\in\mathbb R^{T\times D}$ хранит токены строками; $W_Q,W_K\in\mathbb R^{d_k\times D}$, $W_V\in\mathbb R^{d_v\times D}$. $Q,K$ имеют ширину $d_k$, $V_a$ — $d_v$; обозначение $V_a$ отличает матрицу values от размера словаря $V$. В causal self-attention $M_{tj}=0$ при $j\le t$ и $-\infty$ при $j>t$.

$q$ описывает текущий запрос, $k$ — признак для поиска, $v$ — содержимое, которое смешиваем. Делитель $\sqrt{d_k}$ ограничивает рост масштаба dot product с шириной head.

**Пример:** $Q=K=I_2$, $V_a=[[10,0],[0,20]]$. Первый токен видит только себя, поэтому $o_0=[10,0]$. Для второго logits `[0,1/√2]`, веса `[0.330238,0.669762]`, выход `[3.302385,13.395231]`.

**Multi-head:** каждый из $H$ heads считает свой выход $o_{t,h}$, затем $y_t=W_O\mathrm{Concat}_h(o_{t,h})$, где $W_O\in\mathbb R^{D\times Hd_v}$. В GQA несколько query-heads разделяют один набор K/V; в MQA один набор K/V общий для всех heads. MLA ниже использует сжатое представление KV.

**Decode:** одна новая query может видеть весь прежний cache. Поэтому при `T=1,S>1` causal mask должна учитывать абсолютную позицию query; обычная локальная треугольная маска `[1,S]` может ошибочно оставить только первый key.

[Исходная формула attention](https://arxiv.org/abs/1706.03762).

```python
def causal_attention(q, k, v, query_positions=None, key_positions=None):
    """Один head: q[T,dk], k[S,dk], v[S,dv]. Возвращает y и A.

    Для decode с KV-cache передавайте АБСОЛЮТНЫЕ query_positions.
    Без positions подразумевается квадратный prefill T=S.
    """
    if query_positions is None:
        if q.shape[0] != k.shape[0]:
            raise ValueError("Decode требует абсолютные query_positions")
        query_positions = torch.arange(q.shape[0], device=q.device)
    if key_positions is None:
        key_positions = torch.arange(k.shape[0], device=k.device)
    logits = (q @ k.T) / math.sqrt(q.shape[-1])  # [T,S]
    allowed = key_positions[None, :] <= query_positions[:, None]
    # masked_fill заменяет запрещённые logits; ~ инвертирует bool mask.
    logits = logits.masked_fill(~allowed, -torch.inf)
    probabilities = stable_softmax(logits)
    return probabilities @ v, probabilities

def multihead_attention(q, k, v, query_positions=None, key_positions=None):
    """q[T,H,dk], k[S,H,dk], v[S,H,dv]. Возвращает [T,H,dv].

    Простое перечисление heads, чтобы не прятать математику в einsum.
    Выходы heads затем concatenate и пропускают через W_O.
    """
    outputs = []
    for h in range(q.shape[1]):
        out, _ = causal_attention(q[:, h], k[:, h], v[:, h],
                                  query_positions, key_positions)
        outputs.append(out)
    return torch.stack(outputs, dim=1)
```

```python
q = torch.eye(2)
v = torch.tensor([[10., 0.], [0., 20.]])
out, weights = causal_attention(q, q, v)
print("weights:", weights)
print("output:", out)
decode, _ = causal_attention(q[-1:], q, v, query_positions=torch.tensor([1]))
print("decode last token:", decode)

# Два heads с одинаковыми Q/K, но второй читает values вдвое больше.
qh = torch.stack((q, q), dim=1)
vh = torch.stack((v, 2 * v), dim=1)
heads = multihead_attention(qh, qh, vh)
print("multi-head [T,H,dv]:", heads)
print("concatenated [T,H*dv]:", heads.flatten(1))
```

Вывод примера:

```text
weights: tensor([[1.000000, 0.000000],
        [0.330238, 0.669762]])
output: tensor([[10.000000,  0.000000],
        [ 3.302385, 13.395231]])
decode last token: tensor([[ 3.302385, 13.395231]])
multi-head [T,H,dv]: tensor([[[10.000000,  0.000000],
         [20.000000,  0.000000]],

        [[ 3.302385, 13.395231],
         [ 6.604769, 26.790462]]])
concatenated [T,H*dv]: tensor([[10.000000,  0.000000, 20.000000,  0.000000],
        [ 3.302385, 13.395231,  6.604769, 26.790462]])
```

<a name="block-mla"></a>

## 11 MLA и сжатый KV cache

MLA — attention с общим низкоразмерным представлением K/V для всех heads. Определим $r_q$ — query rank, $r$ — KV rank, $d_n$ — ширину content Q/K, $d_r$ — ширину RoPE, $d_v$ — ширину values.

```math
c_t^Q=N_Q(W_{DQ}x_t),\quad c_t=N_{KV}(W_{DKV}x_t),
```

```math
q_{t,h}^C=W_{UQ,h}c_t^Q,\quad q_{t,h}^R=R_tW_{QR,h}c_t^Q,
```

```math
k_{j,h}^C=W_{UK,h}c_j,\quad v_{j,h}=W_{UV,h}c_j,\quad k_j^R=R_jW_{KR}x_j.
```

Здесь $N_Q,N_{KV}$ — нормы; в базовом DeepSeek RMSNorm. Размеры матриц:

| Матрица | Размер |
|---|---|
| $W_{DQ},W_{DKV}$ | $r_q\times D$, $r\times D$ |
| $W_{UQ,h},W_{QR,h}$ | $d_n\times r_q$, $d_r\times r_q$ |
| $W_{UK,h},W_{UV,h}$ | $d_n\times r$, $d_v\times r$ |
| $W_{KR}$ | $d_r\times D$ |

$h$ — head, $C$ означает content, $R$ — RoPE; positional key $k_j^R$ общий для heads. $R_t$ — вращение из раздела 8.

```math
L_{tj,h}=\frac{(q_{t,h}^C)^\top k_{j,h}^C+(q_{t,h}^R)^\top k_j^R}{\sqrt{d_n+d_r}}+M_{tj},
```

```math
p_{tj,h}=\mathrm{softmax}_j L_{tj,h},\qquad o_{t,h}=\sum_j p_{tj,h}v_{j,h}.
```

Для decode можно перенести расширяющую K-матрицу на query:

```math
\tilde q_{t,h}=W_{UK,h}^\top q_{t,h}^C,\qquad
(q_{t,h}^C)^\top W_{UK,h}c_j=\tilde q_{t,h}^\top c_j.
```

А проекцию V выполнить после суммирования:

```math
z_{t,h}=\sum_j p_{tj,h}c_j,\qquad o_{t,h}=W_{UV,h}z_{t,h}.
```

$\tilde q,z\in\mathbb R^r$. Cache хранит **$c_j$ и $k_j^R$**, без развёрнутых K/V каждого head. У DeepSeek V3.2 $r=512,d_r=64$: 576 скаляров на токен и слой только для MLA; DSA добавляет свой cache. Компрессия Q не означает, что Q нужно кэшировать. [MLA paper](https://arxiv.org/abs/2405.04434), [DeepSeek configuration](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json).

**Пример одного head:** $C=[[1,0],[0,1],[1,1]]$, $W_K=[[1,0],[0,2]]$, $W_V=[[1,0],[1,1]]$ в соглашении `[out,in]`. Тогда $K=CW_K^\top=[[1,0],[0,2],[1,2]]$, $V_a=CW_V^\top=[[1,1],[0,1],[1,2]]$. Пусть $q^C=[1,1],q^R=[1,0]$, $K^R=[[1,0],[0,1],[-1,0]]$. Content scores `[1,2,3]`, RoPE scores `[1,0,-1]`; после деления на $\sqrt4$ logits `[1,1,1]`. Веса равны $1/3$, выход $[2/3,4/3]$.

Учебный core получает уже спроецированные/нормированные входы. Реальный GigaChat использует GatedNorm в latent-ветвях, дополнительные множители $\sqrt{D/r_q}$ и $\sqrt{D/r}$, а на склеенном выходе heads — $\sigma(W_{gate}x)$ перед $W_O$, **без множителя 2**. YaRN также меняет базовый scale. Это дополнительные элементы реального блока, а не изменение тождества absorption.

```python
def mla_decode(q_content, q_rope, cache, rope_keys, w_key, w_value,
               scale=None):
    """MLA core одного decode-токена без разворачивания истории K/V.

    q_content[H,dn], q_rope[H,dr]; cache[S,r], rope_keys[S,dr];
    w_key[H,dn,r], w_value[H,dv,r]. Проекции, нормы и RoPE уже
    применены к входам. cache хранит нормированные latent KV.
    YaRN/FP8/output gate/W_O в этом учебном core отсутствуют.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q_content.shape[-1] + q_rope.shape[-1])
    # einsum: суммируем по d, сохраняем h и c -> [H,r].
    absorbed_query = torch.einsum("hdc,hd->hc", w_key, q_content)
    logits = (absorbed_query @ cache.T + q_rope @ rope_keys.T) * scale
    probabilities = stable_softmax(logits)  # [H,S]
    latent_output = probabilities @ cache  # [H,r]
    # Для каждого head: W_value[h] @ latent_output[h].
    output = torch.einsum("hvc,hc->hv", w_value, latent_output)
    return output, probabilities
```

```python
cache = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
wk = torch.tensor([[[1., 0.], [0., 2.]]])  # [H,dn,r]
wv = torch.tensor([[[1., 0.], [1., 1.]]])  # [H,dv,r]
qc = torch.tensor([[1., 1.]])
qr = torch.tensor([[1., 0.]])
kr = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
out, weights = mla_decode(qc, qr, cache, kr, wk, wv)
print("MLA weights:", weights)
print("MLA absorbed output:", out)
expanded_k = cache @ wk[0].T
expanded_v = cache @ wv[0].T
explicit_weights = stable_softmax((qc @ expanded_k.T + qr @ kr.T) / 2)
explicit_out = explicit_weights @ expanded_v
print("MLA explicit output:", explicit_out)
torch.testing.assert_close(out, explicit_out)
```

Вывод примера:

```text
MLA weights: tensor([[0.333333, 0.333333, 0.333333]])
MLA absorbed output: tensor([[0.666667, 1.333333]])
MLA explicit output: tensor([[0.666667, 1.333333]])
```

<a name="block-dsa"></a>

## 12 DSA и lightning indexer

DSA сначала выбирает полезные позиции, затем выполняет MLA по ним. У индексатора свои Q/K и веса heads:

```math
I_{tj}=\frac{1}{\sqrt{H_I d_I}}\sum_{h=1}^{H_I}a_{t,h}\mathrm{ReLU}\left((q^I_{t,h})^\top k^I_j\right).
```

$H_I,d_I$ — количество heads индексатора и их ширина; $q^I_{t,h},k^I_j\in\mathbb R^{d_I}$; $a_{t,h}$ — **сырые** выходы обученного Linear по $x_t$, без sigmoid/softmax. $I_{tj}$ — ranking score для позиции $j$.

Query индексатора получается из $c_t^Q$ MLA; key — из $\mathrm{LayerNorm}(W_{IK}x_j)$. Затем применяют частичный RoPE. В реализации перед FP8 есть общий нормированный Hadamard для Q/K. Его математика и маленький пример приведены в приложении.

```math
J_t=\mathrm{TopKIndices}_{j\le t}(I_{tj},K_s).
```

$K_s$ — число выбираемых токенов, в V3.2 до 2048. Для короткого контекста выбираются все доступные позиции. Набор $J_t$ общий для MLA-heads. После выбора считаются **отдельные MLA logits** и softmax только по $j\in J_t$; index scores не становятся attention weights. [DSA paper](https://arxiv.org/html/2512.02556v1), [official indexer](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py).

**Пример:** $Q_I=[[1,0],[0,1]]$, $K_I=[[1,0],[0,3],[-1,1]]$, $a=[2,1]$. После ReLU и scale $1/2$ получаем index scores `[1,1.5,0.5]`. Top-2 выбирает позиции `[1,0]`. Пусть отдельные MLA logits по трём позициям равны `[2,1,0]`, values `[[1,0],[0,2],[5,5]]`. В выбранном порядке logits `[1,2]`, softmax `[0.268941,0.731059]`, результат `[0.731059,0.537883]`.

Ниже float-ядро индексатора с уже подготовленными causal Q/K. Полный индексатор всё ещё просматривает историю. Поэтому уменьшение основной attention-части до $O(K_s)$ на decode не означает постоянную стоимость всей DSA по длине контекста.

```python
def dsa_select(q_index, k_index, head_weights, count):
    """DSA indexer core: Q[Hi,di], K[S,di], сырые веса heads[Hi].

    Все переданные позиции уже causal. Проекции/LayerNorm/частичный
    RoPE выполнены до этого core; FP8 здесь отключён.
    """
    heads, width = q_index.shape
    positive_dot = torch.relu(q_index @ k_index.T)  # [Hi,S]
    # [Hi,1] умножается на каждый столбец, затем сумма по heads.
    scores = (head_weights[:, None] * positive_dot).sum(dim=0)
    scores = scores / math.sqrt(heads * width)
    _, ids = torch.topk(scores, k=min(count, k_index.shape[0]))
    return ids, scores
```

```python
qi = torch.tensor([[1., 0.], [0., 1.]])
ki = torch.tensor([[1., 0.], [0., 3.], [-1., 1.]])
ids, index_scores = dsa_select(qi, ki, torch.tensor([2., 1.]), count=2)
main_logits = torch.tensor([2., 1., 0.])
values = torch.tensor([[1., 0.], [0., 2.], [5., 5.]])
attention_weights = stable_softmax(main_logits[ids])
print("index scores:", index_scores)
print("selected positions:", ids)
print("separate attention weights:", attention_weights)
print("output:", attention_weights @ values[ids])
```

Вывод примера:

```text
index scores: tensor([1.000000, 1.500000, 0.500000])
selected positions: tensor([1, 0])
separate attention weights: tensor([0.268941, 0.731059])
output: tensor([0.731059, 0.537883])
```

<a name="block-moe"></a>

## 13 MoE маршрутизация и эксперты

MoE заменяет один FFN набором FFN-экспертов. Для токена $x\in\mathbb R^D$:

```math
s=\sigma(W_rx),\qquad c_e=s_e+b_e^{corr},\qquad W_r\in\mathbb R^{E\times D}.
```

$E$ — число routed experts, $s_e$ — affinity эксперта, $b_e^{corr}$ — balancing correction, $c_e$ — score только для выбора. Router Linear и balancing correction — разные вещи.

В DeepSeek сначала выбирают группы: score группы — сумма двух лучших $c_e$ внутри неё; берут 4 из 8 групп. Затем top-8 экспертов из выбранных групп. Для GigaChat 3.5 этот групповой фильтр отключается конфигурацией одной группы.

Пусть $\mathcal E$ — выбранные experts. Их веса:

```math
g_e=\rho\frac{s_e}{\sum_{i\in\mathcal E}s_i},\quad e\in\mathcal E,\qquad
y=F_{shared}(x)+\sum_{e\in\mathcal E}g_eF_e(x).
```

$F_e$ — SwiGLU FFN выбранного эксперта; $F_{shared}$ — всегда работающий shared FFN; $\rho=2.5$ в этих checkpoints. **Bias не входит в веса смешивания.** Их сумма равна $\rho$, а не обязательно 1. Residual прибавляется снаружи; здесь его нет. [DeepSeekMoE описание](https://arxiv.org/html/2412.19437v2).

**Пример без ограничения групп:** $s=[.8,.6,.4,.2]$, $b^{corr}=[-.5,0,.3,0]$. Corrected scores `[.3,.6,.7,.2]`, IDs `[2,1]`. При $\rho=2.5$ веса `[1,1.5]`. Для $F_2(x)=[0,3]$, $F_1(x)=[2,0]$, shared `[1,1]` получаем $[4,4]$.

Даны входные affinities, но получить соответствующие router logits можно как $\log(s/(1-s))$. Полная функция ниже вычисляет SwiGLU только выбранных experts и shared branch. Python loop здесь объясняет математику; production группирует токены по экспертам и использует grouped GEMM/dispatch/combine.

```python
def moe_route(affinities, correction, count, scale=2.5,
              num_groups=1, kept_groups=1):
    """Один токен: affinities=sigmoid(router logits), оба входа [E].

    DeepSeek V3.2: E=256, count=8, groups=8, kept_groups=4.
    GigaChat 3.5: groups=kept_groups=1, групповой фильтр выключен.
    """
    experts = affinities.numel()
    if num_groups < 1 or experts % num_groups:
        raise ValueError("Число экспертов должно делиться на число групп")
    if not 1 <= kept_groups <= num_groups:
        raise ValueError("Неверное число выбираемых групп")
    if num_groups > 1 and experts // num_groups < 2:
        raise ValueError("Для group top-2 нужно не менее двух экспертов в группе")
    if not 1 <= count <= kept_groups * (experts // num_groups):
        raise ValueError("top-k больше числа доступных экспертов")
    corrected = affinities + correction
    if num_groups > 1:
        # reshape только перегруппировывает элементы; E кратно groups.
        grouped = corrected.reshape(num_groups, -1)
        group_scores = grouped.topk(k=2, dim=-1).values.sum(dim=-1)
        group_ids = group_scores.topk(k=kept_groups).indices
        group_mask = torch.zeros(num_groups, dtype=torch.bool,
                                 device=affinities.device)
        group_mask[group_ids] = True
        allowed = group_mask[:, None].expand_as(grouped).reshape(-1)
        corrected = corrected.masked_fill(~allowed, -torch.inf)
    ids = corrected.topk(k=count).indices
    # Индексы получили по corrected, веса — по исходным affinities!
    weights = affinities[ids]
    weights = scale * weights / (weights.sum() + 1e-20)
    return ids, weights

def moe_forward(x, router, correction, w_gate, w_up, w_down,
                shared_weights, count, scale=2.5,
                num_groups=1, kept_groups=1, limit=None):
    """x[D], router[E,D], gate/up[E,M,D], down[E,D,M].

    shared_weights — tuple (W_gate, W_up, W_down) одного shared FFN.
    Выполняются только выбранные routed experts и shared expert.
    """
    scores = torch.sigmoid(router @ x)
    ids, weights = moe_route(scores, correction, count, scale,
                             num_groups, kept_groups)
    # *tuple распаковывает три матрицы в три аргумента функции.
    output = swiglu(x, *shared_weights, limit=limit)
    for slot in range(ids.numel()):
        expert = int(ids[slot])  # перевод scalar tensor в Python int
        value = swiglu(x, w_gate[expert], w_up[expert],
                       w_down[expert], limit)
        output = output + weights[slot] * value
    return output, ids, weights
```

```python
affinities = torch.tensor([.8, .6, .4, .2])
correction = torch.tensor([-.5, 0., .3, 0.])
ids, weights = moe_route(affinities, correction, count=2)
expert_outputs = torch.tensor([[99., 99.], [2., 0.], [0., 3.], [99., 99.]])
shared = torch.tensor([1., 1.])
mixed = shared + (weights[:, None] * expert_outputs[ids]).sum(dim=0)
print("expert ids:", ids)
print("mix weights:", weights)
print("output:", mixed)

# Полный маленький MoE с вычислением экспертов из матриц.
x = torch.tensor([1., 0.])
router = torch.stack((torch.log(affinities / (1 - affinities)), torch.zeros(4)), dim=1)
gate = torch.eye(2).repeat(4, 1, 1)
up = torch.eye(2).repeat(4, 1, 1)
down = torch.zeros(4, 2, 2)
down[:, :, 0] = expert_outputs / F.silu(torch.tensor(1.))
shared_down = torch.tensor([[1., 0.], [1., 0.]]) / F.silu(torch.tensor(1.))
full, _, _ = moe_forward(x, router, correction, gate, up, down,
                         (torch.eye(2), torch.eye(2), shared_down), count=2)
print("full SwiGLU experts output:", full)
torch.testing.assert_close(full, mixed)
```

Вывод примера:

```text
expert ids: tensor([2, 1])
mix weights: tensor([1.000000, 1.500000])
output: tensor([4., 4.])
full SwiGLU experts output: tensor([4.000000, 4.000000])
```

<a name="op-conv"></a>

## 14 Причинная depthwise свёртка

Это короткое смешивание **соседних токенов внутри каждого канала**:

```math
Y_{t,c}=\sum_{r=0}^{R-1}w_{c,r}X_{t-(R-1)+r,c},\qquad X_{j,c}=0\text{ при }j<0.
```

$X,Y\in\mathbb R^{T\times C}$, $C$ — число каналов, $c$ — индекс канала, $R$ — длина kernel, $w\in\mathbb R^{C\times R}$. `Depthwise` означает независимый фильтр для каждого канала. `Causal` означает отсутствие будущих токенов. В этой записи последний вес относится к текущему токену; порядок совпадает с `torch Conv1d`.

**Пример одного канала:** $X=[1,2,4]$, веса `[1,2]` — прошлый и текущий отсчёт. Выход `[2,5,10]`, потому что $0+2\cdot1=2$, $1+2\cdot2=5$, $2+2\cdot4=10$.

В GigaChat GDN kernel length равен 4. Через conv и затем SiLU проходят Q/K/V. Проекции gates $z,a,b$ обходят эту свёртку.

**C++17.**

```cpp
Mat causal_depthwise_conv(const Mat& x, const Mat& kernel) {
    require(!x.empty() && !x[0].empty(), "conv: empty input");
    const std::size_t channels = x[0].size();
    require(kernel.size() == channels && !kernel[0].empty(), "conv: incompatible kernel");
    const std::size_t width = kernel[0].size();
    for (const Vec& row : x) require(row.size() == channels, "conv: ragged input");
    for (const Vec& row : kernel) require(row.size() == width, "conv: ragged kernel");
    Mat y(x.size(), Vec(channels, 0.0));
    for (std::size_t t = 0; t < x.size(); ++t)
        for (std::size_t c = 0; c < channels; ++c)
            for (std::size_t r = 0; r < width; ++r) {
                const long source = static_cast<long>(t) - static_cast<long>(width - 1 - r);
                if (source >= 0) y[t][c] += kernel[c][r] * x[source][c];
            }
    return y;
}
```

```python
def causal_depthwise_conv(x, weights):
    """x[T,C], weights[C,R], без bias. Порядок весов: old -> current.

    Каналы не смешиваются. Отрицательные позиции — нули.
    """
    steps, channels = x.shape
    kernel = weights.shape[-1]
    out = torch.zeros_like(x)
    for t in range(steps):
        for r in range(kernel):
            source = t - (kernel - 1) + r
            if source >= 0:
                out[t] += weights[:, r] * x[source]
    return out
```

```python
x = torch.tensor([[1.], [2.], [4.]])
weights = torch.tensor([[1., 2.]])
print("convolution:", causal_depthwise_conv(x, weights).flatten())
print("after SiLU:", F.silu(causal_depthwise_conv(x, weights)).flatten())
```

Вывод примера:

```text
convolution: tensor([ 2.,  5., 10.])
after SiLU: tensor([1.761594, 4.966536, 9.999546])
```

<a name="block-gdn"></a>

## 15 Gated DeltaNet

GDN поддерживает матрицу памяти фиксированного размера вместо хранения всех прошлых K/V. Для одного head:

| Обозначение | Значение |
|---|---|
| $q_t,k_t\in\mathbb R^{d_k}$ | Векторы чтения и адреса записи |
| $v_t,o_t\in\mathbb R^{d_v}$ | Записываемое значение и выход |
| $S_t\in\mathbb R^{d_k\times d_v}$ | Состояние после обработки токена $t$ |
| $\alpha_t\in(0,1)$ | Сохранение прежней памяти |
| $\beta_t\in(0,1)$ | Сила корректирующей записи |

### Подготовка

После проекций и conv+SiLU:

```math
q_t=\frac{q_t^{raw}}{\sqrt{\sum_i(q_{t,i}^{raw})^2+\varepsilon}\sqrt{d_k}},\qquad
k_t=\frac{k_t^{raw}}{\sqrt{\sum_i(k_{t,i}^{raw})^2+\varepsilon}}.
```

Коэффициенты конкретного head:

```math
\beta_t=\sigma(b_t),\quad g_t=-e^{A_{log}}\mathrm{softplus}(a_t+d_{bias}),\quad \alpha_t=e^{g_t}.
```

$a_t,b_t$ — скаляры из отдельных проекций текущего входа; $A_{log},d_{bias}$ — обученные параметры head; $g_t$ — логарифм коэффициента сохранения. При $a_t=b_t=A_{log}=d_{bias}=0$ получаем $\alpha_t=\beta_t=0.5$.

<a name="op-outer"></a>

### Обновление памяти и outer product

```math
\bar S_t=\alpha_tS_{t-1},\quad \hat v_t=\bar S_t^\top k_t,
```

```math
\delta_t=\beta_t(v_t-\hat v_t),\quad S_t=\bar S_t+k_t\delta_t^\top,\quad o_t=S_t^\top q_t.
```

$\hat v_t$ — предсказание значения уже **после затухания**; $\delta_t$ — корректировка. Матрица $k_t\delta_t^\top$ — outer product, её элемент $(i,j)$ равен $k_{t,i}\delta_{t,j}$. Так memory не просто суммирует новые values: она исправляет ошибку текущего предсказания по этому ключу. Чтение использует уже обновлённое состояние.

Эквивалентная запись, где $I_{d_k}$ — единичная матрица:

```math
S_t=\alpha_t(I_{d_k}-\beta_tk_tk_t^\top)S_{t-1}+\beta_tk_tv_t^\top.
```

### Два шага на матрице 2 на 2

Пусть $S_0=0$, $q^{raw}=k^{raw}=[1,0]$, $\alpha=\beta=.5$. Для ручного примера $\varepsilon=0$, поэтому $q=[1/\sqrt2,0]$, $k=[1,0]$.

**Шаг 1.** При $v_1=[2,4]$ имеем $\delta_1=[1,2]$,

```math
S_1=\begin{bmatrix}1&2\\0&0\end{bmatrix},\qquad o_1=[0.707107,1.414214].
```

**Шаг 2.** При $v_2=[4,0]$ имеем $\bar S_2=[[.5,1],[0,0]]$, $\hat v_2=[.5,1]$, $\delta_2=[1.75,-.5]$,

```math
S_2=\begin{bmatrix}2.25&.5\\0&0\end{bmatrix},\qquad o_2=[1.590990,0.353553].
```

### Выходная обвязка

В GigaChat каждый head затем получает zero-centered RMSNorm и gate:

```math
u_t=\mathrm{RMSNorm}_{\gamma=1+w}(o_t)\odot 2\sigma(z_t),\qquad
y_t=W_O\mathrm{Concat}_h(u_{t,h}).
```

$z_t\in\mathbb R^{d_v}$ — отдельная проекция входа. $u_{t,h}$ — нормированный выход head после gate, $h$ — индекс head, $W_O\in\mathbb R^{D\times(H_vd_v)}$, $H_v$ — число value heads. Это другой gate, чем bottleneck GatedNorm из раздела 5. При $z=0$ множитель равен 1, но сама RMS-нормировка продолжает действовать. Например, $o=[3,4]$, $w=0,z=[0,\log3]$ дают `[0.848528,1.697056]`.

Реальная конфигурация GigaChat: 32 Q/K heads и 64 value heads, $d_k=d_v=128$; каждый Q/K head повторяется для двух value heads. Наш core показывает один head. Последовательный цикл удобен для decode; prefill/training обычно считают chunkwise эквивалентным алгоритмом. [Gated DeltaNet paper](https://arxiv.org/html/2412.06464v3), [GigaChat source](https://huggingface.co/ai-sage/GigaChat3.5-432B-A28B/blob/main/modeling_gigachat3_5.py).

```python
def gdn_coefficients(a, b, a_log, dt_bias):
    # a,b[...,H]; a_log и dt_bias[H] — обученные параметры heads.
    log_alpha = -torch.exp(a_log) * F.softplus(a + dt_bias)
    return torch.exp(log_alpha), torch.sigmoid(b)

def gdn_core(q, k, v, alpha, beta, initial_state=None, eps=1e-6):
    """Один head: q,k[T,dk], v[T,dv], alpha,beta[T].

    Входы q/k/v уже прошли проекции и conv+SiLU. Внутри только
    L2-нормировка q/k и gated delta recurrence. Выходы до RMS/gate/W_O.
    """
    steps, dk = q.shape
    dv = v.shape[-1]
    q = l2_norm_gdn(q, eps) / math.sqrt(dk)
    k = l2_norm_gdn(k, eps)
    if initial_state is None:
        state = torch.zeros(dk, dv, dtype=q.dtype, device=q.device)
    else:
        state = initial_state.clone()
    outputs = []
    for t in range(steps):
        decayed = alpha[t] * state
        predicted_value = k[t] @ decayed  # [dk] @ [dk,dv] -> [dv]
        delta = beta[t] * (v[t] - predicted_value)
        # outer(k,delta)[i,j]=k[i]*delta[j]: rank-one поправка.
        state = decayed + torch.outer(k[t], delta)
        outputs.append(q[t] @ state)
    return torch.stack(outputs), state

def gdn_output_gate(core_output, z, zero_weight, eps=1e-6):
    # Последняя ось — d_v одного head. z — отдельная проекция входа.
    return rms_norm(core_output, 1.0 + zero_weight, eps) * (
        2.0 * torch.sigmoid(z))
```

```python
zero = torch.zeros(2)
alpha, beta = gdn_coefficients(zero, zero, zero, zero)
print("alpha, beta:", alpha, beta)
qk = torch.tensor([[1., 0.], [1., 0.]])
values = torch.tensor([[2., 4.], [4., 0.]])
out, state = gdn_core(qk, qk, values, alpha, beta, eps=0.0)
print("core output:", out)
print("final state:", state)
print("output norm+gate example:", gdn_output_gate(
    torch.tensor([3., 4.]), torch.tensor([0., math.log(3)]), torch.zeros(2)))
```

Вывод примера:

```text
alpha, beta: tensor([0.500000, 0.500000]) tensor([0.500000, 0.500000])
core output: tensor([[0.707107, 1.414214],
        [1.590990, 0.353553]])
final state: tensor([[2.250000, 0.500000],
        [0.000000, 0.000000]])
output norm+gate example: tensor([0.848528, 1.697056])
```

<a name="op-residual"></a>

## 16 Residual и порядок блока

Residual складывает два вектора одинаковой формы:

```math
y=x+f(x).
```

Здесь $f$ — вычислительная ветвь, а $x$ проходит напрямую. Пример: `[1,2]+[0.1,0.2]=[1.1,2.2]`.

Схематично блок DeepSeek с pre-norm:

```math
h=x+\mathrm{MLA}(N_1(x)),\qquad y=h+\mathrm{FFN/MoE}(N_2(h)).
```

$N_1,N_2$ — разные обученные RMSNorm. В V3.2 MLA включает отбор позиций через DSA. На dense-слоях используется FFN, на остальных MoE.

В GigaChat 3.5 `pre_post` добавляет нормы к выходам ветвей **перед residual add**:

```math
h=x+GN_2\bigl(\mathrm{MLA/GDN}(GN_1(x))\bigr),
```

```math
y=h+GN_4\bigl(\mathrm{FFN/MoE}(GN_3(h))\bigr).
```

$GN_1,\ldots,GN_4$ — отдельные GatedNorm с собственными весами. Это не формула $GN(x+f(x))$.

Проверка всего пути: `IDs → embedding → повторение attention/GDN + FFN/MoE residual-блоков → final norm → LM head → logits`. MoE находится в FFN-части и не заменяет LM head.

**C++17.**

```cpp
Vec residual_add(const Vec& x, const Vec& branch) {
    require(x.size() == branch.size(), "residual_add: incompatible dimensions");
    Vec y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = x[i] + branch[i];
    return y;
}
```

```python
x = torch.tensor([1., 2.])
branch = torch.tensor([.1, .2])
print(x + branch)
```

Вывод примера:

```text
tensor([1.100000, 2.200000])
```

<a name="precision"></a>

## Приложение A Hadamard и FP8

Эти операции присутствуют в реализации DSA и полезны при чтении профиля.

<a name="op-hadamard"></a>

### Нормированный Hadamard

```math
H_4=\frac12\begin{bmatrix}1&1&1&1\\1&-1&1&-1\\1&1&-1&-1\\1&-1&-1&1\end{bmatrix},\qquad y=H_4x.
```

$H_4^\top H_4=I_4$, поэтому $(H_4q)^\top(H_4k)=q^\top k$. Это смена координат, сохраняющая dot product в точной арифметике. После округления в FP8 равенство уже не обязано быть точным. Пример: `[1,2,3,4]` → `[5,-1,-2,0]`. В модели ширина и конкретное быстрое ядро другие.

<a name="op-fp8"></a>

### Квантизация одного FP8 блока

```math
q=\mathrm{cast}_{FP8}\bigl(\mathrm{clip}(x/s,-F_{max},F_{max})\bigr),\qquad \hat x=s\mathrm{float}(q).
```

$s>0$ — масштаб блока, $F_{max}$ — максимальное конечное число формата, $q$ — FP8-представление, $\hat x$ — восстановленное значение. При E4M3FN $F_{max}=448$. FP8 имеет показатель степени: это не INT8 с равномерным шагом. Например, при $s=1$: `[1.1,2.1,-1.1]` → `[1.125,2,-1.125]`.

В примере scale задан вручную; он не воспроизводит выбор block scales, UE8M0 и округления всех реальных ядер. Источники: [DeepSeek FP8/index kernel](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/kernel.py), [DeepSeek report FP8](https://arxiv.org/html/2412.19437v2).

```python
def hadamard4(x):
    """Учебный нормированный Hadamard на последней оси длины 4."""
    matrix = x.new_tensor([[1, 1, 1, 1], [1, -1, 1, -1],
                           [1, 1, -1, -1], [1, -1, -1, 1]]) / 2
    return x @ matrix.T

def fp8_roundtrip(x, scale):
    """Один блок E4M3FN: x -> FP8(x/scale) -> scale*float(q).

    scale задан явно. Выбор scale/UE8M0 и CUDA GEMM сюда не входят.
    """
    maximum = torch.finfo(torch.float8_e4m3fn).max
    quantized = (x / scale).clamp(-maximum, maximum).to(torch.float8_e4m3fn)
    return quantized.to(x.dtype) * scale
```

```python
print("Hadamard:", hadamard4(torch.tensor([1., 2., 3., 4.])))
print("FP8:", fp8_roundtrip(torch.tensor([1.1, 2.1, -1.1]), 1.0))
```

Вывод примера:

```text
Hadamard: tensor([ 5., -1., -2.,  0.])
FP8: tensor([ 1.125000,  2.000000, -1.125000])
```

<a name="python-glossary"></a>

## Приложение B Словарь Python и PyTorch

| Запись | Смысл и пример формы |
|---|---|
| `import torch.nn.functional as F` | Импорт функций; `F.silu` не создаёт обучаемый слой |
| `torch.tensor([...])` | Создать tensor из чисел; целые IDs остаются integer |
| `torch.set_default_dtype(torch.float64)` | Dtype новых float-тензоров; здесь выбран для учебных вычислений |
| `x.shape`, `x.shape[-1]` | Кортеж размеров и размер последней оси |
| `x @ W.T` | Matrix multiplication; `.T` переставляет оси 2D-матрицы |
| `x * y` | Поэлементное умножение, не dot product |
| `dim=-1` | Последняя ось: обычно признаки; у attention scores — keys |
| `.sum(dim=0)`, `.mean(dim=-1)` | Сумма/среднее по указанной оси; по умолчанию ось удаляется |
| `keepdim=True` | Сохранить редуцированную ось длины 1: `[T,D] → [T,1]` |
| Broadcasting | Оси длины 1 автоматически повторяются логически: `[T,D]/[T,1]` |
| `x[:,None]`, `x.unsqueeze(1)` | Добавить ось длины 1; для `[H]` получается `[H,1]` |
| `x[None,:]` | Для `[S]` получается `[1,S]` |
| `x[...,0::2]` | `...` оставляет предыдущие оси; берёт элементы `0,2,4,...` последней |
| `x[:,h]`, `x[-1:]` | Выбрать head; взять последний элемент, сохранив ось длины 1 |
| `x[ids]` | Advanced indexing: собрать элементы/строки по целым IDs |
| `.gather(dim,index)` | Выбрать элементы по индивидуальным индексам вдоль одной оси |
| `.topk(k,dim)` | Возвращает named tuple с `.values` и `.indices` |
| `values, ids = ...` | Распаковать два возвращённых объекта; `_` — обычное имя для неиспользуемого |
| `.transpose(a,b)`, `.T` | Поменять две оси местами; `.T` в примерах используется для 2D-матриц |
| `torch.cat(items,dim)` | Склеить по существующей оси; `stack` создаёт новую ось |
| `.contiguous()` | Материализовать tensor в непрерывной раскладке, если текущие strides этого требуют |
| `.reshape(...)` | Перегруппировать существующие элементы; `-1` вычисляет один размер автоматически |
| `.flatten(-2)` | Слить последнюю и предпоследнюю оси; `[T,H,pairs,2]→[T,H,2*pairs]` |
| `torch.stack(items,dim)` | Создать новую ось; список `[T,d]` превращается в `[T,H,d]` при `dim=1` |
| `.repeat(4,1,1)` | Повторить tensor, создавая четыре копии матрицы в примере experts |
| `.expand_as(other)` | Broadcast-view нужной формы без повторного хранения значений |
| `torch.zeros`, `zeros_like`, `eye` | Нули заданной формы, нули формы входа, единичная матрица |
| `torch.arange(0,d,2)` | Последовательность индексов `0,2,4,...`, не включая `d` |
| `torch.einsum('hdc,hd->hc',W,q)` | Для каждого `h,c` сумма по `d`: `sum_d W[h,d,c]*q[h,d]` |
| `torch.einsum('hvc,hc->hv',W,z)` | Для каждого `h,v` сумма по latent-оси `c` |
| `torch.outer(k,delta)` | Матрица всех попарных произведений, shape `[dk,dv]` |
| `mask = a <= b`, `~mask` | Поэлементная bool-маска и её логическое отрицание |
| `.masked_fill(mask,value)` | Заменить указанные элементы; `-torch.inf` исключает их из softmax |
| `.clamp(min=...,max=...)` | Ограничить диапазон; отсутствие `min` оставляет нижнюю границу свободной |
| `.amax(...)`, `isfinite(...).all()` | Максимум по оси; проверка, что все элементы конечны |
| `exp`, `log`, `sqrt`, `sin`, `cos` | Поэлементные математические функции |
| `.clone()` | Независимая копия tensor; входное GDN-state не изменяется |
| `.index_add_(0,ids,src)` | Сложить `src` в строки назначения; `_` означает изменение tensor на месте |
| `.to(dtype)` | Изменить dtype с возможным округлением; `.to(FP8)` не сохраняет исходную точность |
| `.new_tensor(...)` | Создать tensor на device и с dtype другого tensor |
| `torch.finfo(dtype).max` | Максимальное конечное значение floating-point формата |
| `ids.numel()`, `int(ids[i])` | Число элементов и перевод scalar tensor в целый Python scalar |
| `*shared_weights` | Распаковать tuple матриц в позиционные аргументы функции |
| `tensor / scalar`, `base ** tensor` | Поэлементные деление и степень |
| `torch.testing.assert_close(a,b)` | Проверить совпадение форм/значений с допусками |

Все ссылки на `None`, `True`, `False` — стандартные значения Python. `range(n)` даёт индексы от 0 до n−1; `.append` добавляет элемент списка. `math.sqrt`/`math.log` работают со скалярами Python; `torch.sqrt`/`torch.log` — с тензорами. Эти учебные функции предполагают согласованные формы, положительные размеры и корректные значения `k`/group parameters.

<a name="profiling"></a>

## Приложение C Как читать профиль

| Что видно | Какой участок математики искать |
|---|---|
| GEMM / matmul | Проекции, FFN/expert Linear, score/value products |
| RMSNorm / LayerNorm / fused add norm | Нормировка и иногда соседний residual add |
| SiLU and mul / SwiGLU | Gate-ветвь FFN и поэлементное произведение |
| top-k | DSA positions либо MoE experts; различать по входной форме |
| gather / scatter / dispatch / combine | Выбор и перемещение токенов, сбор expert outputs |
| FlashAttention / FlashMLA | Attention-формула, реализованная с оптимизацией памяти/вычислений |
| causal conv / gated delta | Подготовка и recurrent/chunked GDN |
| quant / dequant / FP8 | Смена представления чисел и scale handling |

Число CUDA kernels не равно числу математических операций: один fused kernel может выполнять нормировку, residual и другое вместе; одна операция может разложиться на несколько kernels. MoE all-to-all — обмен данными между устройствами, а не новый тип нейронного преобразования. MTP/speculative decoding — схема дополнительных предсказаний и их проверки, составленная из уже знакомых блоков.

Для одного decode-токена при фиксированном числе heads, без стоимости проекций:

| Механизм | Память истории | Основная работа по истории |
|---|---|---|
| Обычный attention | $O(S(d_k+d_v))$ | $O(S(d_k+d_v))$ |
| MLA с absorption | $O(S(r+d_r))$ | $O(S(r+d_r))$ и output up-projection |
| DSA поверх MLA | MLA cache плюс index cache | Основной MLA по $K_s$ позициям, indexer по $S$ |
| GDN | $O(d_kd_v)$ state плюс короткий conv cache | $O(d_kd_v)$, не растёт с $S$ |

Это счёт порядка операций/элементов, а не прогноз latency: batching, dtype, обмены, layout и fused kernels меняют практическую скорость. GDN — другой механизм модели, его нельзя подставить вместо обученного softmax attention с ожиданием того же ответа.

<a name="sources"></a>

## Источники и проверка

1. [GigaChat 3.5 config](https://huggingface.co/ai-sage/GigaChat3.5-432B-A28B/blob/main/config.json), файл опубликован в revision `b65b38e`.
2. [GigaChat 3.5 modeling](https://huggingface.co/ai-sage/GigaChat3.5-432B-A28B/blob/main/modeling_gigachat3_5.py).
3. [GigaChat 3.5 model card](https://huggingface.co/ai-sage/GigaChat3.5-432B-A28B).
4. [DeepSeek V3.2 config](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json), файл опубликован в revision `ba658fb`.
5. [DeepSeek V3.2 model card](https://huggingface.co/deepseek-ai/DeepSeek-V3.2), направляет к inference-коду V3.2-Exp с той же архитектурой.
6. [DeepSeek official inference/model.py](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py) и [kernel.py](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/kernel.py).
7. [DeepSeek V3 report](https://arxiv.org/html/2412.19437v2), [V3.2 report](https://arxiv.org/html/2512.02556v1), [MLA V2 report](https://arxiv.org/abs/2405.04434).
8. [Gated DeltaNet paper](https://arxiv.org/html/2412.06464v3), [RoFormer paper](https://arxiv.org/abs/2104.09864), [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
9. [PyTorch LayerNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html), [RMSNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.RMSNorm.html), [topk](https://docs.pytorch.org/docs/2.14/generated/torch.topk.html).

Сверка архитектур и исходных численных примеров выполнена 21 октября 2026 года; Markdown подготовлен 27 октября 2026 года. Численные примеры авторские. Python-примеры исходной версии выполнены, C++ реализации скомпилированы с C++17 и предупреждениями. При подготовке Markdown сохранены алгоритмы и выводы; проверены кодовые блоки и внутренние ссылки. Attention сверен с PyTorch SDPA, LayerNorm — с F.layer_norm, conv — с F.conv1d; MLA проверен явным разворачиванием K/V; MoE — отдельной суммой экспертов; GDN — переносом состояния между шагами и независимой матричной формулой. Реальные модели целиком не запускались. Небольшие реализации демонстрируют операции и не служат заменой inference engine.

Форматирование: [математика в GitHub Markdown](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions), [внутренние ссылки и custom anchors](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax#custom-anchors).
