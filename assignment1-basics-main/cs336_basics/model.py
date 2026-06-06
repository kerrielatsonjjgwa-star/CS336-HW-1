"""
cs336_basics.model —— Transformer 语言模型的核心模块（学生填空版）。

本文件实现 CS336 作业一第 3 章（§3.3 ~ §3.5）所需的全部网络组件，自底向上依次为：

    - Linear                       : 无偏置线性层（§3.3）
    - Embedding                    : 词嵌入查表层（§3.4.1）
    - RMSNorm                      : 均方根层归一化（§3.4.1）
    - SwiGLU                       : 位置前馈网络（§3.4.2）
    - RotaryPositionalEmbedding    : 旋转位置编码 RoPE（§3.4.3）
    - scaled_dot_product_attention : 缩放点积注意力（§3.4.4）
    - MultiHeadSelfAttention       : 多头自注意力（§3.4.5）
    - TransformerBlock             : 预归一化 Transformer 块（§3.4）
    - TransformerLM                : 完整的 Transformer 语言模型（§3.5）

实现约束（务必遵守，否则测试无法装载权重）：
    - 仅可使用 torch.nn.Parameter 与 torch.nn 容器（Module / ModuleList / Sequential），
      不得直接调用 torch.nn 现成层（如 nn.Linear / nn.LayerNorm）或 torch.nn.functional。
    - Linear 权重形状为 (out_features, in_features)，前向计算 y = x W^T。
    - 子模块属性名必须与 API 契约严格一致（如 q_proj / k_proj / v_proj / output_proj、
      ln1 / ln2 / attn / ffn、token_embeddings / layers / ln_final / lm_head），
      以便 adapters 通过 state_dict 装载参考权重。

填空说明：每个 __init__ / forward 仅给出签名、中文 docstring 与 TODO 横幅提示，
具体实现请你补全；公式与形状提示见各处注释及讲义 §3.3 ~ §3.5。
"""

import math

import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum, rearrange  # 推荐用 einops 表达带命名维度的张量运算
import torch.nn.functional as F

from cs336_basics.nn_utils import softmax, silu


class Linear(nn.Module):
    """无偏置线性层（§3.3）。

    计算 y = x W^T，其中权重 W 的形状为 (out_features, in_features)。

    Args:
        in_features (int): 输入特征维度 d_in。
        out_features (int): 输出特征维度 d_out。
        device: 参数所在设备（如 "cpu" / "cuda"），可为 None。
        dtype: 参数数据类型，可为 None。

    Attributes:
        weight (nn.Parameter): 形状 (out_features, in_features) 的权重矩阵。
    """

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        #######################################################################
        # TODO: 创建线性层权重参数 self.weight。
        #   - 形状: (out_features, in_features)（注意：行=输出维, 列=输入维）。
        #   - 用 nn.Parameter 包裹一个 torch.empty(..., device=device, dtype=dtype)。
        #   - 建议用截断正态分布初始化(讲义 §3.3): 均值 0, 方差 2/(in+out),
        #     截断在 [-3σ, 3σ]; 可用 torch.nn.init.trunc_normal_ 原地初始化。
        #   - 同时建议把 in_features / out_features 存为实例属性以备用。
        #   提示: 不要使用 nn.Linear; 见讲义 §3.3。
        #######################################################################
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = 2 / (in_features + out_features)
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0*std, b=3.0*std)
        
        #######################################################################
        #                             END OF YOUR CODE                            #
        #######################################################################

    def forward(self, x: Tensor) -> Tensor:
        """对输入做线性变换。

        Args:
            x (Tensor): 形状 (..., in_features)，前导维度任意。

        Returns:
            Tensor: 形状 (..., out_features)。
        """
        #######################################################################
        # TODO: 实现线性前向。公式: y = x W^T
        #   - x 形状 (..., in_features), self.weight 形状 (out_features, in_features)。
        #   - 输出形状 (..., out_features); 前导维度需广播保留。
        #   提示: 可用 einsum("... d_in, d_out d_in -> ... d_out", x, self.weight),
        #         或 x @ self.weight.transpose(-1, -2)。见讲义 §3.3。
        #######################################################################
        return x @ self.weight.T
        #######################################################################
        #                             END OF YOUR CODE                            #
        #######################################################################


class Embedding(nn.Module):
    """词嵌入查表层（§3.4.1）。

    给定一批 token id，从嵌入矩阵中取出对应的嵌入向量。

    Args:
        num_embeddings (int): 词表大小 vocab_size。
        embedding_dim (int): 嵌入维度 d_model。
        device: 参数所在设备，可为 None。
        dtype: 参数数据类型，可为 None。

    Attributes:
        weight (nn.Parameter): 形状 (num_embeddings, embedding_dim) 的嵌入矩阵。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        #######################################################################
        # TODO: 创建嵌入矩阵参数 self.weight。
        #   - 形状: (num_embeddings, embedding_dim)。
        #   - 用 nn.Parameter 包裹 torch.empty(..., device=device, dtype=dtype)。
        #   - 建议用截断正态分布初始化(均值 0, 标准差 1, 截断 [-3, 3]);
        #     可用 torch.nn.init.trunc_normal_。
        #   提示: 不要使用 nn.Embedding; 见讲义 §3.4.1。
        #######################################################################
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3, b=3)
        #######################################################################
        #                             END OF YOUR CODE                            #
        #######################################################################

    def forward(self, token_ids: Tensor) -> Tensor:
        """按 token id 取嵌入向量。

        Args:
            token_ids (Tensor): 整型张量，形状 (...)，元素取值 [0, num_embeddings)。

        Returns:
            Tensor: 形状 (..., embedding_dim)，即每个 id 对应一行嵌入。
        """
        #######################################################################
        # TODO: 实现嵌入查表。
        #   - 用 token_ids 作为索引在 self.weight 的第 0 维上取行。
        #   - 输入 (...)  ->  输出 (..., embedding_dim)。
        #   提示: 直接用高级索引 self.weight[token_ids] 即可,
        #         无需 one-hot 矩阵乘法。见讲义 §3.4.1。
        #######################################################################
        return self.weight[token_ids]
        #######################################################################
        #                             END OF YOUR CODE                            #
        #######################################################################


class RMSNorm(nn.Module):
    """均方根层归一化 RMSNorm（§3.4.1）。

    对最后一维做均方根归一化后再逐元素乘以可学习增益 weight。

    Args:
        d_model (int): 归一化维度（最后一维）。
        eps (float): 分母数值稳定项，默认 1e-5。
        device: 参数所在设备，可为 None。
        dtype: 参数数据类型，可为 None。

    Attributes:
        weight (nn.Parameter): 形状 (d_model,) 的增益向量。
        eps (float): 数值稳定项。
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        #######################################################################
        # TODO: 保存 eps, 并创建增益参数 self.weight。
        #   - self.eps = eps。
        #   - self.weight: nn.Parameter, 形状 (d_model,);
        #     通常初始化为全 1（torch.ones(d_model, device=device, dtype=dtype)）。
        #   提示: 见讲义 §3.4.1。
        #######################################################################
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        
        #######################################################################
        #                             END OF YOUR CODE                            #
        #######################################################################

    def forward(self, x: Tensor) -> Tensor:
        """RMSNorm 前向。

        Args:
            x (Tensor): 形状 (..., d_model)，前导维度任意。

        Returns:
            Tensor: 与 x 同形状、同 dtype 的归一化结果。
        """
        #######################################################################
        # TODO: 实现 RMSNorm 前向。
        #   公式: rms(x) = sqrt(mean(x^2, 最后一维) + eps)
        #         out    = (x / rms(x)) * weight
        #   - 为保证数值稳定, 先把 x 上转为 float32 再计算均方根,
        #     算完后再转回输入原本的 dtype（先记录 in_dtype = x.dtype）。
        #   - 均方在最后一维上求, 注意保持可广播的维度（keepdim=True）。
        #   - weight 形状 (d_model,), 与 (..., d_model) 自动广播。
        #   提示: 见讲义 §3.4.1。
        #######################################################################
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x*x, dim=-1, keepdim=True) + self.eps)
        out = (x / rms) * self.weight
        return out.to(in_dtype)
        #######################################################################
        #                             END OF YOUR CODE                            #
        #######################################################################


class SwiGLU(nn.Module):
    """SwiGLU 位置前馈网络（§3.4.2）。

    计算 FFN(x) = W2 ( SiLU(W1 x) ⊙ (W3 x) )，其中 ⊙ 为逐元素乘。

    Args:
        d_model (int): 输入/输出维度。
        d_ff (int): 前馈内部隐藏维度（通常约 8/3 * d_model 并向上取整到 64 的倍数）。
        device: 参数所在设备，可为 None。
        dtype: 参数数据类型，可为 None。

    Attributes:
        w1 (Linear): Linear(d_model, d_ff)，权重形状 (d_ff, d_model)。
        w2 (Linear): Linear(d_ff, d_model)，权重形状 (d_model, d_ff)。
        w3 (Linear): Linear(d_model, d_ff)，权重形状 (d_ff, d_model)。
    """

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        #######################################################################
        # TODO: 创建三个线性子模块 self.w1, self.w2, self.w3（都用上面的 Linear）。
        #   - self.w1 = Linear(d_model, d_ff, ...)   权重形状 (d_ff, d_model)
        #   - self.w2 = Linear(d_ff, d_model, ...)   权重形状 (d_model, d_ff)
        #   - self.w3 = Linear(d_model, d_ff, ...)   权重形状 (d_ff, d_model)
        #   - 记得把 device / dtype 透传给每个 Linear。
        #   注意: 属性名必须是 w1 / w2 / w3, 以便 state_dict 装载。见讲义 §3.4.2。
        #######################################################################
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        #######################################################################
        #                             END OF YOUR CODE                            #
        #######################################################################

    def forward(self, x: Tensor) -> Tensor:
        """SwiGLU 前向。

        Args:
            x (Tensor): 形状 (..., d_model)。

        Returns:
            Tensor: 形状 (..., d_model)，与输入同形。
        """
        #######################################################################
        # TODO: 实现 SwiGLU 前向。
        #   公式: out = w2( silu(w1(x)) * w3(x) )
        #   - 先分别计算 a = self.w1(x) 与 b = self.w3(x), 二者形状均为 (..., d_ff)。
        #   - 对 a 应用 silu（顶部已从 nn_utils 导入）, 再与 b 逐元素相乘。
        #   - 最后过 self.w2 投回 (..., d_model)。
        #   提示: silu(z) = z * sigmoid(z); 见讲义 §3.4.2。
        #######################################################################
        return self.w2(silu(self.w1(x)) * self.w3(x))
        #######################################################################
        #                             END OF YOUR CODE                            #
        #######################################################################


class RotaryPositionalEmbedding(nn.Module):
    """旋转位置编码 RoPE（§3.4.3）。

    把输入向量按相邻维度两两分组，对每一对 (x_{2i}, x_{2i+1}) 施加一个与
    token 位置 m 和维度对索引 i 相关的二维旋转，从而注入相对位置信息。

    Args:
        theta (float): RoPE 的基底参数 Θ（频率随维度对衰减的底数）。
        d_k (int): 被旋转向量的维度（须为偶数，通常为单头的 head_dim）。
        max_seq_len (int): 预计算正余弦缓存所支持的最大序列长度。
        device: 缓存所在设备，可为 None。
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        #######################################################################
        # TODO: 预计算并缓存 RoPE 所需的 cos / sin 表。
        #   设维度对数 d_k/2, 第 i 对(i=0..d_k/2-1)的角频率:
        #       freq_i = 1.0 / (theta ** (2*i / d_k))
        #   对位置 m=0..max_seq_len-1, 角度 angle[m, i] = m * freq_i,
        #   形状 (max_seq_len, d_k/2)。
        #   计算 cos_cached = cos(angle), sin_cached = sin(angle)。
        #   - 用 self.register_buffer("cos_cached", ..., persistent=False) 缓存,
        #     sin 同理; 这样它们不算作可学习参数、也不进 state_dict。
        #   - 记得把张量放到 device 上。
        #   提示: 可用 torch.arange 配合广播构造 angle; 见讲义 §3.4.3。
        #######################################################################
        freq = 1.0 / theta**(torch.arange(0, d_k, 2, device=device) / d_k)
        pos = torch.arange(max_seq_len, device=device)
        angle = pos[:, None] * freq[None, :]
        cos_cached = torch.cos(angle)
        sin_cached = torch.sin(angle)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)
        #######################################################################
        #                             END OF YOUR CODE                            #
        #######################################################################

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """对查询或键张量施加 RoPE 旋转。

        Args:
            x (Tensor): 形状 (..., seq_len, d_k)，最后一维为待旋转维度。
            token_positions (Tensor): 整型张量，形状 (..., seq_len)，给出每个位置索引 m，
                用于从缓存表中选取对应的 cos / sin。

        Returns:
            Tensor: 与 x 同形状，已施加旋转。
        """
        #######################################################################
        # TODO: 实现 RoPE 前向。
        #   - 用 token_positions 在 cos_cached / sin_cached 上索引,
        #     得到本批次每个位置的 cos, sin（形状 (..., seq_len, d_k/2)）。
        #   - 把 x 的最后一维按相邻成对拆分: 偶数下标 x1 = x[..., 0::2],
        #     奇数下标 x2 = x[..., 1::2]（各形状 (..., seq_len, d_k/2)）。
        #   - 二维旋转公式（对每一对）:
        #        out_even = x1 * cos - x2 * sin
        #        out_odd  = x1 * sin + x2 * cos
        #   - 再把 out_even / out_odd 交错合并回最后一维 d_k
        #     （注意还原成 0::2 与 1::2 的原始排布; 可用 stack 后 rearrange/flatten）。
        #   提示: 旋转不改变形状; cos/sin 与 x1/x2 的前导维度需广播对齐。
        #         见讲义 §3.4.3。
        #######################################################################
        cos, sin = self.cos_cached[token_positions], self.sin_cached[token_positions]
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., ::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        
        return out 
        #######################################################################
        #                             END OF YOUR CODE                            #
        #######################################################################


def scaled_dot_product_attention(
    Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None
) -> Tensor:
    """缩放点积注意力（§3.4.4）。

    Args:
        Q (Tensor): 查询，形状 (..., queries, d_k)。
        K (Tensor): 键，形状 (..., keys, d_k)。
        V (Tensor): 值，形状 (..., keys, d_v)。
        mask (Tensor | None): 布尔掩码，形状 (..., queries, keys)。
            约定 True 表示「允许注意 / 保留」，False 表示「屏蔽」（对应位置注意力权重置 0）。

    Returns:
        Tensor: 注意力输出，形状 (..., queries, d_v)。
    """
    #######################################################################
    # TODO: 实现缩放点积注意力。
    #   公式: Attention(Q,K,V) = softmax( Q K^T / sqrt(d_k) + bias ) V
    #   步骤:
    #     1) 计算打分 scores = Q @ K^T / sqrt(d_k),
    #        Q (..., queries, d_k), K (..., keys, d_k) -> scores (..., queries, keys);
    #        可用 einsum("... q d, ... k d -> ... q k", Q, K)。
    #     2) 若 mask 非 None: 在 mask 为 False 的位置把 scores 置为 -inf
    #        （例如 scores.masked_fill(~mask, float("-inf"))），
    #        这样 softmax 后该处权重为 0。
    #     3) 对最后一维(keys)做 softmax（用顶部导入的 softmax, 传 dim=-1）。
    #     4) 用权重对 V 加权求和: out = weights @ V -> (..., queries, d_v)。
    #   提示: d_k 取自 Q 的最后一维; 见讲义 §3.4.4。
    #######################################################################
    d = Q.shape[-1]
    score = Q @ K.transpose(-1, -2) / math.sqrt(d)
    if mask is not None:
        score = score.masked_fill(~mask, float("-inf"))
    attn = softmax(score, dim=-1)
    out = attn @ V
    return out
    #######################################################################
    #                             END OF YOUR CODE                            #
    #######################################################################


class MultiHeadSelfAttention(nn.Module):
    """多头因果自注意力（§3.4.5）。

    将输入投影为 Q / K / V，拆分为 num_heads 个头分别做缩放点积注意力（带因果掩码），
    再拼接各头输出并经输出投影。可选地在每个头内对 Q / K 施加 RoPE。

    Args:
        d_model (int): 模型维度（输入输出维度）。
        num_heads (int): 注意力头数，d_model 必须能被其整除。
        use_rope (bool): 是否在 Q / K 上施加 RoPE，默认 False。
        max_seq_len (int | None): 使用 RoPE 时的最大序列长度（用于预计算缓存）。
        theta (float | None): 使用 RoPE 时的 Θ 参数。
        device: 参数所在设备，可为 None。
        dtype: 参数数据类型，可为 None。

    Attributes:
        q_proj, k_proj, v_proj, output_proj (Linear): 四个 (d_model, d_model) 投影。
        rope (RotaryPositionalEmbedding): 仅在 use_rope=True 时存在，作用于单头维度 d_model//num_heads。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        #######################################################################
        # TODO: 保存超参并创建投影子模块（属性名严格如下, 供 state_dict 装载）。
        #   - 断言 d_model % num_heads == 0; 记 head_dim = d_model // num_heads。
        #   - 保存 self.d_model, self.num_heads, self.head_dim, self.use_rope。
        #   - 创建四个 Linear（都是 (d_model, d_model)）:
        #         self.q_proj      = Linear(d_model, d_model, ...)
        #         self.k_proj      = Linear(d_model, d_model, ...)
        #         self.v_proj      = Linear(d_model, d_model, ...)
        #         self.output_proj = Linear(d_model, d_model, ...)
        #   - 若 use_rope 为 True:
        #         self.rope = RotaryPositionalEmbedding(theta, head_dim, max_seq_len, device)
        #     （RoPE 作用在单头维度 head_dim 上, 而非整个 d_model）。
        #   提示: 见讲义 §3.4.5。
        #######################################################################
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.kwargs = {"device": device, "dtype": dtype}
        self.use_rope = use_rope
        if self.use_rope:
            self.rope = RotaryPositionalEmbedding(theta, self.head_dim, max_seq_len, device)
        
        self.q_proj = Linear(d_model, d_model, **self.kwargs)
        self.k_proj = Linear(d_model, d_model, **self.kwargs)
        self.v_proj = Linear(d_model, d_model, **self.kwargs)
        self.output_proj = Linear(d_model, d_model, **self.kwargs)
                                                                       
        #######################################################################
        #                             END OF YOUR CODE                            #
        #######################################################################

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        """多头自注意力前向（带因果掩码）。

        Args:
            x (Tensor): 形状 (..., seq_len, d_model)。
            token_positions (Tensor | None): 形状 (..., seq_len) 的位置索引，
                仅在 use_rope=True 时用于 RoPE；若为 None 可默认取 0..seq_len-1。

        Returns:
            Tensor: 形状 (..., seq_len, d_model)。
        """
        #######################################################################
        # TODO: 实现多头自注意力前向。
        #   步骤:
        #     1) 线性投影: q = self.q_proj(x), k = self.k_proj(x), v = self.v_proj(x),
        #        三者形状均为 (..., seq_len, d_model)。
        #     2) 拆分多头: 把最后一维 d_model 重排为 (num_heads, head_dim) 并把头维提到
        #        seq 之前, 得到 (..., num_heads, seq_len, head_dim)。
        #        可用 rearrange(q, "... s (h d) -> ... h s d", h=self.num_heads)。
        #     3) 若 use_rope: 对 q, k 施加 self.rope(..., token_positions);
        #        token_positions 为 None 时, 用 torch.arange(seq_len) 构造默认位置。
        #        注意 RoPE 作用在 head_dim 上, 需让位置维度与 (..., h, s, d) 对齐广播。
        #     4) 构造因果掩码: 形状 (seq_len, seq_len) 的下三角布尔矩阵
        #        （查询 i 只能注意键 j<=i）, 可用 torch.tril(torch.ones(...)).bool()。
        #     5) 调用 scaled_dot_product_attention(q, k, v, mask) 得到
        #        (..., num_heads, seq_len, head_dim)。
        #     6) 合并多头: rearrange 回 (..., seq_len, d_model)。
        #     7) 过 self.output_proj 得到最终输出。
        #   提示: 注意 mask 约定 True=保留; 见讲义 §3.4.5。
        #######################################################################
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
        B, N, S, D = q.shape
        
        if self.use_rope:
            if token_positions is None:
                token_positions = torch.arange(S, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        
        
        mask = torch.tril(torch.ones(S, S, device=x.device)).bool()
        out = scaled_dot_product_attention(q, k, v, mask)
        out = rearrange(out, "b n s d -> b s (n d)")
        return self.output_proj(out)
        #######################################################################
        #                             END OF YOUR CODE                            #
        #######################################################################


class TransformerBlock(nn.Module):
    """预归一化 Transformer 块（§3.4）。

    结构（残差 + 预归一化）：
        x = x + Attn(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
    其中注意力使用 RoPE。

    Args:
        d_model (int): 模型维度。
        num_heads (int): 注意力头数。
        d_ff (int): 前馈隐藏维度。
        max_seq_len (int): 最大序列长度（供 RoPE 预计算）。
        theta (float): RoPE 的 Θ 参数。
        device: 参数所在设备，可为 None。
        dtype: 参数数据类型，可为 None。

    Attributes:
        ln1 (RMSNorm): 注意力子层前的归一化。
        attn (MultiHeadSelfAttention): 带 RoPE 的多头自注意力。
        ln2 (RMSNorm): 前馈子层前的归一化。
        ffn (SwiGLU): 位置前馈网络。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        #######################################################################
        # TODO: 创建四个子模块（属性名严格如下, 对应 state_dict 键）。
        #   - self.ln1 = RMSNorm(d_model, ...)
        #   - self.attn = MultiHeadSelfAttention(
        #         d_model, num_heads, use_rope=True,
        #         max_seq_len=max_seq_len, theta=theta, device/dtype=...)
        #   - self.ln2 = RMSNorm(d_model, ...)
        #   - self.ffn = SwiGLU(d_model, d_ff, ...)
        #   权重键将形如: ln1.weight, attn.q_proj.weight, ..., ffn.w1.weight, ln2.weight。
        #   提示: 见讲义 §3.4。
        #######################################################################
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, True, max_seq_len, theta, device, dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        #######################################################################
        #                             END OF YOUR CODE                            #
        #######################################################################

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        """Transformer 块前向（预归一化 + 两个残差连接）。

        Args:
            x (Tensor): 形状 (..., seq_len, d_model)。
            token_positions (Tensor | None): 形状 (..., seq_len) 的位置索引，传给注意力的 RoPE。

        Returns:
            Tensor: 形状 (..., seq_len, d_model)。
        """
        #######################################################################
        # TODO: 实现预归一化 Transformer 块前向。
        #   公式:
        #     h = x + self.attn( self.ln1(x), token_positions )
        #     y = h + self.ffn( self.ln2(h) )
        #   - 注意是「先归一化再进子层, 子层输出加回残差」（pre-norm）。
        #   - token_positions 透传给 self.attn。
        #   提示: 见讲义 §3.4。
        #######################################################################
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x
        #######################################################################
        #                             END OF YOUR CODE                            #
        #######################################################################


class TransformerLM(nn.Module):
    """Transformer 语言模型（§3.5）。

    词嵌入 → 若干个预归一化 Transformer 块 → 最终 RMSNorm → 输出投影到词表，
    得到每个位置对下一个 token 的未归一化对数概率（logits）。

    Args:
        vocab_size (int): 词表大小。
        context_length (int): 上下文最大长度（也用于 RoPE 的 max_seq_len）。
        d_model (int): 模型维度。
        num_layers (int): Transformer 块层数。
        num_heads (int): 每层注意力头数。
        d_ff (int): 前馈隐藏维度。
        rope_theta (float): RoPE 的 Θ 参数。
        device: 参数所在设备，可为 None。
        dtype: 参数数据类型，可为 None。

    Attributes:
        token_embeddings (Embedding): (vocab_size, d_model) 词嵌入。
        layers (nn.ModuleList): num_layers 个 TransformerBlock。
        ln_final (RMSNorm): 最终归一化。
        lm_head (Linear): (vocab_size, d_model) 输出投影到词表。
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        #######################################################################
        # TODO: 搭建语言模型骨架（属性名严格如下, 对应 state_dict 键）。
        #   - self.token_embeddings = Embedding(vocab_size, d_model, ...)
        #   - self.layers = nn.ModuleList([
        #         TransformerBlock(d_model, num_heads, d_ff,
        #                          max_seq_len=context_length, theta=rope_theta, ...)
        #         for _ in range(num_layers)
        #     ])
        #   - self.ln_final = RMSNorm(d_model, ...)
        #   - self.lm_head = Linear(d_model, vocab_size, ...)
        #   （可选）保存 self.context_length 等超参以备 forward 使用。
        #   权重键将形如: token_embeddings.weight, layers.{i}.<块内键>,
        #               ln_final.weight, lm_head.weight。见讲义 §3.5。
        #######################################################################
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device, dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)
        self.context_length = context_length
        #######################################################################
        #                             END OF YOUR CODE                            #
        #######################################################################

    def forward(self, token_ids: Tensor) -> Tensor:
        """语言模型前向。

        Args:
            token_ids (Tensor): 整型张量，形状 (batch_size, seq_len)，seq_len ≤ context_length。

        Returns:
            Tensor: 形状 (batch_size, seq_len, vocab_size) 的未归一化 logits。
        """
        #######################################################################
        # TODO: 实现语言模型前向。
        #   步骤:
        #     1) 嵌入: x = self.token_embeddings(token_ids) -> (batch, seq, d_model)。
        #     2) 构造位置索引 token_positions = torch.arange(seq_len)（设备与输入一致),
        #        供各层的 RoPE 使用（可在层内默认生成, 这里也可显式传入）。
        #     3) 依次过 self.layers 中每个 TransformerBlock:
        #            x = block(x, token_positions)
        #     4) x = self.ln_final(x) 做最终归一化。
        #     5) logits = self.lm_head(x) -> (batch, seq, vocab_size)。
        #   注意: 返回的是未经 softmax 的 logits（交叉熵在外部计算）。见讲义 §3.5。
        #######################################################################
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.ln_final(x))
        return logits
        #######################################################################
        #                             END OF YOUR CODE                            #
        #######################################################################
