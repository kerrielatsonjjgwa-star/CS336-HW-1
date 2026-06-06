"""
神经网络基础工具函数 (nn_utils)。

本模块收纳一组与具体模型结构无关、但被 Transformer 各处反复使用的张量算子与训练辅助函数：

- ``softmax``           : 数值稳定的 softmax 归一化，见讲义 §3.4.4 (Softmax)。
- ``silu``              : SiLU (Swish) 激活，SwiGLU 前馈网络的门控激活，见讲义 §3.4.4 / §3.4.2。
- ``cross_entropy``     : 数值稳定的平均交叉熵损失，语言模型的训练目标，见讲义 §4.1。
- ``gradient_clipping`` : 按全局 L2 范数对梯度做原地裁剪，稳定训练，见讲义 §4.5。

注意：这些函数全部为纯函数式实现（不持有可学习参数），因此写成普通函数而非 ``nn.Module``。
按作业要求，**不得**使用 ``torch.nn.functional`` 中的现成实现（如 ``F.softmax`` / ``F.cross_entropy`` /
``torch.nn.utils.clip_grad_norm_`` 等），需要自己用基础张量算子拼出来。
"""

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
from torch import Tensor


def softmax(in_features: Tensor, dim: int) -> Tensor:
    """数值稳定的 softmax。

    在指定维度 ``dim`` 上对输入做 softmax 归一化，使该维度上的元素非负且求和为 1。

    Args:
        in_features (Tensor): 任意形状的输入张量（logits）。
        dim (int): 要做归一化的维度。

    Returns:
        Tensor: 与 ``in_features`` 形状相同的张量，沿 ``dim`` 维已归一化为概率分布。
    """
    ###########################################################################
    # TODO: 实现数值稳定的 softmax。公式: softmax(x)_i = exp(x_i) / sum_j exp(x_j)
    #   数值稳定技巧: 先在 dim 维上减去该维最大值再取指数, 即
    #       z = x - max(x, dim, keepdim=True).values
    #       out = exp(z) / sum(exp(z), dim, keepdim=True)
    #   减去常数不改变 softmax 结果, 但能避免 exp 上溢。
    #   形状提示: 用 keepdim=True 保证 max/sum 结果可与原张量沿 dim 广播。
    #   禁止使用 torch.softmax / F.softmax; 见讲义 §3.4.4。
    ###########################################################################
    z = in_features - torch.max(in_features, dim=dim, keepdim=True).values
    out = z.exp() / torch.sum(z.exp(), dim=dim, keepdim=True)
    return out
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def silu(in_features: Tensor) -> Tensor:
    """SiLU (Swish) 激活函数，逐元素作用。

    Args:
        in_features (Tensor): 任意形状的输入张量。

    Returns:
        Tensor: 与 ``in_features`` 形状相同的张量，每个元素已应用 SiLU。
    """
    ###########################################################################
    # TODO: 实现 SiLU 激活, 逐元素计算。公式: silu(x) = x * sigmoid(x)
    #   其中 sigmoid(x) = 1 / (1 + exp(-x)), 可用 torch.sigmoid。
    #   形状提示: 输出形状与输入完全一致, 纯逐元素运算无需广播。
    #   见讲义 §3.4.4 (SwiGLU 中作为门控激活, 另见 §3.4.2)。
    ###########################################################################
    return in_features * torch.sigmoid(in_features)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """平均交叉熵损失（数值稳定）。

    给定未归一化的 logits 与正确类别索引，计算一个 batch 上的平均交叉熵损失。

    Args:
        inputs (Tensor): 形状 ``(batch_size, vocab_size)``。``inputs[i, j]`` 是第 ``i`` 个样本
            第 ``j`` 个类别的未归一化 logit。
        targets (Tensor): 形状 ``(batch_size,)``，每个元素是对应样本的正确类别索引，
            取值范围 ``[0, vocab_size - 1]``。

    Returns:
        Tensor: 标量张量（形状 ``()``），即 batch 上的平均交叉熵损失。
    """
    ###########################################################################
    # TODO: 实现数值稳定的平均交叉熵。
    #   单个样本损失: l_i = -log( softmax(inputs_i)[target_i] )
    #              = -inputs_i[target_i] + log( sum_j exp(inputs_i[j]) )
    #   数值稳定: 计算 log-sum-exp 前先在 vocab 维减去每行最大值 m_i:
    #       log( sum_j exp(x_j) ) = m_i + log( sum_j exp(x_j - m_i) )
    #   不要先做 softmax 再取 log, 直接用上式 (log-sum-exp) 更稳。
    #   形状提示:
    #     - inputs: (batch_size, vocab_size); targets: (batch_size,) 为整型索引。
    #     - 取正确类别 logit 可用高级索引 inputs[arange(batch), targets], 或 torch.gather。
    #     - 先得到每个样本的损失 (batch_size,), 再对 batch 维取均值得到标量。
    #   禁止使用 F.cross_entropy / F.log_softmax; 见讲义 §4.1。
    ###########################################################################
    B = inputs.shape[0]
    m = torch.max(inputs, dim=-1, keepdim=True).values # (B, 1)
    log_sum_exp = m + torch.log(torch.sum(torch.exp(inputs - m), dim=-1, keepdim=True)) # (B, 1)
    ids = torch.arange(B, device=inputs.device)
    logit = inputs[ids, targets][:, None] # (B, 1)
    loss = torch.mean(-logit + log_sum_exp)
    return loss
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def gradient_clipping(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None:
    """按全局梯度 L2 范数做原地梯度裁剪。

    把所有参数梯度拼成一个大向量，计算其整体 L2 范数；若超过 ``max_l2_norm``，
    则按比例缩放所有梯度，使裁剪后的整体范数恰为 ``max_l2_norm``。原地修改 ``p.grad``。

    Args:
        parameters (Iterable[nn.Parameter]): 可训练参数集合（其中部分 ``p.grad`` 可能为 None）。
        max_l2_norm (float): 允许的最大整体 L2 范数（正数）。

    Returns:
        None: 直接原地修改各参数的 ``.grad``，无返回值。
    """
    ###########################################################################
    # TODO: 实现全局 L2 范数梯度裁剪。
    #   记所有参数梯度拼接成的向量为 g, 其整体范数 ||g||_2 = sqrt( sum_p sum(p.grad^2) )。
    #   裁剪规则: 若 ||g||_2 > max_l2_norm, 则对每个梯度乘以缩放系数
    #       clip_coef = max_l2_norm / (||g||_2 + eps)   # eps 取 1e-6, 防止除零
    #       p.grad <- p.grad * clip_coef                # 原地修改, 见 mul_ / detach
    #     否则 (||g||_2 <= max_l2_norm) 不做任何改动。
    #   实现提示:
    #     - 跳过 p.grad 为 None 的参数; 只对存在梯度的参数累加平方和与缩放。
    #     - 先遍历一遍累加 sum(p.grad^2) 得到总范数, 再遍历一遍按系数缩放。
    #     - 原地操作请用 in-place 算子 (如 grad.mul_(clip_coef)), 不要重建新张量。
    #   禁止使用 torch.nn.utils.clip_grad_norm_; 见讲义 §4.5。
    ###########################################################################
    params = [p for p in parameters if p.grad is not None]
    global_grad_norm = 0.0
    for p in params:
        global_grad_norm += (p.grad**2).sum()
    global_grad_norm = torch.sqrt(global_grad_norm)
    if global_grad_norm > max_l2_norm:
        eps = 1e-6
        clip_coef = max_l2_norm / (global_grad_norm + eps)
        for p in params:
            p.grad.mul_(clip_coef)
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
