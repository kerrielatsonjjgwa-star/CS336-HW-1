"""
数据加载工具 (讲义 §5.1)。

本模块负责从一个已分词、扁平化的整型 token 序列 (1D 数组) 中采样
用于语言模型训练的小批量 (mini-batch)。语言模型采用「下一个 token 预测」
目标: 给定一个长度为 context_length 的输入序列, 其标签 (label) 是把输入序列
整体向右平移一位得到的序列。

数据集通常很大, 一般以 np.memmap 的形式驻留在磁盘上而非全部读入内存;
本模块的 get_batch 只按随机起点切片取出需要的片段, 因此内存友好。
对应讲义 §5.1 (Data Loading / Batching)。
"""

import math
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从 1D 的 token 数组中随机采样一个用于语言建模的批次。

    语言模型训练目标为「下一个 token 预测」: 对每个采样位置, 取一段长度为
    context_length 的连续片段作为输入 x, 把该片段整体右移一位的连续片段作为
    标签 y (即 y[t] 是 x[t] 的下一个 token)。

    参数:
        dataset (npt.NDArray): 1D 的整型 numpy 数组 (token id 序列), 形状 (n,);
            可能是 np.memmap, 不应整体拷贝进内存。
        batch_size (int): 批大小 B, 即一次采样多少条样本。
        context_length (int): 上下文长度 m, 即每条样本的序列长度。
        device (str): PyTorch 设备字符串 (如 'cpu'、'cuda:0'),
            采样得到的张量应放置到该设备上。

    返回:
        tuple[torch.Tensor, torch.Tensor]:
            - inputs:  形状 (batch_size, context_length) 的 torch.LongTensor, 输入序列。
            - targets: 形状 (batch_size, context_length) 的 torch.LongTensor, 对应的下一 token 标签。
    """
    ###########################################################################
    # TODO: 实现语言模型的批次采样。设 n = len(dataset)。
    #   1) 合法起点范围: 每个样本占用 context_length+1 个连续 token
    #      (输入 m 个 + 标签需要再多 1 个), 故起点 i 必须满足
    #      0 <= i <= n - context_length - 1。
    #   2) 随机抽 batch_size 个起点: 用 np.random.randint(0, n - context_length, size=batch_size)
    #      得到形状 (batch_size,) 的起点数组 (允许重复采样)。
    #   3) 构造输入与标签 (两种常见做法, 任选其一):
    #        - 输入  x[b, t] = dataset[start[b] + t]          , t = 0..context_length-1
    #        - 标签  y[b, t] = dataset[start[b] + t + 1]       , 即 x 右移一位
    #      提示: 可用广播构造索引矩阵 idx = start[:, None] + np.arange(context_length)[None, :],
    #            形状 (batch_size, context_length), 再用 dataset[idx]、dataset[idx + 1] 一次取出。
    #   4) 转成张量并搬运到 device:
    #        - 先 np.ascontiguousarray / 复制成可写数组 (memmap 切片可能不可写),
    #          用 torch.from_numpy(...) 或 torch.tensor(..., dtype=torch.long)。
    #        - dtype 必须为 torch.long (int64); 最后 .to(device)。
    #   形状提示: inputs、targets 均为 (batch_size, context_length)。
    #   见讲义 §5.1。
    ###########################################################################
    n = len(dataset)
    start = np.random.randint(0, n - context_length, size=batch_size)
    idx = start[:, None] + np.arange(context_length)[None, :]
    x = dataset[idx]
    y = dataset[idx + 1]
    input = torch.tensor(x, dtype=torch.long, device=device)
    target = torch.tensor(y, dtype=torch.long, device=device)   
    return (input, target)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def iter_epoch_batches(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
    *,
    shuffle: bool = True,
    drop_last: bool = True,
    seed: int | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """按 epoch 遍历数据：切成不重叠窗口 -> (可选)打乱 -> 按 batch_size 成批产出。

    与 get_batch 的区别（见讲义 §5.1 的设计讨论）：
      - get_batch          : 每步「随机有放回」采样，LLM 预训练标准做法（iteration 式，
                             允许窗口重叠/重复；单元测试要求其起点服从二项分布）。
      - iter_epoch_batches : 「无放回」遍历，一个 epoch 内每个不重叠窗口恰好用一次
                             （epoch 式，适合监督学习 / 微调 / 小数据多轮训练）。

    每个窗口取 context_length+1 个连续 token：输入 x = 前 ctx 个，标签 y = 整体右移一位。
    窗口起点为 0, ctx, 2·ctx, ...（互不重叠）。每个 epoch 重新打乱窗口顺序。

    Args:
        dataset (npt.NDArray): 1D 整型 token 数组（可为 np.memmap）。
        batch_size (int): 批大小 B。
        context_length (int): 序列长度 m。
        device (str): 张量放置设备。
        shuffle (bool): 是否打乱窗口顺序（默认 True；每次调用都重新打乱）。
        drop_last (bool): 丢弃最后不足 B 的尾批，使每批大小一致（训练常用，默认 True）。
        seed (int | None): 打乱用随机种子；None 则每次（每个 epoch）都不同。

    Yields:
        tuple[torch.Tensor, torch.Tensor]:
            inputs、targets，形状均 (batch_size, context_length) 的 LongTensor，在 device 上。

    用法（跑多个 epoch，每轮换一个 seed 以重新打乱）：
        for epoch in range(num_epochs):
            for x, y in iter_epoch_batches(data, B, m, device, seed=epoch):
                ...  # 前向 / 反向 / 更新
    """
    n = len(dataset)
    num_windows = (n - 1) // context_length                  # 不重叠窗口数（每窗需 ctx+1 个 token）
    starts = np.arange(num_windows, dtype=np.int64) * context_length  # 起点 0, ctx, 2ctx, ...
    if shuffle:
        np.random.default_rng(seed).shuffle(starts)          # 原地打乱窗口顺序

    num_batches = num_windows // batch_size if drop_last else math.ceil(num_windows / batch_size)
    arange_ctx = np.arange(context_length)[None, :]
    for b in range(num_batches):
        batch_starts = starts[b * batch_size : (b + 1) * batch_size]  # (≤B,)
        idx = batch_starts[:, None] + arange_ctx                       # (B, ctx)
        x = torch.tensor(dataset[idx], dtype=torch.long, device=device)
        y = torch.tensor(dataset[idx + 1], dtype=torch.long, device=device)
        yield x, y
