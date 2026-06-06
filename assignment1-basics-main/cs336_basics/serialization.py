"""序列化与检查点 (Checkpointing)。

对应讲义 §5.2。

训练大模型耗时很长, 需要周期性地把「模型权重 + 优化器状态 + 当前迭代步数」
一起保存到磁盘 (checkpoint), 以便中断后能从断点恢复继续训练。

本模块提供两个函数:
  - save_checkpoint: 把 (model, optimizer, iteration) 打包写入 out。
  - load_checkpoint: 从 src 读回, 原地恢复 model 与 optimizer 的状态, 并返回保存时的 iteration。

实现要点 (思考方向):
  - PyTorch 中 nn.Module 与 torch.optim.Optimizer 的可恢复状态分别由
    `model.state_dict()` / `optimizer.state_dict()` 给出, 并可用
    `model.load_state_dict(...)` / `optimizer.load_state_dict(...)` 原地加载。
  - 持久化用 `torch.save(obj, path_or_file)` 与 `torch.load(path_or_file)`;
    二者均同时支持「路径字符串 / os.PathLike」与「已打开的二进制文件对象」。
"""

import os
from typing import IO, BinaryIO

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """把模型、优化器与迭代步数序列化保存到 out。

    参数:
        model (nn.Module): 需要保存其参数状态的模型。
        optimizer (torch.optim.Optimizer): 需要保存其内部状态 (如 AdamW 的一/二阶矩) 的优化器。
        iteration (int): 当前已完成的训练迭代步数, 一并保存以便恢复时知道从第几步继续。
        out (str | os.PathLike | BinaryIO | IO[bytes]): 目标路径或已打开的二进制文件对象。

    返回:
        None。本函数只产生副作用 (向 out 写入数据)。
    """
    ###########################################################################
    # TODO: 实现 save_checkpoint。把三样东西打包成一个可序列化对象再写出。
    #   步骤提示:
    #     1) 取出 model.state_dict() 与 optimizer.state_dict() (均为 dict)。
    #     2) 连同 iteration 组装成一个字典, 例如约定如下键名 (load 时需对应):
    #          {"model": <model 状态>, "optimizer": <optimizer 状态>, "iteration": <int>}
    #     3) 用 torch.save(obj, out) 写出; out 既可以是路径也可以是文件对象,
    #        torch.save 都能处理, 无需自己 open。
    #   见讲义 §5.2。
    ###########################################################################
    obj = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    torch.save(obj, out)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """从 src 恢复模型与优化器状态, 并返回保存时的迭代步数。

    参数:
        src (str | os.PathLike | BinaryIO | IO[bytes]): 检查点的路径或已打开的二进制文件对象。
        model (nn.Module): 将被原地恢复参数状态的模型 (需与保存时结构一致)。
        optimizer (torch.optim.Optimizer): 将被原地恢复内部状态的优化器。

    返回:
        int: 之前 save_checkpoint 时写入的 iteration (训练已完成的迭代步数)。
    """
    ###########################################################################
    # TODO: 实现 load_checkpoint, 与 save_checkpoint 的键名约定保持一致。
    #   步骤提示:
    #     1) 用 torch.load(src) 读回保存时的那个字典对象。
    #     2) model.load_state_dict(obj["model"])      # 原地恢复模型参数
    #        optimizer.load_state_dict(obj["optimizer"])  # 原地恢复优化器状态
    #        注意: 这两个 load_state_dict 是「就地修改」, 没有返回值, 不要去接收它。
    #     3) 返回 obj["iteration"] (一个 int)。
    #   见讲义 §5.2。
    ########################################################################### 
    obj = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
