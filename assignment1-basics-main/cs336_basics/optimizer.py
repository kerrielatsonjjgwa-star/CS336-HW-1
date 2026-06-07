"""优化器与学习率调度模块。

本模块对应讲义 §4.3（AdamW 优化器）与 §4.4（带预热的余弦退火学习率调度）。

包含:
  - AdamW: 直接继承 ``torch.optim.Optimizer`` 基类实现的 AdamW 优化器
    （解耦权重衰减，见 Loshchilov & Hutter, 2019）。
  - get_lr_cosine_schedule: 给定迭代步数返回该步的学习率，实现「线性预热 +
    余弦退火」调度。

注意:
  - AdamW 只能继承 ``torch.optim.Optimizer`` 基类自行实现, 不得使用
    ``torch.optim.Adam`` 等现成优化器。
"""

import math

import torch
import torch.nn as nn


class AdamW(torch.optim.Optimizer):
    """AdamW 优化器（解耦权重衰减版本的 Adam），对应讲义 §4.3。

    与原始 Adam 的区别在于：权重衰减不再混入梯度的一阶/二阶矩估计，而是作为
    一个独立的、与梯度无关的项直接作用于参数（解耦权重衰减）。

    参数:
        params: 待优化的参数（可迭代对象，或 param group 的列表/字典），
            会原样传给 ``torch.optim.Optimizer`` 基类。
        lr (float): 学习率 alpha，默认 1e-3。
        betas (tuple[float, float]): 一阶矩、二阶矩的指数衰减系数
            (beta1, beta2)，默认 (0.9, 0.999)。
        eps (float): 加在分母上的数值稳定项 epsilon，默认 1e-8。
        weight_decay (float): 解耦权重衰减系数 lambda，默认 0.01。

    用法:
        opt = AdamW(model.parameters(), lr=1e-3)
        opt.zero_grad(); loss.backward(); opt.step()
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        #######################################################################
        # TODO: 实现 AdamW.__init__。
        #   1. 校验超参合法性: lr >= 0; 0 <= beta1 < 1 且 0 <= beta2 < 1; eps >= 0;
        #      weight_decay >= 0; 否则抛 ValueError。
        #   2. 把这些超参打包进 defaults 字典:
        #        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        #   3. 调用基类构造: super().__init__(params, defaults)。
        #      基类会据此建立 self.param_groups, 每个 group 是一个 dict。
        #   提示: 逐参数的状态（步数 t、一阶矩 m、二阶矩 v）不在这里创建,
        #         而是在 step() 里用 self.state[p] 惰性初始化; 见讲义 §4.3。
        #######################################################################
        if lr < 0:
            raise ValueError(f"非法学习率 lr={lr}, 需 lr >= 0")
        beta1, beta2 = betas
        if not (0 <= beta1 < 1):
            raise ValueError(f"非法 beta1={beta1}, 需 0 <= beta1 < 1")
        if not (0 <= beta2 < 1):
            raise ValueError(f"非法 beta2={beta2}, 需 0 <= beta2 < 1")
        if eps < 0:
            raise ValueError(f"非法 eps={eps}, 需 eps >= 0")
        if weight_decay < 0:
            raise ValueError(f"非法 weight_decay={weight_decay}, 需 weight_decay >= 0")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        #######################################################################
        #                             END OF YOUR CODE                            #
        #######################################################################

    def step(self, closure=None):
        """执行一步参数更新，对应讲义 §4.3 的 AdamW 更新公式。

        参数:
            closure (Callable | None): 可选的闭包，重新计算并返回 loss；
                若提供则需在更新前调用以重算损失。

        返回:
            loss (float | None): 若提供了 closure 则返回其计算的 loss，否则为 None。
        """
        #######################################################################
        # TODO: 实现 AdamW.step。遍历 self.param_groups 中每个 group 与其参数 p。
        #
        #   若 closure 不为 None: loss = closure() （否则 loss = None）。
        #
        #   对每个 group 读取超参: lr, (beta1, beta2)=betas, eps, weight_decay。
        #   对该 group 内每个 p（跳过 p.grad is None 的参数）:
        #     grad = p.grad.data           # 形状与 p 相同
        #     state = self.state[p]
        #     若 state 为空（首次遇到该参数）则惰性初始化:
        #         state["t"] = 0
        #         state["m"] = torch.zeros_like(p.data)   # 一阶矩, 形状同 p
        #         state["v"] = torch.zeros_like(p.data)   # 二阶矩, 形状同 p
        #
        #   AdamW 更新（t 从 1 开始计数）:
        #     t <- t + 1
        #     m <- beta1 * m + (1 - beta1) * grad                  # 一阶矩(动量)
        #     v <- beta2 * v + (1 - beta2) * grad^2               # 二阶矩(逐元素平方)
        #     alpha_t <- lr * sqrt(1 - beta2^t) / (1 - beta1^t)    # 偏差修正后的步长
        #     p <- p - alpha_t * m / (sqrt(v) + eps)               # 梯度步（逐元素）
        #     p <- p - lr * weight_decay * p                       # 解耦权重衰减
        #   注意: 权重衰减项用的是 lr（而非 alpha_t），且作用在已更新的 p 上;
        #         所有运算建议用原地操作并写回 state["m"]/state["v"]/state["t"]。
        #
        #   最后 return loss。见讲义 §4.3（算法 1）。
        #######################################################################
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                
                if not state:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                
                t, m, v = state["t"], state["m"], state["v"]
                t += 1
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data = p.data - alpha_t * m / (torch.sqrt(v) + eps)
                p.data = p.data - lr * weight_decay * p.data
                
                state["t"] = t                            
                state["m"] = m
                state["v"] = v     
                  
        return loss  
        #######################################################################
        #                             END OF YOUR CODE                            #
        #######################################################################


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """带线性预热的余弦退火学习率调度，对应讲义 §4.4。

    参数:
        it (int): 当前迭代步数（从 0 开始计数）。
        max_learning_rate (float): alpha_max，调度的峰值学习率。
        min_learning_rate (float): alpha_min，退火结束后的最小（最终）学习率。
        warmup_iters (int): T_w，线性预热的迭代步数。
        cosine_cycle_iters (int): T_c，余弦退火结束对应的迭代步数（含预热）。

    返回:
        float: 第 it 步对应的学习率。
    """
    #######################################################################
    # TODO: 实现 get_lr_cosine_schedule。按 it 落在三个区间分段返回学习率:
    #
    #   1) 预热阶段  it < warmup_iters:
    #        lr = (it / warmup_iters) * max_learning_rate
    #        （从 0 线性升到 max_learning_rate）
    #
    #   2) 退火阶段  warmup_iters <= it <= cosine_cycle_iters:
    #        令 progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    #        lr = min_learning_rate
    #             + 0.5 * (1 + cos(pi * progress)) * (max_learning_rate - min_learning_rate)
    #        （从 max_learning_rate 余弦下降到 min_learning_rate）
    #
    #   3) 退火结束后 it > cosine_cycle_iters:
    #        lr = min_learning_rate （保持最小学习率）
    #
    #   提示: 余弦用 math.cos, 圆周率用 math.pi; 注意整数除法要转 float;
    #         边界 it == warmup_iters 与 it == cosine_cycle_iters 都应被正确覆盖。
    #         见讲义 §4.4。
    #######################################################################
    if it < warmup_iters:
        lr = (it / warmup_iters) * max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        lr = min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (max_learning_rate - min_learning_rate)
    else:
        lr = min_learning_rate
        
    return lr
    #######################################################################
    #                             END OF YOUR CODE                            #
    #######################################################################


# ============================== Muon 优化器 ==============================
# 参考 Keller Jordan 原版 Muon + Moonshot/Kimi「Muon is Scalable」的工程改进:
#   - 把动量更新用 Newton-Schulz 五次迭代「正交化」(近似 U·Vᵀ),让各奇异方向步长均衡,
#     比 Adam 更高效地利用每一步梯度;
#   - Kimi 的可扩展改进: 解耦权重衰减(decoupled wd) + 按 sqrt(max(1, fan_out/fan_in)) 缩放更新,
#     统一不同形状权重的更新 RMS,使其与 AdamW 量级可比、可直接配合余弦调度。
# 仅适用于 2D 隐藏权重(attn / ffn 矩阵); embedding / lm_head / norm 等交给 AdamW。


def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """对 2D 矩阵 G 做 Newton-Schulz 五次迭代正交化(在 bf16 上算,快且省显存)。

    返回与 G 同形、奇异值≈1 的近似正交矩阵。系数 (a,b,c) 为 Keller Jordan 调优值。
    """
    assert G.ndim == 2, "newton_schulz5 仅支持 2D 矩阵"
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transpose = G.size(0) > G.size(1)        # 让行数≤列数, 迭代更稳
    if transpose:
        X = X.mT
    X = X / (X.norm() + eps)                  # 归一化使奇异值落入收敛区间
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon(MomentUm Orthogonalized by Newton-schulz),Kimi 风格,仅用于 2D 隐藏权重。

    Args:
        params: 仅 2D 权重(attn/ffn 矩阵);其余参数请用 AdamW。
        lr: 学习率(因更新已正交归一,通常比 AdamW 大,如 0.02)。
        momentum: 动量系数(默认 0.95)。
        weight_decay: 解耦权重衰减(Kimi 改进,默认 0.1)。
        nesterov: 是否用 Nesterov 动量(默认 True)。
        ns_steps: Newton-Schulz 迭代步数(默认 5)。
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 weight_decay: float = 0.1, nesterov: bool = True, ns_steps: int = 5):
        if lr < 0:
            raise ValueError(f"非法 lr={lr}")
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]; mom = group["momentum"]; wd = group["weight_decay"]
            nesterov = group["nesterov"]; ns = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    raise ValueError("Muon 只接受 2D 参数;请把 embedding/lm_head/norm 交给 AdamW")
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)                                # 动量累积
                upd = g.add(buf, alpha=mom) if nesterov else buf     # Nesterov 动量
                ortho = newton_schulz5(upd, ns)                      # 正交化更新方向
                p.mul_(1 - lr * wd)                                  # 解耦权重衰减(Kimi)
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5       # 统一更新 RMS
                p.add_(ortho, alpha=-lr * scale)
        return loss
