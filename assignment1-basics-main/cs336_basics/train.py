"""训练脚本：在 token 化语料上训练 Transformer 语言模型（讲义 §5.3）。

命令行入口，串起整套训练流程：
    np.memmap 读取 token 数据 -> get_batch 采样 -> 前向 + 交叉熵 ->
    反向传播 -> 梯度裁剪 -> AdamW.step -> 余弦学习率调度 ->
    周期性 save_checkpoint -> (wandb 日志，桩，已注释)。

本文件是「可运行骨架」：argparse 与 main() 结构完整，运行时会在
核心训练步骤处抛 NotImplementedError，属预期。学生需补全各 TODO 横幅。

用法示例：
    python -m cs336_basics.train \\
        --train-data data/train.npy --valid-data data/valid.npy \\
        --vocab-size 10000 --context-length 256 --d-model 512 \\
        --num-layers 4 --num-heads 16 --d-ff 1344 --rope-theta 10000 \\
        --lr 3e-4 --batch-size 64 --max-iters 5000 --checkpoint-dir ckpts
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.data import get_batch, iter_epoch_batches
from cs336_basics.serialization import save_checkpoint, load_checkpoint

import wandb  # 可选：实验追踪（取消注释以启用 wandb 日志桩）


def parse_args() -> argparse.Namespace:
    """解析命令行超参数。

    返回:
        argparse.Namespace: 包含模型结构、优化器、调度、数据路径、设备等全部超参。
    """
    p = argparse.ArgumentParser(description="训练 Transformer 语言模型 (CS336 作业一)")

    # ---- 模型结构超参（须与 TransformerLM.__init__ 一致）----
    p.add_argument("--vocab-size", type=int, default=10000, help="词表大小")
    p.add_argument("--context-length", type=int, default=256, help="上下文长度（序列长度）")
    p.add_argument("--d-model", type=int, default=512, help="模型隐藏维度")
    p.add_argument("--num-layers", type=int, default=4, help="Transformer 层数")
    p.add_argument("--num-heads", type=int, default=16, help="多头注意力头数")
    p.add_argument("--d-ff", type=int, default=1344, help="前馈层内部维度")
    p.add_argument("--rope-theta", type=float, default=10000.0, help="RoPE Theta 参数")

    # ---- 优化器 / 学习率调度超参 ----
    p.add_argument("--lr", type=float, default=3e-4, help="最大学习率 alpha_max")
    p.add_argument("--min-lr", type=float, default=3e-5, help="最小学习率 alpha_min")
    p.add_argument("--weight-decay", type=float, default=0.01, help="AdamW 权重衰减")
    p.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    p.add_argument("--beta2", type=float, default=0.999, help="AdamW beta2")
    p.add_argument("--eps", type=float, default=1e-8, help="AdamW eps")
    p.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪的最大 L2 范数")
    p.add_argument("--warmup", type=int, default=200, help="线性预热迭代数 T_w")

    # ---- 训练循环超参 ----
    p.add_argument("--batch-size", type=int, default=64, help="批大小")
    p.add_argument("--max-iters", type=int, default=5000, help="总训练迭代数")
    p.add_argument(
        "--cosine-cycle-iters",
        type=int,
        default=None,
        help="余弦退火迭代数 T_c（默认取 max-iters）",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="epoch 数：设置后启用 epoch 模式（iter_epoch_batches 无放回遍历，覆盖 --max-iters；"
             "余弦周期自动取 epochs×每epoch步数）",
    )

    # ---- 数据 / checkpoint / 日志路径 ----
    p.add_argument("--train-data", type=str, required=True, help="训练 token 的 .npy/.bin 路径（uint16 1D）")
    p.add_argument("--valid-data", type=str, default=None, help="验证 token 路径（可选）")
    p.add_argument("--dtype-tokens", type=str, default="uint16", help="memmap token 的 numpy dtype")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="checkpoint 输出目录")
    p.add_argument("--resume-from", type=str, default=None, help="从该 checkpoint 续训（可选）")
    p.add_argument("--ckpt-interval", type=int, default=1000, help="每多少步保存一次 checkpoint")
    p.add_argument("--log-interval", type=int, default=10, help="每多少步打印一次训练日志")
    p.add_argument("--eval-interval", type=int, default=500, help="每多少步评估一次验证集")

    # ---- 设备 / 其它 ----
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--wandb-project", type=str, default=None, help="wandb 项目名（启用日志桩时使用）")

    return p.parse_args()


@torch.no_grad()
def evaluate(model, data, batch_size: int, context_length: int, device: str, eval_batches: int = 20) -> float:
    """在数据集上估计平均交叉熵 loss（关闭梯度；多批取平均，估计更稳）。

    切到 eval 模式 -> 采样 eval_batches 个批次算平均 loss -> 切回 train 模式。
    """
    model.eval()
    total = 0.0
    for _ in range(eval_batches):
        x, y = get_batch(data, batch_size, context_length, device)
        logits = model(x)
        total += cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
    model.train()
    return total / eval_batches


def main() -> None:
    """训练主流程：构建数据/模型/优化器，执行训练循环并周期性保存 checkpoint。"""
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if args.cosine_cycle_iters is None:
        args.cosine_cycle_iters = args.max_iters

    #######################################################################
    # TODO: 用 numpy memmap 以只读方式加载 token 数据，避免一次性载入内存。
    #   提示: train_data = np.memmap(args.train_data, dtype=args.dtype_tokens, mode="r")
    #         若有验证集亦同样加载 valid_data；这是 1D uint16 token 序列。
    #######################################################################
    train_data = np.memmap(args.train_data, dtype=args.dtype_tokens, mode="r")
    valid_data = None
    if args.valid_data:
        valid_data = np.memmap(args.valid_data, dtype=args.dtype_tokens, mode="r")
    #######################################################################
    #                             END OF YOUR CODE                            #
    #######################################################################

    #######################################################################
    # TODO: 构建模型与优化器，并移动到目标设备。
    #   model = TransformerLM(vocab_size, context_length, d_model, num_layers,
    #                         num_heads, d_ff, rope_theta, device=...).to(args.device)
    #   optimizer = AdamW(model.parameters(), lr=args.lr,
    #                     betas=(args.beta1, args.beta2), eps=args.eps,
    #                     weight_decay=args.weight_decay)
    #   start_iter = 0；若 args.resume_from 非空，用 load_checkpoint 续训并取回 start_iter。
    #######################################################################
    model = TransformerLM(args.vocab_size, args.context_length, args.d_model, args.num_layers,
                          args.num_heads, args.d_ff, args.rope_theta, args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr, 
                      betas=(args.beta1, args.beta2), eps=args.eps,
                      weight_decay=args.weight_decay)
    if args.resume_from is not None:
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
    else:
        start_iter = 0
    #######################################################################
    #                             END OF YOUR CODE                            #
    #######################################################################

    # ---- 可选：初始化 wandb（日志桩，默认注释掉）----
    if args.wandb_project is not None:
        wandb.init(project=args.wandb_project, config=vars(args))

    # ============ epoch 模式（--epochs 启用：无放回遍历不重叠窗口，跑完即返回）============
    if args.epochs and args.epochs > 0:
        steps_per_epoch = ((len(train_data) - 1) // args.context_length) // args.batch_size
        total_steps = args.epochs * steps_per_epoch
        print(f"[epoch 模式] {args.epochs} epoch × {steps_per_epoch} 步/epoch = {total_steps} 步（余弦周期取 {total_steps}）")
        step = start_iter
        for epoch in range(args.epochs):
            for x, y in iter_epoch_batches(
                train_data, args.batch_size, args.context_length, args.device, seed=args.seed + epoch
            ):
                lr = get_lr_cosine_schedule(step, args.lr, args.min_lr, args.warmup, total_steps)
                for g in optimizer.param_groups:
                    g["lr"] = lr
                logits = model(x)
                loss = cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
                if not torch.isfinite(loss):    # NaN/inf 早停
                    print(f"step {step:6d} | loss={loss.item()} → DIVERGED（发散），提前停止")
                    if args.wandb_project is not None:
                        wandb.log({"diverged": 1.0}, step=step)
                    return
                optimizer.zero_grad()
                loss.backward()
                gradient_clipping(model.parameters(), args.grad_clip)
                optimizer.step()

                if step % args.log_interval == 0:
                    print(f"step {step:6d} | epoch {epoch} | lr {lr:.3e} | loss {loss.item():.4f}")
                    if args.wandb_project is not None:
                        wandb.log({"train/loss": loss.item(), "lr": lr}, step=step)
                if valid_data is not None and step % args.eval_interval == 0:
                    val_loss = evaluate(model, valid_data, args.batch_size, args.context_length, args.device)
                    print(f"step {step:6d} | val/loss {val_loss:.4f}")
                    if args.wandb_project is not None:
                        wandb.log({"val/loss": val_loss}, step=step)
                if (step + 1) % args.ckpt_interval == 0:
                    save_checkpoint(model, optimizer, step + 1,
                                    os.path.join(args.checkpoint_dir, f"ckpt_{step + 1}.pt"))
                step += 1
            print(f"===== epoch {epoch + 1}/{args.epochs} 完成（累计 {step} 步）=====")

        if valid_data is not None:
            val_loss = evaluate(model, valid_data, args.batch_size, args.context_length, args.device)
            print(f"final | val/loss {val_loss:.4f}")
            if args.wandb_project is not None:
                wandb.log({"val/loss": val_loss}, step=total_steps)
        final_path = os.path.join(args.checkpoint_dir, "ckpt_final.pt")
        save_checkpoint(model, optimizer, total_steps, final_path)
        print(f"训练完成（epoch 模式），最终 checkpoint -> {final_path}")
        return

    # ====================== 训练主循环 ======================
    for it in range(start_iter, args.max_iters):
        ###################################################################
        # TODO: 按余弦调度更新本步学习率，并写回优化器各 param_group。
        #   lr = get_lr_cosine_schedule(it, args.lr, args.min_lr,
        #                               args.warmup, args.cosine_cycle_iters)
        #   for g in optimizer.param_groups: g["lr"] = lr
        ###################################################################
        lr = get_lr_cosine_schedule(it, args.lr, args.min_lr, args.warmup, args.cosine_cycle_iters)
        for g in optimizer.param_groups:
            g["lr"] = lr
        ###################################################################
        #                             END OF YOUR CODE                            #
        ###################################################################

        ###################################################################
        # TODO: 采样一个 batch 并前向 + 交叉熵。
        #   x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        #     形状均为 (batch_size, context_length)。
        #   logits = model(x)                      # (batch, ctx, vocab_size)
        #   loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        ###################################################################
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
        if not torch.isfinite(loss):    # NaN/inf 早停: 优化器发散, 不再空跑
            print(f"iter {it:6d} | loss={loss.item()} → DIVERGED（发散），提前停止")
            if args.wandb_project is not None:
                wandb.log({"diverged": 1.0}, step=it)
            return
        ###################################################################
        #                             END OF YOUR CODE                            #
        ###################################################################

        ###################################################################
        # TODO: 反向传播 -> 梯度裁剪 -> AdamW 一步更新。
        #   optimizer.zero_grad(set_to_none=True)
        #   loss.backward()
        #   gradient_clipping(model.parameters(), args.grad_clip)
        #   optimizer.step()
        ###################################################################
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()
        ###################################################################
        #                             END OF YOUR CODE                            #
        ###################################################################

        # ---- 训练日志（含 wandb）----
        if it % args.log_interval == 0:
            print(f"iter {it:6d} | lr {lr:.3e} | loss {loss.item():.4f}")
            if args.wandb_project is not None:
                wandb.log({"train/loss": loss.item(), "lr": lr}, step=it)

        # ---- 周期性验证集评估 + 记录 val/loss ----
        if valid_data is not None and (it % args.eval_interval == 0 or it == args.max_iters - 1):
            val_loss = evaluate(model, valid_data, args.batch_size, args.context_length, args.device)
            print(f"iter {it:6d} | val/loss {val_loss:.4f}")
            if args.wandb_project is not None:
                wandb.log({"val/loss": val_loss}, step=it)

        ###################################################################
        # TODO: 周期性保存 checkpoint（含验证评估桩）。
        #   if (it + 1) % args.ckpt_interval == 0:
        #       path = os.path.join(args.checkpoint_dir, f"ckpt_{it+1}.pt")
        #       save_checkpoint(model, optimizer, it + 1, path)
        #   可选: 每 eval_interval 步在 valid_data 上估计验证 loss（关闭梯度）。
        ###################################################################
        if (it + 1) % args.ckpt_interval == 0:
            path = os.path.join(args.checkpoint_dir, f"ckpt_{it+1}.pt")
            save_checkpoint(model, optimizer, it + 1, path)
        ###################################################################
        #                             END OF YOUR CODE                            #
        ###################################################################

    # ---- 训练结束：保存最终 checkpoint ----
    final_path = os.path.join(args.checkpoint_dir, "ckpt_final.pt")
    save_checkpoint(model, optimizer, args.max_iters, final_path)
    print(f"训练完成，最终 checkpoint 已保存到 {final_path}")


if __name__ == "__main__":
    main()
