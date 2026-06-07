"""加速版训练 + Muon 实验脚本(不动主 train.py)。

支持:
  --compile        torch.compile(model) 融合小 kernel、消启动开销
  --bf16           前向用 bf16 autocast(matmul 半精度);损失/RMSNorm 仍 fp32(数值安全)
  --optimizer muon Kimi 风格 Muon:2D 隐藏权重走 Muon,embedding/lm_head/norm 走 AdamW

自带 tok/s 基准(排除前 --warmup-measure 步的 compile 预热)。
主实验(已完成的 4 个)仍以 cs336_basics.train(fp32 + AdamW)为准;本脚本用于加速基准与 Muon 对比。

用法示例:
  uv run python experiments/train_fast.py --max-iters 400 --batch-size 128 --lr 1.5e-3 \
    --compile --bf16 --optimizer muon --muon-lr 0.02 --wandb-project tinystories-fast --wandb-name muon
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW, Muon, get_lr_cosine_schedule
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.data import get_batch

import wandb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="加速版训练 + Muon 实验")
    # 模型
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    # 优化 / 调度
    p.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    p.add_argument("--lr", type=float, default=1.5e-3, help="AdamW 学习率(也是余弦参考基准)")
    p.add_argument("--min-lr", type=float, default=1.5e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--muon-lr", type=float, default=0.02, help="Muon 学习率(随余弦同形缩放)")
    p.add_argument("--muon-momentum", type=float, default=0.95)
    p.add_argument("--muon-wd", type=float, default=0.1)
    # 加速开关
    p.add_argument("--compile", action="store_true", help="启用 torch.compile")
    p.add_argument("--bf16", action="store_true", help="前向用 bf16 autocast(损失仍 fp32)")
    # 训练循环
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-iters", type=int, default=400)
    p.add_argument("--warmup-measure", type=int, default=30, help="计 tok/s 时跳过的预热步(排除 compile 编译)")
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--eval-interval", type=int, default=200)
    # 数据 / 杂项
    p.add_argument("--train-data", type=str, default="data/ts_train.bin")
    p.add_argument("--valid-data", type=str, default="data/ts_valid.bin")
    p.add_argument("--checkpoint-dir", type=str, default="experiments/runs/fast")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-name", type=str, default=None)
    return p.parse_args()


def build_optimizers(model, args):
    """返回 (优化器列表, 各自基准 lr 列表)。muon: 2D 隐藏权重→Muon,其余→AdamW。"""
    if args.optimizer == "muon":
        named = list(model.named_parameters())
        def is_muon(n, p):
            return p.ndim == 2 and "token_embeddings" not in n and "lm_head" not in n
        muon_p = [p for n, p in named if is_muon(n, p)]
        other_p = [p for n, p in named if not is_muon(n, p)]
        muon = Muon(muon_p, lr=args.muon_lr, momentum=args.muon_momentum, weight_decay=args.muon_wd)
        adamw = AdamW(other_p, lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps,
                      weight_decay=args.weight_decay)
        print(f"[Muon] {len(muon_p)} 个 2D 隐藏权重 → Muon(lr={args.muon_lr}); "
              f"{len(other_p)} 个(embedding/lm_head/norm) → AdamW(lr={args.lr})")
        return [muon, adamw], [args.muon_lr, args.lr]
    adamw = AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps,
                  weight_decay=args.weight_decay)
    return [adamw], [args.lr]


def forward_loss(model, x, y, bf16: bool):
    """前向 + 交叉熵。bf16 时:前向 matmul 用 bf16,损失(log-sum-exp)在 fp32。"""
    if bf16:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        return cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
    logits = model(x)
    return cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))


@torch.no_grad()
def evaluate(model, data, args, n: int = 20) -> float:
    model.eval()
    tot = 0.0
    for _ in range(n):
        x, y = get_batch(data, args.batch_size, args.context_length, args.device)
        tot += forward_loss(model, x, y, args.bf16).item()
    model.train()
    return tot / n


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train = np.memmap(args.train_data, dtype="uint16", mode="r")
    valid = np.memmap(args.valid_data, dtype="uint16", mode="r") if args.valid_data else None

    model = TransformerLM(args.vocab_size, args.context_length, args.d_model, args.num_layers,
                          args.num_heads, args.d_ff, args.rope_theta, args.device)
    if args.compile:
        model = torch.compile(model)
        print("[torch.compile] 已启用(首步会编译,稍慢)")
    print(f"[config] optimizer={args.optimizer} bf16={args.bf16} compile={args.compile} batch={args.batch_size}")

    opts, base_lrs = build_optimizers(model, args)
    ref_lr = args.lr

    if args.wandb_project:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    t0 = time.perf_counter()
    tok = 0
    for it in range(args.max_iters):
        if it == args.warmup_measure:           # 排除 compile 预热后重置计时,测稳态 tok/s
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            tok = 0

        cos = get_lr_cosine_schedule(it, args.lr, args.min_lr, args.warmup, args.max_iters)
        for opt, base in zip(opts, base_lrs):
            for g in opt.param_groups:
                g["lr"] = cos * (base / ref_lr)   # 各优化器按同一余弦形状缩放到各自基准

        x, y = get_batch(train, args.batch_size, args.context_length, args.device)
        loss = forward_loss(model, x, y, args.bf16)
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        for opt in opts:
            opt.step()
        tok += args.batch_size * args.context_length

        if it % args.log_interval == 0:
            torch.cuda.synchronize()
            dt = max(time.perf_counter() - t0, 1e-9)
            tps = tok / dt if it >= args.warmup_measure else 0
            print(f"iter {it:5d} | loss {loss.item():.4f} | lr {cos:.2e} | "
                  f"{tps/1000:6.0f}k tok/s" + ("(预热中)" if it < args.warmup_measure else ""))
            if args.wandb_project:
                wandb.log({"train/loss": loss.item(), "lr": cos, "tok_per_s": tps}, step=it)
        if valid is not None and it % args.eval_interval == 0 and it > 0:
            vl = evaluate(model, valid, args)
            print(f"iter {it:5d} | val/loss {vl:.4f}")
            if args.wandb_project:
                wandb.log({"val/loss": vl}, step=it)

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    steady_tps = tok / dt
    final_val = evaluate(model, valid, args) if valid is not None else float("nan")
    print(f"\n==== [done] optimizer={args.optimizer} bf16={args.bf16} compile={args.compile} ====")
    print(f"稳态吞吐(排除{args.warmup_measure}步预热): {steady_tps/1000:.0f}k tok/s  "
          f"({steady_tps/(args.batch_size*args.context_length):.2f} it/s)")
    print(f"final val/loss: {final_val:.4f}")
    if args.wandb_project:
        wandb.log({"steady_tok_per_s": steady_tps, "final_val": final_val})
        wandb.finish()


if __name__ == "__main__":
    main()
