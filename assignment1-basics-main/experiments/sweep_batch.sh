#!/bin/bash
# 实验2: 批大小扫描(固定计算预算 327.68M token/run,固定 lr=1.5e-3 即 exp1 最优)。
# 每个 batch 的步数 = 327.68M / (batch × ctx256),保证总 token(FLOPs)一致。
# batch=128 复用 exp1 的 lr_1.5e-3(val 1.375),这里只补 64 / 192 / 256(256 预期 OOM)。
# 在 assignment1-basics-main 目录下运行。
set -u
LR=1.5e-3
TOK=327680000
CTX=256
for b in 64 192 256; do
  steps=$(( TOK / (b * CTX) ))
  evalint=$(( steps / 20 )); [ "$evalint" -lt 1 ] && evalint=1
  name="batch_${b}"
  echo "===== BATCH RUN START: ${name} (batch=${b}, steps=${steps}, lr=${LR}) $(date +%H:%M:%S) ====="
  bash experiments/run_train.sh "${name}" \
    --batch-size ${b} --max-iters ${steps} --lr ${LR} \
    --log-interval 100 --eval-interval ${evalint} --ckpt-interval 100000 \
    --wandb-project tinystories-batch
  echo "===== BATCH RUN DONE:  ${name} (exit 见日志, OOM 会非0) $(date +%H:%M:%S) ====="
done
echo "===== SWEEP_BATCH_ALL_DONE $(date +%H:%M:%S) ====="
