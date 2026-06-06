#!/bin/bash
# 实验1: 学习率随机搜索 sweep。在 assignment1-basics-main 目录下运行。
# 顺序跑多个 log 均匀采样的学习率(跨收敛→发散),其余超参固定,各 600 步。
# 单次进度在本脚本 stdout(tee 到 runs/sweep_lr.log);各 run 详细日志在 runs/lr_<lr>.log。
set -u
LRS=(1e-4 3e-4 6e-4 1.5e-3 3e-3)
ITERS=10000   # full 计算预算: batch128 × 10000 × ctx256 = 327,680,000 token / run
for lr in "${LRS[@]}"; do
  name="lr_${lr}"
  echo "===== SWEEP RUN START: ${name} (lr=${lr}, iters=${ITERS}) $(date +%H:%M:%S) ====="
  bash experiments/run_train.sh "${name}" \
    --batch-size 128 --max-iters ${ITERS} --lr "${lr}" \
    --log-interval 100 --eval-interval 500 --ckpt-interval 100000 \
    --wandb-project tinystories-sweep
  echo "===== SWEEP RUN DONE:  ${name} $(date +%H:%M:%S) ====="
done
echo "===== SWEEP_LR_ALL_DONE $(date +%H:%M:%S) ====="
