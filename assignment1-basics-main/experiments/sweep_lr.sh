#!/bin/bash
# 实验1: 学习率随机搜索 sweep。在 assignment1-basics-main 目录下运行。
# 顺序跑多个 log 均匀采样的学习率(跨收敛→发散),其余超参固定,各 600 步。
# 单次进度在本脚本 stdout(tee 到 runs/sweep_lr.log);各 run 详细日志在 runs/lr_<lr>.log。
set -u
LRS=(3e-5 1e-4 3e-4 6e-4 1e-3 2e-3 4e-3 8e-3)
ITERS=600
for lr in "${LRS[@]}"; do
  name="lr_${lr}"
  echo "===== SWEEP RUN START: ${name} (lr=${lr}, iters=${ITERS}) $(date +%H:%M:%S) ====="
  bash experiments/run_train.sh "${name}" \
    --batch-size 128 --max-iters ${ITERS} --lr "${lr}" \
    --log-interval 20 --eval-interval 100 --ckpt-interval 100000 \
    --wandb-project tinystories-sweep
  echo "===== SWEEP RUN DONE:  ${name} $(date +%H:%M:%S) ====="
done
echo "===== SWEEP_LR_ALL_DONE $(date +%H:%M:%S) ====="
