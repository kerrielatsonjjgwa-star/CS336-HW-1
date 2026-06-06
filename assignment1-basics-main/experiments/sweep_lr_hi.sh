#!/bin/bash
# 实验1 续:更高学习率,用于触发并定位发散边界(前 5 个 LR 到 3e-3 仍未发散)。
# NaN/inf 早停 → 发散的 run 几分钟即止。在 assignment1-basics-main 目录下运行。
set -u
LRS=(6e-3 1.2e-2 2.5e-2)
ITERS=10000   # full 预算: 128×10000×256 = 327.68M token / run
for lr in "${LRS[@]}"; do
  name="lr_${lr}"
  echo "===== SWEEP RUN START: ${name} (lr=${lr}) $(date +%H:%M:%S) ====="
  bash experiments/run_train.sh "${name}" \
    --batch-size 128 --max-iters ${ITERS} --lr "${lr}" \
    --log-interval 100 --eval-interval 500 --ckpt-interval 100000 \
    --wandb-project tinystories-sweep
  echo "===== SWEEP RUN DONE:  ${name} $(date +%H:%M:%S) ====="
done
echo "===== SWEEP_LR_HI_ALL_DONE $(date +%H:%M:%S) ====="
