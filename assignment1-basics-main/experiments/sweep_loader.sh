#!/bin/bash
# 实验3: 数据加载策略对比(同 FLOPs)。在 assignment1-basics-main 目录下运行。
#   iteration 模式 = get_batch 随机有放回(复用 exp1 的 lr_1.5e-3, val 1.375)。
#   epoch 模式     = iter_epoch_batches 无放回遍历不重叠窗口。
# 两者同预算: batch=128, lr=1.5e-3, 10000 步 = 327.68M token, 余弦周期 10000。
# 这里只跑 epoch 模式一次(iteration 结果直接引用 exp1)。
set -u
echo "===== LOADER RUN START: loader_epoch (epoch 模式, 10000 步, batch128, lr1.5e-3) $(date +%H:%M:%S) ====="
bash experiments/run_train.sh loader_epoch \
  --epochs 1 --max-steps 10000 --batch-size 128 --lr 1.5e-3 \
  --log-interval 100 --eval-interval 500 --ckpt-interval 100000 \
  --wandb-project tinystories-loader
echo "===== LOADER RUN DONE: loader_epoch $(date +%H:%M:%S) ====="
echo "===== SWEEP_LOADER_ALL_DONE $(date +%H:%M:%S) ====="
