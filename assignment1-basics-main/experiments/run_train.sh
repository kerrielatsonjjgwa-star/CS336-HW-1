#!/bin/bash
# 通用单次训练封装(在 assignment1-basics-main 目录下运行)。
# 用法: bash experiments/run_train.sh <run_name> [传给 train.py 的额外参数...]
# 固定模型/数据配置, 额外参数覆盖 batch/lr/max-iters/epochs/intervals/wandb 等。
# 输出 tee 到 experiments/runs/<run_name>.log, 首尾打 START/END 时间戳便于算耗时。
set -u
name="$1"; shift
export WANDB_NAME="$name"     # wandb run 名 = run_name, 便于区分 sweep 各次
export PYTHONUNBUFFERED=1     # 关闭 stdout 块缓冲, 让 print 的 iter/loss 实时进日志
mkdir -p experiments/runs
log="experiments/runs/${name}.log"
{
  echo "RUN_NAME=${name}"
  echo "EXTRA_ARGS=$*"
  echo "START=$(date +%s)"
  uv run python -m cs336_basics.train \
    --train-data data/ts_train.bin --valid-data data/ts_valid.bin \
    --vocab-size 10000 --context-length 256 --d-model 512 --num-layers 4 \
    --num-heads 16 --d-ff 1344 --rope-theta 10000 \
    --checkpoint-dir "experiments/runs/${name}" "$@"
  echo "EXIT=$?"
  echo "END=$(date +%s)"
} 2>&1 | tee "$log"
