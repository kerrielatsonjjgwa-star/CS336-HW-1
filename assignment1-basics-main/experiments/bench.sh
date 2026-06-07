#!/bin/bash
# 加速基准:对比 fp32-AdamW / bf16 / compile+bf16 / +Muon 的稳态 tok/s 与早期 loss。
# 在 assignment1-basics-main 目录下运行。用法: bash experiments/bench.sh [iters]
set -u
ITERS=${1:-300}
common="--max-iters ${ITERS} --batch-size 128 --lr 1.5e-3 --warmup 100 --eval-interval 150 --log-interval 100 --warmup-measure 40"
g() { grep -E "config|\[Muon\]|val/loss|稳态吞吐|final val"; }
echo "##### A: fp32 + AdamW(基线)#####"
uv run python experiments/train_fast.py $common 2>&1 | g
echo "##### B: bf16 + AdamW #####"
uv run python experiments/train_fast.py $common --bf16 2>&1 | g
echo "##### C: compile + bf16 + AdamW #####"
uv run python experiments/train_fast.py $common --compile --bf16 2>&1 | g
echo "##### D: compile + bf16 + Muon #####"
uv run python experiments/train_fast.py $common --compile --bf16 --optimizer muon --muon-lr 0.02 2>&1 | g
echo "##### BENCH_DONE #####"
