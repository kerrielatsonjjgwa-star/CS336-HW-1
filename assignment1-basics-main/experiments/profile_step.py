"""Profile 单步训练:数据准备(get_batch) vs GPU 计算(前向+反向+step)的耗时占比。
用 torch.cuda.synchronize() 保证异步 GPU 计时准确。预热若干步(缓存 .bin + 暖 CUDA)后测 N 步取平均。
用法(assignment 目录): uv run python experiments/profile_step.py
"""
import time
import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.data import get_batch

device = "cuda"
B, CTX = 128, 256
train = np.memmap("data/ts_train.bin", dtype="uint16", mode="r")
model = TransformerLM(10000, CTX, 512, 4, 16, 1344, 10000.0, device)
opt = AdamW(model.parameters(), lr=1.5e-3)


def step():
    t0 = time.perf_counter()
    x, y = get_batch(train, B, CTX, device)     # 数据准备: np 取数 + 转 tensor + 拷到 GPU
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    logits = model(x)                            # GPU 计算: 前向 + 反向 + 裁剪 + 更新
    loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    opt.zero_grad()
    loss.backward()
    gradient_clipping(model.parameters(), 1.0)
    opt.step()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1


print("预热 20 步(缓存数据 + 暖 CUDA)...")
for _ in range(20):
    step()

N = 100
data_t = gpu_t = 0.0
for _ in range(N):
    d, g = step()
    data_t += d
    gpu_t += g
data_t /= N
gpu_t /= N
total = data_t + gpu_t

print(f"\n==== 单步耗时分解(batch={B}, ctx={CTX}, {N} 步平均)====")
print(f"数据准备 get_batch : {data_t*1000:7.2f} ms/step  ({100*data_t/total:5.1f}%)")
print(f"GPU 计算 fwd+bwd   : {gpu_t*1000:7.2f} ms/step  ({100*gpu_t/total:5.1f}%)")
print(f"合计               : {total*1000:7.2f} ms/step  -> {1/total:4.2f} it/s  -> {B*CTX/total/1000:.0f}k tok/s")
print(f"\n>> 数据准备占 {100*data_t/total:.1f}% → pinned+预取最多省这么多;"
      f"GPU 占 {100*gpu_t/total:.1f}% → 真要提速靠 torch.compile / bf16")
