# CS336 作业一 §7 实验报告 — TinyStories

> 自动化实验 campaign(由 Claude Code 通过 `/loop` 在租用的 RTX 5090 上自主执行 + 监控)。
> 监控快照见 [`log/`](./log/);本文件汇总最终结果与图表。

## 实验设置

| 项 | 值 |
|---|---|
| 硬件 | RTX 5090 (32 GB) · Xeon Gold 6530 (14 vCPU) · Paratera |
| 数据 | TinyStories,vocab=10000;`ts_train.bin` 540,796,778 token / `ts_valid.bin` 5,461,210 token |
| 固定模型 | d_model=512 · num_layers=4 · num_heads=16 · d_ff=1344 · ctx=256 · RoPEθ=10000(≈17M 参数) |
| 优化器 | AdamW(β=(0.9,0.999), wd=0.01, grad_clip=1.0) · 余弦 LR + 200 步 warmup |
| 监控 | wandb project `tinystories-sweep` + 本地日志解析 |

> **验证损失口径**:`val/loss` 为每词元(per-token)平均交叉熵,目标实验 4 要求 < 1.45。

---

## 进度跟踪

| 阶段 | 状态 | 说明 |
|---|---|---|
| 校准(测速) | ⏳ 进行中 | 测 batch=128 的 it/s,据此定 sweep 规模 |
| 实验 1:LR sweep | ⬜ 待开始 | 随机搜索学习率,画多条学习曲线 |
| 实验 2:batch sweep | ⬜ 待开始 | 增大 batch 直到 OOM |
| 实验 3:epoch vs iteration(同 FLOPs) | ⬜ 待开始 | 两种数据加载策略对比 |
| 实验 4:达到 val/loss < 1.45 | ⬜ 待开始 | 调参命中目标 |

---

## 校准(训练速度)

_待填:batch=128 下的 it/s、单步耗时、据此估计的各实验 per-run 步数与 sweep 次数。_

---

## 实验 1:学习率扫描(随机搜索)

**方法**:固定其余超参,在 log 均匀区间随机采样若干学习率,各跑相同步数;记录最终 train/val loss,发散则标注 `DIVERGED`。

_待填:超参表 + 各 LR 最终损失表 + 学习曲线图(`plots/exp1_lr_sweep.png`)。_

---

## 实验 2:批大小扫描

**方法**:逐步增大 batch(64 → … → OOM),记录可行的最大 batch、各自吞吐与学习曲线;必要时重调 LR。

_待填:batch–显存–吞吐表 + 学习曲线图(`plots/exp2_batch_sweep.png`)。_

---

## 实验 3:数据加载策略(epoch vs iteration,同 FLOPs)

**方法**:在相同总计算量(相同总 step×batch×ctx)下,对比 iteration 模式(`get_batch` 随机有放回)与 epoch 模式(`iter_epoch_batches` 无放回)的验证损失曲线。

_待填:两曲线对比图(`plots/exp3_loader.png`)+ 结论。_

---

## 实验 4:命中 val/loss < 1.45

**方法**:基于前几个实验的最优 LR/batch,加大训练步数(必要时调模型/调度)直到验证损失 < 1.45。

_待填:命中配置 + 最终 val/loss + 收敛曲线(`plots/exp4_target.png`)。_

---

## 结论

_待填。_
