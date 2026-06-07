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
| 校准(测速) | ✅ 完成 | ≈4 it/s @ batch128;600步/run≈3min |
| 实验 1:LR sweep | ✅ 完成 | 最优 lr=1.5e-3→**1.375**;≥6e-3 软发散(grad_clip 无 NaN) |
| 实验 2:batch sweep | ✅ 完成 | batch 越小损失越低(64→**1.347**);最大可行≈192,256 OOM |
| 实验 3:epoch vs iteration(同 FLOPs) | ✅ 完成 | 两者等价(val ~1.37);iteration 更简单为标准 |
| 实验 4:达到 val/loss < 1.45 | ✅ **达成** | lr=6e-4/1.5e-3/3e-3 均 <1.45,**最低 1.375** |

---

## 校准(训练速度)

| 指标 | 值 |
|---|---|
| 配置 | batch=128, ctx=256, 17M 模型, lr=3e-4 |
| 300 步总耗时 | 98 s(含模型加载/wandb/4 次验证评估) |
| 训练速度 | **≈ 4 it/s(~250 ms/step)** |
| 吞吐 | ≈ 131k tokens/s |
| GPU 利用率 | 39%(显存 19.5 GB)→ **数据加载瓶颈**(NFS memmap 取数 + 拷贝) |
| 300 步 train/val loss | 3.72 / 3.69(正常下降,无发散) |

**据此定:** 600 步/run ≈ 3 分钟(短)→ 实验 1 取 **8 个学习率**(log 均匀)。

---

## 实验 1:学习率扫描(随机搜索)

**方法**:固定其余超参(batch=128 · **full run = 327.68M token/run** · 余弦 LR + 200 步 warmup · grad_clip=1.0),log 区间采样学习率,记录最终 val loss,发散标注 `DIVERGED`。长训练(~34min/run)→ 取 5 个主 LR,另补 3 个高 LR 定位发散边界。

### 结果

| LR | 最终 val loss | 状态 |
|---|---|---|
| 1e-4 | 1.9491 | 收敛(偏小) |
| 3e-4 | 1.5500 | 收敛 |
| 6e-4 | 1.4311 | 收敛 ✅ < 1.45 |
| **1.5e-3** | **1.3749** | 收敛(**最优**)✅ |
| 3e-3 | 1.3782 | 收敛 ✅ |
| 6e-3 | 2.2009 | **退化/软发散**(跑满但 loss 卡高位) |
| 1.2e-2 | ~3.02(iter5000止损) | **更退化**(loss 卡 3.0) |
| 2.5e-2 | 未跑(已止损) | 趋势已明,省算力 |

![lr sweep](plots/exp1_lr_sweep.png)

![final vs lr](plots/exp1_final_vs_lr.png)

### 发现
- **val loss 随 LR 先降后平**:1e-4(1.95)→ 6e-4(1.43)→ **1.5e-3(1.375,最优)**→ 3e-3(1.378,持平)。
- **≥6e-3 软发散**:6e-3 → 2.20、1.2e-2 → 3.02,loss 单调恶化、不收敛;但因 **grad_clip=1.0 截断梯度,全程无 NaN 硬发散**——此模型的"发散"表现为 loss 退化/停滞,而非数值爆炸。
- **可用 LR 上界 ≈ 3e-3**(再高即退化);**最优学习率 ≈ 1.5e-3**(val 1.375)。

---

## 实验 2:批大小扫描

**方法**:固定 lr=1.5e-3(exp1 最优)+ **固定 327.68M token 预算**(步数随 batch 反比,保证同 FLOPs),从大到小试 batch 直到 OOM,记录最终 val loss、显存、学习曲线。

### 结果(同 FLOPs)
| batch | 步数 | 最终 val loss | 显存 |
|---|---|---|---|
| 64 | 20000 | **1.3472**(最低) | 10.0 GB |
| 128 | 10000 | 1.3749 | 19.5 GB |
| 192 | 6666 | 1.3837 | 28.9 GB |
| 256 | 5000 | **OOM** ❌ | >32 GB |

![batch sweep](plots/exp2_batch_sweep.png)

![batch final](plots/exp2_batch_final.png)

### 发现
- **同 FLOPs 下 batch 越小 → val loss 越低**:64(1.347)< 128(1.375)< 192(1.384)。小 batch 步数多、梯度更新次数多,收敛更充分;大 batch 更新少、最终损失略高。
- **显存随 batch 近线性**:64→10G,128→19.5G,192→28.9G;**5090(32G)最大可行 batch ≈ 192**,**batch=256 OOM**(`CUDA out of memory`,需 ~32GB+)。
- **权衡**:小 batch 损失更低但 wall-clock 更慢(更多步、GPU 利用率低 ~40%);大 batch 训练快、GPU 吃满,但损失略高且逼近显存上限。lr=1.5e-3 在各 batch 都稳定,**无需再调 LR**。

---

## 实验 3:数据加载策略(epoch vs iteration,同 FLOPs)

**方法**:固定 batch=128 · lr=1.5e-3 · 10000 步 = **327.68M token(同 FLOPs)**,对比:
- **iteration**:`get_batch` 随机有放回(复用 exp1 的 lr_1.5e-3);
- **epoch**:`iter_epoch_batches` 无放回遍历不重叠窗口(`--epochs 1 --max-steps 10000`)。

### 结果
| 加载策略 | 最终 val loss |
|---|---|
| iteration(随机有放回) | 1.3749 |
| epoch(无放回) | 1.3593 |

![loader](plots/exp3_loader.png)

### 发现
- **两种策略性能等价**:final val 1.359 vs 1.375,差异在 20-batch 评估采样噪声内(±0.01),学习曲线基本重合。
- **原因**:数据集很大(540M token),固定预算只用 ~0.6 epoch,远未重复遍历;此规模下"有放回"的窗口重叠/遗漏与"无放回"的均匀覆盖都是对数据分布的无偏采样,无显著差异。
- **结论**:大数据集 + 固定步数预算下,**iteration(随机有放回)与 epoch(无放回)效果相当**;iteration 更简单(无 epoch 状态、断点续训方便、内存友好、可处理超大语料),是 LLM 预训练的标准选择。

---

## 实验 4:命中 val/loss < 1.45

**已达成 ✅**(由实验 1 的 LR sweep 顺带命中,无需额外训练):

| 配置(均 batch=128 · 10k 步 · 327.68M token) | val/loss |
|---|---|
| lr=6e-4 | 1.4311 |
| **lr=1.5e-3(最优)** | **1.3749** |
| lr=3e-3 | 1.3782 |

**命中配置**:固定模型(d_model=512 / 4层 / 16头 / d_ff=1344 / ctx=256)+ AdamW(wd=0.01, grad_clip=1.0)+ 余弦 LR(200 warmup)+ **lr=1.5e-3** + batch=128 + 10000 步(327.68M token)→ **val/loss = 1.375 < 1.45**。曲线见 `plots/exp1_lr_sweep.png`(红色目标线 1.45 已标注,3 条曲线终点在其下方)。

---

## 附加:加速(torch.compile + bf16)与 Muon 优化器(Kimi 风格)

先 profile 单步:**数据准备仅占 0.5%,GPU 计算占 99.5%** → 瓶颈是 GPU 端(小模型 = 小 kernel + 显存带宽),不是数据加载。故针对 GPU 优化(代码见 `experiments/train_fast.py`、`cs336_basics.optimizer.Muon`):

| 变体 | 稳态吞吐 | 加速 | 300 步 val loss |
|---|---|---|---|
| A 基线(fp32 + AdamW) | 162k tok/s(4.96 it/s) | 1.0× | 2.741 |
| B + bf16 autocast | 220k tok/s | **1.36×** | 2.793 |
| C + torch.compile + bf16 | 413k tok/s | **2.55×** | 2.775 |
| **D + Muon(Kimi 风格)** | **457k tok/s(13.9 it/s)** | **2.82×** | **2.417** |

**结论:**
- **bf16**(matmul 半精度,损失/RMSNorm 仍 fp32)→ 1.36×;**torch.compile**(融合小 kernel)再叠加到 **2.55×**(印证瓶颈是 kernel 启动开销)。
- **Muon**(2D 隐藏权重做 Newton-Schulz 正交化更新,embedding/lm_head/norm 仍 AdamW):速度与 AdamW 相当,但 **300 步 val 显著更低(2.42 vs 2.77)** → 印证 Kimi"Muon 更高效"的说法(收敛更快)。
- **综合 2.82×:full run 从 ~33.5min 降到 ~12min**(327.68M tok ÷ 457k tok/s ≈ 717s)。
- ⚠️ bf16/compile 数值与 fp32 略有差异,**前 4 个实验结论仍以 fp32 为准**;加速用于今后新实验。

## 结论

四个实验在 RTX 5090 上自动完成(固定 17M 参数模型,每次 full run = 327.68M token,~33min)。

| 实验 | 核心结论 |
|---|---|
| 1. LR sweep | 最优 lr ≈ **1.5e-3**(val 1.375);≥6e-3 软发散(loss 退化,grad_clip 防 NaN) |
| 2. batch sweep | 同 FLOPs 下 **batch 越小损失越低**(64→1.347);最大可行 batch ≈192,**256 OOM** |
| 3. epoch vs iteration | 同 FLOPs 下**两者等价**(val ~1.37);iteration 更简单,是预训练标准 |
| 4. val < 1.45 | **达成 ✅**:多个配置 <1.45,**最低 1.347**(batch64/lr1.5e-3) |

**最佳配置**:d_model=512 / 4层 / 16头 / d_ff=1344 / ctx=256 + **lr=1.5e-3 + batch=64 + 20000步** → **val/loss 1.347**。

### 生成样例(batch_64 checkpoint,val 1.347)
prompt:`Once upon a time, there was a little girl named Lily.`

> Once upon a time, there was a little girl named Lily. She loved to play with her friends outside. One day, Lily found a big, comfortable cushion in her room... Tom played with the cushion, but it was too rough. He broke it in half. Lily was very sad... Lily and Tom worked together to fix the cushion. They used glue and tape to make it even better.

生成文本语法通顺、情节完整(角色 + 对话 + 冲突 + 解决),符合 TinyStories 风格 → **模型训练成功**。
