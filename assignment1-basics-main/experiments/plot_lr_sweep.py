"""解析 experiments/runs/lr_*.log,画 LR sweep 学习曲线 + 最终损失 vs LR。
用法(assignment 目录下): uv run python experiments/plot_lr_sweep.py
输出: experiments/plots/exp1_lr_sweep.png, exp1_final_vs_lr.png
"""
import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_DIR, OUT_DIR = "experiments/runs", "experiments/plots"
os.makedirs(OUT_DIR, exist_ok=True)
LRS = ["1e-4", "3e-4", "6e-4", "1.5e-3", "3e-3"]   # sweep 的 5 个学习率

VAL_RE = re.compile(r"iter\s+(\d+)\s+\|\s+val/loss\s+([\d.]+|nan|inf)")


def parse(logfile):
    vi, vl, diverged = [], [], False
    with open(logfile) as f:
        for line in f:
            if "DIVERGED" in line:
                diverged = True
            m = VAL_RE.search(line)
            if m:
                vi.append(int(m.group(1)))
                vl.append(float(m.group(2)) if m.group(2) not in ("nan", "inf") else float("nan"))
    return vi, vl, diverged


# ---- 图1: 各 LR 的验证损失学习曲线 ----
plt.figure(figsize=(9, 6))
finals = []
for lr in LRS:
    log = os.path.join(RUN_DIR, f"lr_{lr}.log")
    if not os.path.exists(log):
        continue
    vi, vl, div = parse(log)
    if not vi:
        continue
    final = vl[-1]
    finals.append((lr, final, div))
    label = f"lr={lr}" + (" [DIVERGED]" if div else f" (val={final:.3f})")
    plt.plot(vi, vl, marker="o", ms=3, label=label)
plt.axhline(1.45, color="r", ls="--", lw=1, label="target 1.45")
plt.xlabel("iteration"); plt.ylabel("val loss")
plt.title("TinyStories LR sweep - val loss curves (full run = 327.68M tokens/run)")
plt.legend(); plt.grid(True, alpha=0.3); plt.ylim(1.2, 4.0)
plt.savefig(os.path.join(OUT_DIR, "exp1_lr_sweep.png"), dpi=120, bbox_inches="tight")
print("saved exp1_lr_sweep.png；finals =", finals)

# ---- 图2: 最终验证损失 vs 学习率 ----
plt.figure(figsize=(7, 5))
conv = [(float(lr), fl) for lr, fl, div in finals if not div and fl == fl]  # 非发散且非 nan
if conv:
    xs, ys = zip(*conv)
    plt.plot(xs, ys, marker="o", color="tab:blue")
    for x, y in conv:
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 6), fontsize=8)
for lr, fl, div in finals:
    if div or fl != fl:
        plt.scatter([float(lr)], [4.0], marker="x", color="red")
        plt.annotate("DIVERGED", (float(lr), 4.0), color="red", fontsize=8)
plt.axhline(1.45, color="r", ls="--", lw=1, label="target 1.45")
plt.xscale("log"); plt.xlabel("learning rate (log)"); plt.ylabel("final val loss")
plt.title("final val loss vs learning rate"); plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUT_DIR, "exp1_final_vs_lr.png"), dpi=120, bbox_inches="tight")
print("saved exp1_final_vs_lr.png")
