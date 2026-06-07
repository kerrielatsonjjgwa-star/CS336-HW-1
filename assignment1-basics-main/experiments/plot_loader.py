"""画实验3:epoch(无放回) vs iteration(随机有放回) 数据加载策略,同 FLOPs 下的验证损失曲线。
用法(assignment 目录下): uv run python experiments/plot_loader.py
"""
import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_DIR, OUT_DIR = "experiments/runs", "experiments/plots"
os.makedirs(OUT_DIR, exist_ok=True)
VAL_RE = re.compile(r"(?:iter|step)\s+(\d+)\s+\|\s+val/loss\s+([\d.]+)")


def parse(log):
    xs, ys = [], []
    for line in open(log):
        m = VAL_RE.search(line)
        if m:
            xs.append(int(m.group(1))); ys.append(float(m.group(2)))
    return xs, ys


plt.figure(figsize=(9, 6))
runs = [
    ("lr_1.5e-3", "iteration (random, with replacement)"),
    ("loader_epoch", "epoch (shuffled, no replacement)"),
]
for name, label in runs:
    log = os.path.join(RUN_DIR, name + ".log")
    if not os.path.exists(log):
        continue
    xs, ys = parse(log)
    if xs:
        plt.plot(xs, ys, marker="o", ms=3, label=f"{label}  (final val={ys[-1]:.4f})")
plt.axhline(1.45, color="r", ls="--", lw=1, label="target 1.45")
plt.xlabel("step"); plt.ylabel("val loss")
plt.title("Exp3: data loader @ same FLOPs (batch128, 10k steps = 327.68M tokens, lr=1.5e-3)")
plt.legend(); plt.grid(True, alpha=0.3); plt.ylim(1.3, 3.2)
plt.savefig(os.path.join(OUT_DIR, "exp3_loader.png"), dpi=120, bbox_inches="tight")
print("saved exp3_loader.png")
