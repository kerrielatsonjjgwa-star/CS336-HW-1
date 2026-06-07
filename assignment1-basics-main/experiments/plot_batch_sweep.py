"""画实验2 batch sweep:val loss vs 已处理 token(同 327.68M 预算)+ 最终损失/显存 vs batch。
用法(assignment 目录下): uv run python experiments/plot_batch_sweep.py
"""
import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_DIR, OUT_DIR = "experiments/runs", "experiments/plots"
os.makedirs(OUT_DIR, exist_ok=True)
CTX = 256
RUNS = {64: "batch_64", 128: "lr_1.5e-3", 192: "batch_192"}   # 128 复用 exp1
MEM = {64: 10.0, 128: 19.5, 192: 28.9, 256: float("nan")}      # GB(256 OOM)
VAL_RE = re.compile(r"iter\s+(\d+)\s+\|\s+val/loss\s+([\d.]+)")

# ---- 图1: val loss vs 已处理 token ----
plt.figure(figsize=(9, 6))
finals = {}
for b in sorted(RUNS):
    log = os.path.join(RUN_DIR, RUNS[b] + ".log")
    if not os.path.exists(log):
        continue
    its, vls = [], []
    for line in open(log):
        m = VAL_RE.search(line)
        if m:
            its.append(int(m.group(1))); vls.append(float(m.group(2)))
    if not its:
        continue
    tokens = [it * b * CTX / 1e6 for it in its]   # 百万 token
    finals[b] = vls[-1]
    plt.plot(tokens, vls, marker="o", ms=3, label=f"batch={b} (val={vls[-1]:.3f})")
plt.axhline(1.45, color="r", ls="--", lw=1, label="target 1.45")
plt.xlabel("tokens processed (M)"); plt.ylabel("val loss")
plt.title("Batch sweep - val loss vs tokens (fixed 327.68M budget, lr=1.5e-3)")
plt.legend(); plt.grid(True, alpha=0.3); plt.ylim(1.3, 3.2)
plt.savefig(os.path.join(OUT_DIR, "exp2_batch_sweep.png"), dpi=120, bbox_inches="tight")
print("saved exp2_batch_sweep.png; finals =", finals)

# ---- 图2: 最终 val loss + 显存 vs batch ----
fig, ax1 = plt.subplots(figsize=(7, 5))
bs = sorted(finals)
ax1.plot(bs, [finals[b] for b in bs], "o-", color="tab:blue", label="final val loss")
for b in bs:
    ax1.annotate(f"{finals[b]:.3f}", (b, finals[b]), textcoords="offset points", xytext=(0, 6), fontsize=8)
ax1.set_xlabel("batch size"); ax1.set_ylabel("final val loss", color="tab:blue")
ax2 = ax1.twinx()
mb = [b for b in [64, 128, 192] if b in MEM]
ax2.plot(mb, [MEM[b] for b in mb], "s--", color="tab:red", label="GPU mem (GB)")
ax2.axhline(32, color="gray", ls=":", lw=1)
ax2.annotate("batch=256 OOM (>32GB)", (192, 30), color="red", fontsize=8)
ax2.set_ylabel("GPU memory (GB)", color="tab:red")
plt.title("final val loss & GPU memory vs batch size")
fig.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "exp2_batch_final.png"), dpi=120, bbox_inches="tight")
print("saved exp2_batch_final.png")
