# -*- coding: utf-8 -*-
"""
生成 CS336 作业一（基础）引导式 Jupyter 笔记本。
风格参考 CS231n assignment2：Markdown 讲解 + 代码/测试 cell + Inline Question/解答 块。
本笔记本只「引导」，不提供任何题解实现（遵守作业 AI 政策）。
输出：assignment1-basics-main/CS336_Assignment1.ipynb
"""
import json
import os

ROOT = "/home/zhangzt22/CS336/assignment1-basics-main"
OUT = os.path.join(ROOT, "CS336_Assignment1.ipynb")

cells = []
_counter = [0]


def _cid():
    _counter[0] += 1
    return f"cell{_counter[0]:03d}"


def _src(s):
    s = s.strip("\n")
    lines = s.split("\n")
    if not lines:
        return [""]
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def md(s):
    cells.append({"cell_type": "markdown", "id": _cid(), "metadata": {}, "source": _src(s)})


def code(s):
    cells.append({
        "cell_type": "code",
        "id": _cid(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _src(s),
    })


def answer(extra=""):
    body = "## 解答：\n\n> _在此填写你的书面解答（提交时请排版到 writeup.pdf）。_"
    if extra:
        body += "\n>\n> " + extra
    md(body)


# =====================================================================
# 0. 封面 / 总览
# =====================================================================
md(r"""
**生成式 AI 使用说明（重要）**：CS336 作业一**不允许**用 AI 工具实现作业的任何部分（包括编码代理与 AI 自动补全）。本笔记本仅作为**引导与自测脚手架**：它告诉你每道题要改哪个文件、对应跑哪个测试、书面题要回答什么——所有实现与书面解答都必须由你**独立完成**。允许用 AI 询问高层概念或库 API 文档；若使用聊天机器人，请在 `writeup.pdf` 中附上所用提示词。
""")

md(r"""
# CS336 作业一（基础）：从零构建 Transformer 语言模型 — 引导笔记本

> 版本 26.0.3 · 2026 春季 · 配套讲义：`cs336_assignment1_basics_zh.pdf`

本笔记本按讲义顺序，把**全部题目（书面题 + 代码题 + 实验）**串成一条可执行的路线。每道代码题的范式与 CS231n 一致：

1. 在 `cs336_basics/*.py`（以及个别 `tests/adapters.py` 胶水）中补全标了 `# TODO ... # END OF YOUR CODE` 的实现；
2. 回到本笔记本运行对应的测试 cell（底层调用 `uv run pytest -k <测试名>`）；
3. 全绿即通过。书面题在 `## 解答：` 块作答，实验题填写学习曲线与结论。

**你要实现的代码文件**（仅这些，测试文件勿改）：

| 文件 | 内容 | 讲义 |
|---|---|---|
| `cs336_basics/tokenizer.py` | `train_bpe` 函数、`Tokenizer` 类 | §2 |
| `cs336_basics/model.py` | `Linear`/`Embedding`/`RMSNorm`/`SwiGLU`/`RoPE`/注意力/`TransformerBlock`/`TransformerLM` | §3 |
| `cs336_basics/nn_utils.py` | `softmax`/`silu`/`cross_entropy`/`gradient_clipping` | §3.4.4/§4 |
| `cs336_basics/optimizer.py` | `AdamW`、`get_lr_cosine_schedule` | §4 |
| `cs336_basics/data.py` | `get_batch` | §5.1 |
| `cs336_basics/serialization.py` | `save_checkpoint`/`load_checkpoint` | §5.2 |
| `cs336_basics/train.py` / `decode.py` / `run_bpe_training.py` | 可运行脚本（训练/采样/BPE CLI） | §5.3/§6/§2.5 |

**目录**

- §2 BPE 分词器：`unicode1` · `unicode2` · `train_bpe` · `train_bpe_tinystories` · `train_bpe_expts_owt` · `tokenizer` · `tokenizer_experiments`
- §3 Transformer 架构：`linear` · `embedding` · `rmsnorm` · `swiglu` · `rope` · `softmax` · `scaled_dot_product_attention` · `multihead_self_attention` · `transformer_block` · `transformer_lm` · `transformer_accounting`
- §4 训练组件：`cross_entropy` · `learning_rate_tuning` · `adamw` · `adamw_accounting` · `learning_rate_schedule` · `gradient_clipping`
- §5 训练循环：`data_loading` · `checkpointing` · `training_together`
- §6 文本生成：`decoding`
- §7 实验与排行榜：`experiment_log` · `learning_rate` · `batch_size_experiment` · `generate` · `layer_norm_ablation` · `pre_norm_ablation` · `no_pos_emb` · `swiglu_ablation` · `main_experiment` · `leaderboard`
""")

# ---------------------------------------------------------------------
# 环境 setup
# ---------------------------------------------------------------------
md(r"""
## 0. 环境与测试脚手架

先运行下面三个 cell：切换工作目录、定义测试运行器、做环境自检。`autoreload` 会在你修改 `cs336_basics/*.py` 后自动重载，无需重启内核。
""")

code(r"""
# 本地环境 setup：切到作业根目录 + 打开 autoreload
import os, sys
ASSIGNMENT_ROOT = "/home/zhangzt22/CS336/assignment1-basics-main"
os.chdir(ASSIGNMENT_ROOT)
if ASSIGNMENT_ROOT not in sys.path:
    sys.path.insert(0, ASSIGNMENT_ROOT)
%load_ext autoreload
%autoreload 2
print("CWD =", os.getcwd())
""")

code(r"""
# 测试运行器：底层用 `uv run pytest`，与作业 README 一致。
# 若你的 Jupyter 内核就是本项目的 .venv，可把 USE_UV=False 改用 python -m pytest（更快）。
import subprocess, shlex, sys

USE_UV = True

def run_test(k=None, path="tests", timeout=3600, extra=None):
    # 运行 pytest。k=测试名(支持 -k 表达式)；path 可指定单个测试文件。
    base = ["uv", "run", "pytest"] if USE_UV else [sys.executable, "-m", "pytest"]
    cmd = base + [path, "-q", "--no-header", "-p", "no:cacheprovider"]
    if k:
        cmd += ["-k", k]
    if extra:
        cmd += list(extra)
    print(">>", " ".join(shlex.quote(c) for c in cmd), "\n")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print("!! 超时（实现可能死循环 / 数据太大）"); return
    out = (p.stdout or "") + (p.stderr or "")
    print(out[-6000:])
    print("\n=>", "✅ PASSED" if p.returncode == 0 else f"❌ FAILED (exit {p.returncode})")

def rel_error(x, y):
    import numpy as np
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    return np.max(np.abs(x - y) / np.maximum(1e-8, np.abs(x) + np.abs(y)))

print("run_test / rel_error 就绪")
""")

code(r"""
# 环境自检
import torch, numpy as np, importlib.util
print("torch", torch.__version__, "| numpy", np.__version__)
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("默认计算设备 DEVICE =", DEVICE)
for pkg in ["regex", "tiktoken", "einops", "einx", "jaxtyping", "tqdm"]:
    print(f"  {pkg:10s}:", "OK" if importlib.util.find_spec(pkg) else "缺失(uv sync)")
# 初次运行 pytest 时所有未实现处都会以 NotImplementedError 失败，这是预期的起点。
""")

md(r"""
### 下载数据集（运行实验前需要）

TinyStories 与 OpenWebText 的小样本。**只在你要做 §2.5 之后的实验时才需要**（前面的单元测试用 `tests/fixtures/` 里的小样本即可）。

```bash
mkdir -p data && cd data
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz && gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz && gunzip owt_valid.txt.gz
cd ..
```

> 在终端用 `!` 前缀也可在本会话执行，例如 `!mkdir -p data`。
""")

# =====================================================================
# §2 BPE 分词器
# =====================================================================
md(r"""
# 第 2 节 · 字节对编码（BPE）分词器

我们要训练并实现一个**字节级** BPE 分词器：任意 Unicode 字符串 → UTF-8 字节序列 → 在字节上做 BPE 合并。初始词表是 256 个字节，外加特殊 token。
""")

# --- unicode1 ---
md(r"""
## 题目 (unicode1)：理解 Unicode（1 分）· 书面

`ord()` 把单字符转码位整数，`chr()` 反过来。回答：

- **(a)** `chr(0)` 返回什么 Unicode 字符？（一句话）
- **(b)** 这个字符的 `__repr__()` 表示与它的「打印表示」有何不同？（一句话）
- **(c)** 当它出现在文本中时会发生什么？（一句话）

先用下面的 cell 亲手试一试，再作答。
""")
code(r"""
print(repr(chr(0)))
print(chr(0))
print(repr("this is a test" + chr(0) + "string"))
print("this is a test" + chr(0) + "string")
print("ord('牛') =", ord("牛"), "| chr(29275) =", chr(29275))
""")
answer()

# --- unicode2 ---
md(r"""
## 题目 (unicode2)：Unicode 编码（3 分）· 书面

- **(a)** 为什么我们更倾向在 **UTF-8** 字节上训练分词器，而非 UTF-16/UTF-32？（比较不同编码对各种字符串的输出会有帮助）
- **(b)** 下面这个「逐字节解码」函数为什么是错的？给出一个会产生错误结果的输入字节串。
- **(c)** 给出一个**不能**解码为任何 Unicode 字符的双字节序列，并解释。
""")
code(r"""
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

s = "hello! こんにちは!"
print("utf-8 :", list(s.encode("utf-8")), "len", len(s.encode("utf-8")))
print("utf-16:", list(s.encode("utf-16")), "len", len(s.encode("utf-16")))
print("utf-32:", list(s.encode("utf-32")), "len", len(s.encode("utf-32")))
print(decode_utf8_bytes_to_str_wrong("hello".encode("utf-8")))
# (b) 试一个非 ASCII 字符，例如 "牛"，看看逐字节 decode 会怎样
# (c) 试 bytes([0xff, 0xff]).decode("utf-8") 会发生什么？
""")
answer()

# --- bpe_example ---
md(r"""
## 示例 (bpe_example)：BPE 训练流程（不计分，务必读懂）

语料 `low×5, lower×2, widest×3, newest×6`，预分词按空白切分，频率表
`{(l,o,w):5,(l,o,w,e,r):2,(w,i,d,e,s,t):3,(n,e,w,e,s,t):6}`。

合并规则要点：统计**相邻字节对**频率求和 → 取最高频；**平局时取字典序更大者**（如 `('s','t')` 胜 `('e','s')`）。前 6 次合并得到词表项 `st, est, ow, low, west, ne`，于是 `newest → [ne, west]`。这是你接下来要在 `train_bpe` 里实现的逻辑。
""")
code(r"""
# 平局取字典序更大者，对应 Python max 的天然行为：
print(max([("A","B"), ("A","C"), ("B","ZZ"), ("BA","A")]))  # ('BA', 'A')
""")

# --- train_bpe ---
md(r"""
## 题目 (train_bpe)：实现 BPE 训练（15 分）· 代码

实现 `train_bpe(input_path, vocab_size, special_tokens) -> (vocab, merges)`：

- **输入**：`input_path: str`、`vocab_size: int`（含 256 字节基底 + 合并 + 特殊 token）、`special_tokens: list[str]`（视为**硬边界**，不跨其合并，且**不计入**合并统计）。
- **输出**：`vocab: dict[int, bytes]`、`merges: list[tuple[bytes, bytes]]`（按创建顺序）。

实现步骤：① 词表初始化（256 字节 + 特殊 token）；② 预分词——先用 `re.split` 在特殊 token 处切段，再用 GPT-2 正则 `re.finditer` 统计预 token 频率；③ 计算合并——对相邻字节对计数取最高频（**平局取字典序更大**），增量更新计数以提速。预分词建议用 `multiprocessing` 并行（参考 `cs336_basics/pretokenization_example.py` 取 chunk 边界）。

> 📝 **编辑**：`cs336_basics/tokenizer.py::train_bpe`（胶水 `tests/adapters.py::run_train_bpe` 已写好）。
""")
md(r"""GPT-2 预分词正则（直接复制使用）：""")
code(r"""
PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
import regex as re
print(re.findall(PAT, "some text that i'll pre-tokenize"))
""")
md(r"""✅ **测试**（含速度测试与特殊 token 边界测试）：""")
code(r"""run_test("test_train_bpe or test_train_bpe_special_tokens", path="tests/test_train_bpe.py")""")

# --- train_bpe_tinystories ---
md(r"""
## 题目 (train_bpe_tinystories)：在 TinyStories 上训练 BPE（2 分）· 实验

- **(a)** 在 TinyStories 上训练词表大小 **10,000** 的字节级 BPE（特殊 token `<|endoftext|>`），序列化到磁盘。**用时多少？峰值内存多少？词表中最长的 token 是什么？它有意义吗？**（资源上限：≤30 分钟、≤30GB）
- **(b)** 对代码做性能剖析（`cProfile`），训练中**哪一步最耗时**？

> 提示：`<|endoftext|>` 分隔文档、且在合并前作为特殊情形处理；并行预分词可把训练压到 2 分钟内。先在**验证集**（22K 文档）上调试再上全量。
""")
code(r"""
# 在 TinyStories 验证集上训练（小、快，便于调试）；正式实验可换 train 文件。
import time, cProfile, pstats, io
from cs336_basics.tokenizer import train_bpe

INPUT = "data/TinyStoriesV2-GPT4-valid.txt"   # 没下数据时，可先用 tests/fixtures/tinystories_sample.txt
t0 = time.time()
vocab, merges = train_bpe(INPUT, vocab_size=10000, special_tokens=["<|endoftext|>"])
print(f"用时 {time.time()-t0:.1f}s | vocab={len(vocab)} | merges={len(merges)}")
longest = max(vocab.values(), key=len)
print("最长 token:", longest, "(len", len(longest), ")")

# 序列化（自行实现可逆编码，见 run_bpe_training.save_vocab_and_merges 的 TODO）
# import json, base64
# json.dump({str(i): base64.b64encode(b).decode() for i,b in vocab.items()}, open("ts_vocab.json","w"))

# (b) 剖析示例：
# cProfile.run("train_bpe(INPUT, 10000, ['<|endoftext|>'])", "bpe.prof")
# pstats.Stats("bpe.prof").sort_stats("cumtime").print_stats(15)
""")
answer()

# --- train_bpe_expts_owt ---
md(r"""
## 题目 (train_bpe_expts_owt)：在 OpenWebText 上训练 BPE（2 分）· 实验

- **(a)** 在 OWT 上训练词表大小 **32,000** 的 BPE 并序列化。**最长 token 是什么？有意义吗？**（上限：≤12 小时、≤100GB）
- **(b)** 对比 TinyStories（10K）与 OWT（32K）两个分词器的异同。
""")
code(r"""
# 资源消耗较大，建议在终端用脚本跑：
# !uv run python -m cs336_basics.run_bpe_training --input data/owt_train.txt \
#     --vocab-size 32000 --special-tokens "<|endoftext|>" \
#     --vocab-out owt_vocab.json --merges-out owt_merges.txt
print("见上方命令；完成后在此对比两个词表的最长 token / 风格差异。")
""")
answer()

# --- tokenizer ---
md(r"""
## 题目 (tokenizer)：实现 Tokenizer（编码/解码）（15 分）· 代码

实现 `Tokenizer` 类（从 vocab + merges 构造），把文本 ↔ token id 互转，并支持用户自定义特殊 token：

- `__init__(self, vocab, merges, special_tokens=None)`
- `from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)`（类方法）
- `encode(self, text) -> list[int]`：预分词 → 每个预 token 内**按合并创建顺序**应用合并 → 查表得 id；特殊 token 整体成单 id。
- `encode_iterable(self, iterable) -> Iterator[int]`：对文件句柄等惰性逐 id 产出（大文件内存友好）。
- `decode(self, ids) -> str`：查表拼接字节再 `bytes.decode("utf-8", errors="replace")`（非法字节用 U+FFFD）。

> 📝 **编辑**：`cs336_basics/tokenizer.py::Tokenizer`（胶水 `tests/adapters.py::get_tokenizer` 已写好）。该文件测试很多（往返一致性、与 `tiktoken` 对齐、特殊 token、内存占用等），整文件跑：
""")
code(r"""run_test(path="tests/test_tokenizer.py")""")

# --- tokenizer_experiments ---
md(r"""
## 题目 (tokenizer_experiments)：分词器实验（4 分）· 实验

- **(a)** 从 TinyStories / OWT 各采 10 个文档，用各自分词器（10K / 32K）编码。每个分词器的**压缩比**（字节/token）是多少？
- **(b)** 用 **TinyStories 分词器**编码 **OWT 样本**会怎样？比较压缩比并定性描述。
- **(c)** 估计分词器**吞吐量**（字节/秒）。给 Pile（825GB）分词要多久？
- **(d)** 用各自分词器把训练集/开发集编码为 token id 序列（`np.uint16`）存盘，供后续训练。
""")
code(r"""
from cs336_basics.tokenizer import Tokenizer
import numpy as np, time

# tok = Tokenizer.from_files("ts_vocab.json", "ts_merges.txt", special_tokens=["<|endoftext|>"])
# text = open("data/TinyStoriesV2-GPT4-valid.txt").read()[:1_000_000]
# t0=time.time(); ids = tok.encode(text); dt=time.time()-t0
# print("压缩比 字节/token =", len(text.encode('utf-8'))/len(ids))
# print("吞吐量 字节/s     =", len(text.encode('utf-8'))/dt)
# print("Pile 825GB 预计   =", 825*1024**3 / (len(text.encode('utf-8'))/dt) / 3600, "小时")

# (d) 用 encode_iterable 流式编码整文件并存 uint16：
# with open("data/TinyStoriesV2-GPT4-train.txt") as f:
#     arr = np.fromiter(tok.encode_iterable(f), dtype=np.uint16)
# np.save("ts_train.npy", arr)
print("取消注释并填好 vocab/merges 路径后运行。")
""")
answer()

# =====================================================================
# §3 Transformer 架构
# =====================================================================
md(r"""
# 第 3 节 · Transformer 语言模型架构

输入 `(batch, seq)` token id → 词嵌入 → `num_layers` 个**预归一化**块（RMSNorm → 因果多头注意力(带 RoPE) → 残差；RMSNorm → SwiGLU → 残差）→ 最终 RMSNorm → LM 头 → `(batch, seq, vocab)` logits。

**实现约定**：行主序，`Linear` 权重形状 `(out, in)`，计算 $y = xW^\top$；只用 `nn.Parameter` 与 `nn` 容器，**禁用** `nn.Linear`/`nn.LayerNorm`/`nn.functional`。强烈建议用 `einops.einsum`/`rearrange` 表达带命名维度的张量运算。初始化：Linear 截断正态 $\mathcal N(0, 2/(d_{in}+d_{out}))$ 截断于 $[-3\sigma,3\sigma]$；Embedding $\mathcal N(0,1)$ 截断 $[-3,3]$；RMSNorm 全 1。

> 所有 §3 模块都在 `cs336_basics/model.py`（softmax/silu 在 `cs336_basics/nn_utils.py`），胶水 `tests/adapters.py` 已写好。
""")

md(r"""
## 题目 (linear)：实现 Linear（1 分）· 代码

无偏置线性层，权重 `(out_features, in_features)`，前向 $y = xW^\top$，截断正态初始化。
📝 编辑 `model.py::Linear`。
""")
code(r"""run_test("test_linear", path="tests/test_model.py")""")

md(r"""
## 题目 (embedding)：实现 Embedding（1 分）· 代码

嵌入查表层，权重 `(num_embeddings, embedding_dim)`，对 `(batch, seq)` 的 id 索引返回 `(batch, seq, d_model)`。**禁用** `nn.Embedding`。📝 编辑 `model.py::Embedding`。
""")
code(r"""run_test("test_embedding", path="tests/test_model.py")""")

md(r"""
## 题目 (rmsnorm)：实现 RMSNorm（1 分）· 代码

$$\mathrm{RMSNorm}(a_i)=\dfrac{a_i}{\mathrm{RMS}(a)}\,g_i,\quad \mathrm{RMS}(a)=\sqrt{\tfrac{1}{d_{model}}\textstyle\sum_i a_i^2+\varepsilon}\;(\varepsilon=10^{-5})$$

**注意**：求平方前把输入 `upcast` 到 `float32`，算完再 `downcast` 回原 dtype。📝 编辑 `model.py::RMSNorm`。
""")
code(r"""run_test("test_rmsnorm", path="tests/test_model.py")""")

md(r"""
## 题目 (swiglu)：实现 SwiGLU 位置前馈网络（2 分）· 代码

$$\mathrm{SiLU}(x)=x\,\sigma(x),\qquad \mathrm{FFN}(x)=W_2\big(\mathrm{SiLU}(W_1 x)\odot W_3 x\big)$$

形状：`w1,w3: (d_ff, d_model)`、`w2: (d_model, d_ff)`，无偏置。子模块属性名须为 `w1/w2/w3`。
📝 编辑 `nn_utils.py::silu` 与 `model.py::SwiGLU`。
""")
code(r"""run_test("test_silu_matches_pytorch or test_swiglu", path="tests/test_model.py")""")

md(r"""
## 题目 (rope)：实现 RoPE（2 分）· 代码

对位置 $i$、维度对 $k$ 用角度 $\theta_{i,k}=i/\Theta^{(2k-2)/d}$ 做 2×2 旋转。建议在 `__init__` 用 `register_buffer(persistent=False)` 预存 `cos/sin`，`forward(x, token_positions)` 按位置切片。无可学习参数；要容忍任意前导批维度。📝 编辑 `model.py::RotaryPositionalEmbedding`。
""")
code(r"""run_test("test_rope", path="tests/test_model.py")""")

md(r"""
## 题目 (softmax)：实现 softmax（1 分）· 代码

对指定维度做数值稳定 softmax（减去该维最大值）。📝 编辑 `nn_utils.py::softmax`。
""")
code(r"""run_test("test_softmax_matches_pytorch", path="tests/test_nn_utils.py")""")

md(r"""
## 题目 (scaled_dot_product_attention)：缩放点积注意力（5 分）· 代码

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\Big(\tfrac{QK^\top}{\sqrt{d_k}}+\text{mask}\Big)V$$

支持任意前导批维度；可选布尔掩码 `(seq,seq)`：`True` 位允许注意（softmax 后求和为 1），`False` 位概率为 0（实现上对 `False` 位的分数加 $-\infty$）。📝 编辑 `model.py::scaled_dot_product_attention`。
""")
code(r"""run_test("test_scaled_dot_product_attention or test_4d_scaled_dot_product_attention", path="tests/test_model.py")""")

md(r"""
## 题目 (multihead_self_attention)：因果多头自注意力（5 分）· 代码

$d_k=d_v=d_{model}/h$；用**单次**矩阵乘得到全部头的 Q/K/V 投影。因果掩码：位置 $i$ 只能注意 $j\le i$（`torch.triu` 或广播比较）。RoPE 只施加于 Q/K（不含 V），头维当作批维。子模块属性名须为 `q_proj/k_proj/v_proj/output_proj`（带 RoPE 时含 `rope`）。📝 编辑 `model.py::MultiHeadSelfAttention`。
""")
code(r"""run_test("test_multihead_self_attention or test_multihead_self_attention_with_rope", path="tests/test_model.py")""")

md(r"""
## 题目 (transformer_block)：预归一化 Transformer 块（3 分）· 代码

$$y_1 = x + \mathrm{MHSA}(\mathrm{RMSNorm}(x)),\qquad y = y_1 + \mathrm{FFN}(\mathrm{RMSNorm}(y_1))$$

子模块属性名须为 `ln1 / attn(use_rope=True) / ln2 / ffn`。📝 编辑 `model.py::TransformerBlock`。
""")
code(r"""run_test("test_transformer_block", path="tests/test_model.py")""")

md(r"""
## 题目 (transformer_lm)：完整 Transformer 语言模型（3 分）· 代码

词嵌入 → `num_layers` 个块 → 最终 RMSNorm → LM 头（Linear）。子模块属性名须为 `token_embeddings / layers(ModuleList) / ln_final / lm_head`。📝 编辑 `model.py::TransformerLM`。
""")
code(r"""run_test("test_transformer_lm or test_transformer_lm_truncated_input", path="tests/test_model.py")""")

md(r"""
## 题目 (transformer_accounting)：资源核算（5 分）· 书面

GPT-2 XL 形状：`vocab=50257, context=1024, num_layers=48, d_model=1600, num_heads=25, d_ff=4288`。

- **(a)** 可训练参数有多少？单精度仅加载模型需多少内存？
- **(b)** 一次前向（输入 `context_length` 个 token）的全部矩阵乘各需多少 FLOPs，总计多少？规则：$A_{m\times n}B_{n\times p}$ 需 $2mnp$ FLOPs。
""")
code(r"""
# 帮助核对：实现 TransformerLM 后可用它数参数，验证你手算的公式。
import torch
from cs336_basics.model import TransformerLM
cfg = dict(vocab_size=50257, context_length=1024, d_model=1600,
           num_layers=48, num_heads=25, d_ff=4288, rope_theta=10000.0)
m = TransformerLM(**cfg)
n = sum(p.numel() for p in m.parameters())
print(f"参数量 = {n:,} ≈ {n/1e9:.3f} B | float32 内存 ≈ {n*4/1e9:.2f} GB")
# FLOPs 请按 (b) 逐个矩阵乘手算后在解答中列出。
""")
answer("(a) 参数量与内存；(b) 逐矩阵乘 FLOPs 列表与总和。")

# =====================================================================
# §4 训练组件
# =====================================================================
md(r"""
# 第 4 节 · 训练 Transformer 语言模型

损失（交叉熵）+ 优化器（AdamW）+ 学习率调度 + 梯度裁剪。
""")

md(r"""
## 题目 (cross_entropy)：实现交叉熵（1 分）· 代码

$\ell_i=-\log\mathrm{softmax}(o_i)[x_{i+1}]$。数值稳定（减最大值、抵消 $\log\exp$）、处理批维并返回**均值**。困惑度 $=\exp(\frac1m\sum_i\ell_i)$。📝 编辑 `nn_utils.py::cross_entropy`。
""")
code(r"""run_test("test_cross_entropy", path="tests/test_nn_utils.py")""")

md(r"""
## 题目 (learning_rate_tuning)：调节学习率（1 分）· 书面

用讲义里的玩具 SGD（$\theta_{t+1}=\theta_t-\frac{\alpha}{\sqrt{t+1}}\nabla L$），分别用 `lr = 1e1, 1e2, 1e3` 各跑 **10 步**，观察 loss 是更快下降、更慢、还是发散。
""")
code(r"""
import torch, math
class ToySGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        super().__init__(params, {"lr": lr})
    def step(self, closure=None):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None: continue
                st = self.state[p]; t = st.get("t", 0)
                p.data -= g["lr"] / math.sqrt(t + 1) * p.grad.data
                st["t"] = t + 1

for lr in [1e1, 1e2, 1e3]:
    torch.manual_seed(0)
    w = torch.nn.Parameter(5 * torch.randn(10, 10))
    opt = ToySGD([w], lr=lr)
    losses = []
    for _ in range(10):
        opt.zero_grad(); loss = (w ** 2).mean(); loss.backward(); opt.step()
        losses.append(loss.item())
    print(f"lr={lr:>6}: {['%.2e'%x for x in losses]}")
""")
answer()

md(r"""
## 题目 (adamw)：实现 AdamW（2 分）· 代码

按讲义算法 1（$t$ 从 1 起）：$m\leftarrow\beta_1 m+(1-\beta_1)g$，$v\leftarrow\beta_2 v+(1-\beta_2)g^2$，$\alpha_t=\alpha\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}$，$\theta\leftarrow\theta-\alpha_t\frac{m}{\sqrt v+\varepsilon}$，**解耦**权重衰减 $\theta\leftarrow\theta-\alpha\lambda\theta$。状态存 `self.state[p]`。📝 编辑 `optimizer.py::AdamW`。
""")
code(r"""run_test("test_adamw", path="tests/test_optimizer.py")""")

md(r"""
## 题目 (adamw_accounting)：AdamW 训练资源核算（2 分）· 书面

全程 float32，$d_{ff}=\tfrac83 d_{model}$。用 `batch_size` 与模型超参表示：

- **(a)** 峰值内存按**参数 / 激活 / 梯度 / 优化器状态**分解。激活只计：每块的 RMSNorm、MHSA（QKV 投影、$QK^\top$、softmax、加权和 V、输出投影）、SwiGLU（$W_1$、$W_3$ 门控分支的 SiLU、逐元素乘、$W_2$）。
- **(b)**（如讲义续问）给出数值估计 / 随 batch_size 的关系。
""")
answer("按 参数 / 激活 / 梯度 / 优化器状态 四部分写出含 batch_size 与超参的表达式。")

md(r"""
## 题目 (learning_rate_schedule)：余弦调度 + 预热（1 分）· 代码

预热 $t<T_w:\ \alpha_t=\frac{t}{T_w}\alpha_{max}$；退火 $T_w\le t\le T_c:\ \alpha_t=\alpha_{min}+\frac12(1+\cos\frac{t-T_w}{T_c-T_w}\pi)(\alpha_{max}-\alpha_{min})$；之后 $\alpha_t=\alpha_{min}$。📝 编辑 `optimizer.py::get_lr_cosine_schedule`。
""")
code(r"""run_test("test_get_lr_cosine_schedule", path="tests/test_optimizer.py")""")
code(r"""
# 可视化（实现后运行）：
import matplotlib.pyplot as plt
from cs336_basics.optimizer import get_lr_cosine_schedule
xs = range(0, 1000)
ys = [get_lr_cosine_schedule(t, 1e-3, 1e-5, warmup_iters=100, cosine_cycle_iters=900) for t in xs]
plt.plot(list(xs), ys); plt.xlabel("step"); plt.ylabel("lr"); plt.title("cosine schedule"); plt.show()
""")

md(r"""
## 题目 (gradient_clipping)：梯度裁剪（1 分）· 代码

合并所有参数梯度算 $\ell_2$ 范数；若 $>M$，按 $\frac{M}{\|g\|_2+\varepsilon}$（$\varepsilon=10^{-6}$）原地缩放。📝 编辑 `nn_utils.py::gradient_clipping`。
""")
code(r"""run_test("test_gradient_clipping", path="tests/test_nn_utils.py")""")

# =====================================================================
# §5 训练循环
# =====================================================================
md(r"""
# 第 5 节 · 训练循环
""")

md(r"""
## 题目 (data_loading)：实现数据加载（2 分）· 代码

输入 1D token 数组 `x`、`batch_size`、`context_length`、`device`，随机采样返回 `(inputs, targets)`，两者形状均 `(batch_size, context_length)`，`targets` 是 `inputs` 右移一位的下一个 token，放到指定设备。大数据用 `np.memmap` / `np.load(mmap_mode='r')`。📝 编辑 `data.py::get_batch`。
""")
code(r"""run_test("test_get_batch", path="tests/test_data.py")""")

md(r"""
## 题目 (checkpointing)：模型检查点（1 分）· 代码

`save_checkpoint(model, optimizer, iteration, out)`：用 `state_dict()` + `torch.save` 转储模型/优化器/迭代数。`load_checkpoint(src, model, optimizer) -> int`：`torch.load` 后 `load_state_dict` 恢复，返回迭代数。📝 编辑 `serialization.py`。
""")
code(r"""run_test("test_checkpointing", path="tests/test_serialization.py")""")

md(r"""
## 题目 (training_together)：组装训练脚本（4 分）· 代码

`cs336_basics/train.py` 串起：`np.memmap` 加载 → `get_batch` → 前向 + `cross_entropy` → 反向 → `gradient_clipping` → `AdamW.step` + 余弦调度 → 周期 `save_checkpoint` + 记录 train/val 指标。支持命令行超参。📝 编辑 `train.py`（argparse 骨架已就绪）。

先看帮助，再用**调试规模**跑几步确认能收敛（过拟合单 batch）。
""")
code(r"""
# !uv run python -m cs336_basics.train --help
""")
code(r"""
# 小规模冒烟测试（数据需先用 tokenizer_experiments(d) 编码成 .npy）：
# !uv run python -m cs336_basics.train \
#     --train-data ts_train.npy --valid-data ts_valid.npy \
#     --vocab-size 10000 --context-length 256 --d-model 512 \
#     --num-layers 4 --num-heads 16 --d-ff 1344 --rope-theta 10000 \
#     --lr 3e-4 --batch-size 32 --max-iters 200 --warmup 50 \
#     --checkpoint-dir ckpts/smoke --log-interval 10
print("实现 train.py 后取消注释运行；训练损失应能快速下降。")
""")

# =====================================================================
# §6 文本生成
# =====================================================================
md(r"""
# 第 6 节 · 文本生成（解码）

## 题目 (decoding)：实现解码（3 分）· 代码

自回归采样：喂入 prompt，取最后位置 logits → softmax → 采样 → 追加，直到 `<|endoftext|>` 或达到最大新 token 数。支持：**温度** $\mathrm{softmax}(v,\tau)_i=\frac{\exp(v_i/\tau)}{\sum_j\exp(v_j/\tau)}$；**top-p（核采样）**：按概率降序累加，取最小集合 $V(p)$ 使 $\sum_{j\in V(p)}q_j\ge p$，其余置 0 重新归一化。📝 编辑 `decode.py::generate`。
""")
code(r"""
# !uv run python -m cs336_basics.decode \
#     --checkpoint ckpts/smoke/ckpt_final.pt --vocab ts_vocab.json --merges ts_merges.txt \
#     --prompt "Once upon a time" --max-new-tokens 256 --temperature 0.8 --top-p 0.95
print("实现 decode.py 后取消注释运行。")
""")

# =====================================================================
# §7 实验与排行榜
# =====================================================================
md(r"""
# 第 7 节 · 实验与消融

TinyStories 起步超参：`vocab=10000, context=256, d_model=512, d_ff=1344, rope_theta=10000, layers=4, heads=16`（约 1700 万非嵌入参数）；总处理 token ≈ `batch×steps×context` ≈ 3.28e8。需自行调：学习率、预热、AdamW $(\beta_1,\beta_2,\varepsilon)$、权重衰减。

> 调试技巧：先**过拟合单个小 batch**（loss 应迅速趋零）；监控激活/权重/梯度范数是否爆炸或消失。低资源（CPU/MPS）：把总 token 降到 4e7、目标 val loss 放宽到 2.0，并让余弦衰减恰好在末步结束。
""")

md(r"""
## 题目 (experiment_log)：实验日志基础设施（3 分）· 实验

搭建可按**梯度步**与**挂钟时间**追踪 loss 曲线的设施（推荐 [Weights & Biases](https://wandb.ai)）。交付：日志代码 + 一份记录你所有尝试的实验日志文档。
""")
code(r"""
# 一个轻量本地记录器示例（不依赖 wandb）：把 (step, wall_time, train_loss, val_loss) 追加到 csv，
# 训练循环里调用，事后用下面的绘图 cell 画学习曲线。
import csv, time
class RunLogger:
    def __init__(self, path): self.path=path; self.t0=time.time(); self.f=open(path,"w",newline=""); self.w=csv.writer(self.f); self.w.writerow(["step","wall_s","train_loss","val_loss"])
    def log(self, step, train_loss=None, val_loss=None): self.w.writerow([step, round(time.time()-self.t0,2), train_loss, val_loss]); self.f.flush()
print("把 RunLogger 接进 train.py，或直接用 wandb.init/wandb.log。")
""")
code(r"""
# 学习曲线绘制工具（多次实验对比）
import pandas as pd, matplotlib.pyplot as plt
def plot_curves(csvs, x="step"):
    plt.figure(figsize=(8,5))
    for name, path in csvs.items():
        df = pd.read_csv(path)
        d = df.dropna(subset=["val_loss"])
        plt.plot(d[x], d["val_loss"], label=name)
    plt.xlabel(x); plt.ylabel("val loss"); plt.legend(); plt.title("learning curves"); plt.grid(True); plt.show()
# plot_curves({"lr=3e-4":"runs/lr3e4.csv", "lr=1e-3":"runs/lr1e3.csv"})
""")
answer("附实验日志文档链接 / 摘要。")

md(r"""
## 题目 (learning_rate)：调优学习率（3 分）· 实验

- **(a)** 在学习率上做一次扫描，报告各自最终 loss（发散则注明）。**交付**：多条学习率的学习曲线 + 搜索策略说明；以及一个在 TinyStories 上 **val loss ≤ 1.45**（低资源可放宽到 2.0）的模型。
- **(b)** 研究「发散临界学习率」与「最佳学习率」的关系（含至少一次发散运行）。
""")
code(r"""
# 扫描示例：对每个 lr 跑一个短训练，记录 csv，再 plot_curves 对比。
# for lr in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
#     !uv run python -m cs336_basics.train --train-data ts_train.npy --valid-data ts_valid.npy \
#         --lr {lr} --batch-size 64 --max-iters 5000 --checkpoint-dir ckpts/lr_{lr}
print("扫描后用 plot_curves 出图，并在解答中分析。")
""")
answer()

md(r"""
## 题目 (batch_size_experiment)：批大小变化（1 分）· 实验

把 batch 从 1 调到显存上限（含 64、128 等），必要时重调学习率。**交付**：不同 batch 的学习曲线 + 几句发现讨论。
""")
answer()

md(r"""
## 题目 (generate)：生成文本（1 分）· 实验

用你的解码器与训练好的 checkpoint，生成**≥256 token**（或到首个 `<|endoftext|>`）。调温度/top-p 取得流畅输出。**交付**：文本转储 + 流畅度点评 + 至少两个影响质量的因素。
""")
answer()

md(r"""
## 题目 (layer_norm_ablation)：移除 RMSNorm（1 分）· 实验

从每个块移除所有 RMSNorm 再训练。原最优 lr 下会怎样？降低 lr 能否恢复稳定？**交付**：移除后的学习曲线 + 最优 lr 下的曲线 + 对 RMSNorm 作用的点评。
""")
answer()

md(r"""
## 题目 (pre_norm_ablation)：改用 post-norm（1 分）· 实验

把 pre-norm 改为 post-norm：$z=\mathrm{RMSNorm}(x+\mathrm{MHSA}(x))$，$y=\mathrm{RMSNorm}(z+\mathrm{FFN}(z))$。训练并与 pre-norm 对比。**交付**：两者学习曲线对比。
""")
answer()

md(r"""
## 题目 (no_pos_emb)：NoPE（1 分）· 实验

移除位置编码（不加 RoPE），与 RoPE 基线对比。**交付**：RoPE vs NoPE 的学习曲线。
""")
answer()

md(r"""
## 题目 (swiglu_ablation)：SwiGLU vs SiLU（1 分）· 实验

对比 SwiGLU 与无门控的 $\mathrm{FFN}_{\mathrm{SiLU}}(x)=W_2\,\mathrm{SiLU}(W_1 x)$（为参数量大致匹配，SiLU 版取 $d_{ff}=4d_{model}$）。**交付**：参数匹配下的对比曲线 + 几句讨论。
""")
answer()

md(r"""
## 题目 (main_experiment)：在 OpenWebText 上训练（2 分）· 实验

用与 TinyStories 相同架构与总迭代数训练 OWT（可能需重调 lr/batch）。**交付**：OWT 学习曲线 + 与 TinyStories 损失差异的解读；同格式生成文本 + 为何同算力下质量更差的分析。
""")
answer()

md(r"""
## 题目 (leaderboard)：排行榜（6 分）· 实验

规则：单卡 B200 ≤45 分钟、只用给定 OWT 数据。目标在 0.75 B200·小时内最小化 val loss（朴素基线 5.0）。可尝试：权重绑定、调架构/超参、参考 Llama3/Qwen2.5、nanoGPT speedrun。**交付**：最终 val loss + 以挂钟时间(<45min)为横轴的学习曲线 + 做法说明。提交到 `assignment1-basics-leaderboard` 仓库。
""")
answer()

# =====================================================================
# 收尾
# =====================================================================
md(r"""
# 提交清单

1. **跑通全部测试**（下面 cell；实现完整后应全绿）。
2. `writeup.pdf`：所有书面题（`unicode1/2`、`transformer_accounting`、`adamw_accounting`、`learning_rate_tuning`）+ 各实验的曲线/日志/结论，排版后导出。
3. `code.zip`：`bash make_submission.sh` 生成（大数据/checkpoint 记得加入排除列表）。
4. （可选）排行榜：向 `assignment1-basics-leaderboard` 提 PR。
""")
code(r"""
# 一键全量自测（耗时较长；含 BPE 速度测试）。日常请用各题的 run_test 单测。
run_test(path="tests")
""")

# =====================================================================
# 写出 notebook
# =====================================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("WROTE", OUT, "| cells:", len(cells),
      "| md:", sum(c["cell_type"] == "markdown" for c in cells),
      "| code:", sum(c["cell_type"] == "code" for c in cells))
