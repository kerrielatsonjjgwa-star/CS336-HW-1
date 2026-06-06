# CS336 Assignment 1 — 脚手架模块 API 契约（唯一事实来源）

所有脚手架 agent **必须**严格遵守以下签名/属性命名。`tests/adapters.py` 的胶水层会按此契约调用学生代码。
任何偏差都会导致测试无法连接。包名为 `cs336_basics`（见 pyproject.toml）。

## 通用约定
- 所有 `nn.Module` 子类只能用 `torch.nn.Parameter`、`torch.nn` 容器（`Module/ModuleList/Sequential`）、`torch.optim.Optimizer` 基类；**不得**使用 `torch.nn` 现成层或 `torch.nn.functional`。
- 权重张量形状遵循讲义：`Linear` 权重为 `(out_features, in_features)`，计算 `y = x W^T`。
- 每个模块文件顶部含中文模块级 docstring，注明对应讲义章节。
- 每个待实现处用 CS231n 风格三段式：中文 docstring → `# TODO` 横幅块（含公式/形状/提示）→ `# END OF YOUR CODE` 横幅；函数体以 `raise NotImplementedError` 占位，保证 `py_compile` 通过。
- `__init__` 必须调用 `super().__init__()`，其余参数创建留作 TODO（含形状提示注释）。

---

## 文件 1：`cs336_basics/nn_utils.py`（讲义 §3.4.4 / §4.1 / §4.5）
```python
def softmax(in_features: Tensor, dim: int) -> Tensor: ...      # 数值稳定 softmax；backs run_softmax
def silu(in_features: Tensor) -> Tensor: ...                    # x*sigmoid(x)；backs run_silu
def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor: ...# 平均交叉熵；backs run_cross_entropy；inputs (batch, vocab), targets (batch,)
def gradient_clipping(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None: ...# 原地裁剪；backs run_gradient_clipping
```

## 文件 2：`cs336_basics/model.py`（讲义 §3）
`from cs336_basics.nn_utils import softmax, silu`
```python
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None): ...
    # 属性: self.weight: nn.Parameter  形状 (out_features, in_features)
    def forward(self, x: Tensor) -> Tensor: ...                 # backs run_linear

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None): ...
    # 属性: self.weight: nn.Parameter  形状 (num_embeddings, embedding_dim)
    def forward(self, token_ids: Tensor) -> Tensor: ...         # backs run_embedding

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None): ...
    # 属性: self.weight: nn.Parameter 形状 (d_model,); self.eps
    def forward(self, x: Tensor) -> Tensor: ...                 # backs run_rmsnorm

class SwiGLU(nn.Module):                                        # 位置前馈网络 (§3.4.2)
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None): ...
    # 子模块: self.w1: Linear(d_model,d_ff); self.w2: Linear(d_ff,d_model); self.w3: Linear(d_model,d_ff)
    # 即 w1.weight (d_ff,d_model), w2.weight (d_model,d_ff), w3.weight (d_ff,d_model)
    def forward(self, x: Tensor) -> Tensor: ...                 # backs run_swiglu: w2(silu(w1 x) * w3 x)

class RotaryPositionalEmbedding(nn.Module):                    # RoPE (§3.4.3)
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None): ...
    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor: ...# backs run_rope

def scaled_dot_product_attention(Q, K, V, mask=None) -> Tensor: ...# backs run_scaled_dot_product_attention

class MultiHeadSelfAttention(nn.Module):                       # §3.4.5
    def __init__(self, d_model: int, num_heads: int, use_rope: bool=False,
                 max_seq_len: int|None=None, theta: float|None=None, device=None, dtype=None): ...
    # 子模块(均为 Linear, 属性名严格如下，供 state_dict 加载):
    #   self.q_proj, self.k_proj, self.v_proj, self.output_proj  (各 (d_model,d_model))
    #   若 use_rope: self.rope = RotaryPositionalEmbedding(theta, d_model//num_heads, max_seq_len)
    def forward(self, x: Tensor, token_positions: Tensor|None=None) -> Tensor: ...
    # backs run_multihead_self_attention (use_rope=False) 与 run_multihead_self_attention_with_rope (use_rope=True)

class TransformerBlock(nn.Module):                             # 预归一化块 §3.4
    def __init__(self, d_model:int, num_heads:int, d_ff:int, max_seq_len:int, theta:float, device=None, dtype=None): ...
    # 子模块属性名严格如下(对应 run_transformer_block 的 weights 键):
    #   self.ln1: RMSNorm ; self.attn: MultiHeadSelfAttention(use_rope=True) ; self.ln2: RMSNorm ; self.ffn: SwiGLU
    #   权重键: ln1.weight, attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight, attn.output_proj.weight,
    #           ffn.w1.weight, ffn.w2.weight, ffn.w3.weight, ln2.weight
    def forward(self, x: Tensor, token_positions: Tensor|None=None) -> Tensor: ...

class TransformerLM(nn.Module):                               # §3.5
    def __init__(self, vocab_size:int, context_length:int, d_model:int, num_layers:int,
                 num_heads:int, d_ff:int, rope_theta:float, device=None, dtype=None): ...
    # 子模块属性名严格如下(对应 run_transformer_lm 的 weights 键):
    #   self.token_embeddings: Embedding ; self.layers: nn.ModuleList[TransformerBlock] ;
    #   self.ln_final: RMSNorm ; self.lm_head: Linear
    #   权重键: token_embeddings.weight, layers.{i}.<块内键>, ln_final.weight, lm_head.weight
    def forward(self, token_ids: Tensor) -> Tensor: ...        # backs run_transformer_lm
```

## 文件 3：`cs336_basics/tokenizer.py`（讲义 §2）
```python
def train_bpe(input_path, vocab_size: int, special_tokens: list[str], **kwargs
              ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: ...   # backs run_train_bpe

class Tokenizer:
    def __init__(self, vocab: dict[int,bytes], merges: list[tuple[bytes,bytes]], special_tokens: list[str]|None=None): ...
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None) -> "Tokenizer": ...
    def encode(self, text: str) -> list[int]: ...
    def encode_iterable(self, iterable) -> "Iterator[int]": ...
    def decode(self, ids: list[int]) -> str: ...
# get_tokenizer(vocab, merges, special_tokens) 直接 return Tokenizer(...)
```

## 文件 4：`cs336_basics/optimizer.py`（讲义 §4.3 / §4.4）
```python
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01): ...
    def step(self, closure=None): ...                          # get_adamw_cls() return AdamW

def get_lr_cosine_schedule(it:int, max_learning_rate:float, min_learning_rate:float,
                           warmup_iters:int, cosine_cycle_iters:int) -> float: ...# backs run_get_lr_cosine_schedule
```

## 文件 5：`cs336_basics/data.py`（讲义 §5.1）
```python
def get_batch(dataset, batch_size:int, context_length:int, device:str) -> tuple[Tensor, Tensor]: ...# backs run_get_batch
```

## 文件 6：`cs336_basics/serialization.py`（讲义 §5.2）
```python
def save_checkpoint(model, optimizer, iteration:int, out) -> None: ...# backs run_save_checkpoint
def load_checkpoint(src, model, optimizer) -> int: ...               # backs run_load_checkpoint
```

## 可运行脚本（带 TODO 的骨架，§5.3 / §6 / §2.5）
- `cs336_basics/train.py`：argparse 超参 → 构建 np.memmap 数据/模型/AdamW/余弦调度/梯度裁剪/checkpoint/(wandb 日志桩) 的训练循环；含 `main()` 与 `if __name__=="__main__"`。
- `cs336_basics/decode.py`：加载 checkpoint+tokenizer，自回归采样（temperature、top-p）生成文本。
- `cs336_basics/run_bpe_training.py`：CLI 在语料上调用 `tokenizer.train_bpe` 并序列化 vocab/merges。

## 文件 7：`tests/adapters.py`（**完整可用胶水，非 TODO**）
保留每个 `run_*` / `get_*` 的签名与 docstring，把函数体从 `raise NotImplementedError` 改为：导入 `cs336_basics` 对应符号 → 构造/调用 → 装载给定权重 → 返回输出。
- 单权重模块（Linear/Embedding/RMSNorm）：构造后 `m.weight.data = weights` 再前向。
- SwiGLU：`m.w1.weight.data=w1_weight` 等，或 `m.load_state_dict({...})`。
- MHA：把四个投影权重写入 `q_proj/k_proj/v_proj/output_proj` 的 `.weight.data`；`with_rope` 版构造 `use_rope=True` 并传 `token_positions`。
- TransformerBlock / TransformerLM：`m.load_state_dict(weights)` 后前向（键已对齐）。
- 学生未实现时构造即抛 `NotImplementedError`，故初始 `uv run pytest` 全部失败——符合预期起点。
