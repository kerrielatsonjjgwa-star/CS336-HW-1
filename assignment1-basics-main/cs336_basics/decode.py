"""解码 / 文本生成脚本：加载 checkpoint 与 Tokenizer 自回归采样（讲义 §5.3 / §6）。

命令行入口：从磁盘载入训练好的 TransformerLM 权重与 BPE Tokenizer，
按给定 prompt 进行自回归采样，支持 temperature 缩放与 top-p（核采样）截断，
直至生成 <|endoftext|> 或达到最大新 token 数。

本文件是「可运行骨架」：argparse 与 main() 结构完整，采样核心循环留 TODO，
运行时会抛 NotImplementedError，属预期。

用法示例：
    python -m cs336_basics.decode \\
        --checkpoint ckpts/ckpt_final.pt \\
        --vocab vocab.json --merges merges.txt \\
        --prompt "Once upon a time" \\
        --max-new-tokens 256 --temperature 0.8 --top-p 0.95
"""

from __future__ import annotations

import argparse

import torch

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import softmax
from cs336_basics.serialization import load_checkpoint
from cs336_basics.tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    """解析命令行参数（checkpoint/tokenizer 路径、模型结构、采样超参）。

    返回:
        argparse.Namespace: 含权重路径、tokenizer 文件、模型结构超参与采样控制项。
    """
    p = argparse.ArgumentParser(description="用训练好的 Transformer LM 自回归生成文本 (CS336 作业一)")

    # ---- checkpoint 与 tokenizer 文件 ----
    p.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径 (.pt)")
    p.add_argument("--vocab", type=str, required=True, help="序列化 vocab 文件路径")
    p.add_argument("--merges", type=str, required=True, help="序列化 merges 文件路径")
    p.add_argument(
        "--special-tokens",
        type=str,
        nargs="*",
        default=["<|endoftext|>"],
        help="特殊 token 列表（生成遇到首个 <|endoftext|> 即停止）",
    )

    # ---- 模型结构超参（须与训练时一致，以便正确加载权重）----
    p.add_argument("--vocab-size", type=int, default=10000, help="词表大小")
    p.add_argument("--context-length", type=int, default=256, help="上下文长度")
    p.add_argument("--d-model", type=int, default=512, help="模型隐藏维度")
    p.add_argument("--num-layers", type=int, default=4, help="Transformer 层数")
    p.add_argument("--num-heads", type=int, default=16, help="注意力头数")
    p.add_argument("--d-ff", type=int, default=1344, help="前馈层内部维度")
    p.add_argument("--rope-theta", type=float, default=10000.0, help="RoPE Theta 参数")

    # ---- 采样控制 ----
    p.add_argument("--prompt", type=str, default="", help="生成所用的初始提示文本")
    p.add_argument("--max-new-tokens", type=int, default=256, help="最多生成的新 token 数")
    p.add_argument("--temperature", type=float, default=1.0, help="温度（>0；越小越确定，=1 不缩放）")
    p.add_argument("--top-p", type=float, default=1.0, help="核采样累计概率阈值（1.0 表示不截断）")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    p.add_argument("--seed", type=int, default=42, help="随机种子")

    return p.parse_args()


@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    context_length: int,
    device: str,
    eot_id: int | None = None,
) -> str:
    """自回归采样生成文本。

    流程：把 prompt 编码为 token id -> 反复地 [前向取末位 logits -> 温度缩放
    -> softmax -> top-p 截断并重新归一化 -> 多项式采样下一个 token -> 追加] ->
    最后 decode 回字符串。

    Args:
        model: 已载入权重并置于 eval 模式的 TransformerLM。
        tokenizer: BPE Tokenizer，用于 encode/decode。
        prompt (str): 初始提示文本。
        max_new_tokens (int): 最多生成的新 token 数。
        temperature (float): 温度系数（>0）。
        top_p (float): 核采样累计概率阈值，<1 时启用截断。
        context_length (int): 模型最大上下文长度（超出时需截断输入窗口）。
        device (str): 计算设备。
        eot_id (int | None): <|endoftext|> 的 token id；采到它则提前停止。

    Returns:
        str: 生成的完整文本（含 prompt 解码结果）。
    """
    #######################################################################
    # TODO: 把 prompt 编码为初始 token id 序列（list[int] -> 1D LongTensor）。
    #   ids = tokenizer.encode(prompt)
    #   x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]  # (1, T)
    #   提示: prompt 为空时可用一个起始 token 或直接从空序列起步，按需处理边界。
    #######################################################################
    ids = tokenizer.encode(prompt) if prompt is not None else [eot_id]
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
    #######################################################################
    #                             END OF YOUR CODE                            #
    #######################################################################
    with torch.no_grad():
        for _ in range(max_new_tokens):
            ###################################################################
            # TODO: 前向取最后一个位置的 logits 并按温度缩放。
            #   ctx = x[:, -context_length:]                 # 截断到上下文窗口
            #   logits = model(ctx)[:, -1, :]                # (1, vocab_size)
            #   if temperature > 0: logits = logits / temperature
            ###################################################################
            ctx = x[:, -context_length:]
            logits = model(ctx)[:, -1, :]
            if temperature > 0:
                logits = logits / temperature
            ###################################################################
            #                             END OF YOUR CODE                            #
            ###################################################################

            ###################################################################
            # TODO: softmax 转概率，并按 top-p（核采样）截断后重新归一化。
            #   probs = softmax(logits, dim=-1)              # (1, vocab_size)
            #   若 top_p < 1: 对 probs 降序排序，累加到首次 >= top_p 处，
            #     将其余概率置 0 并重新归一化（保证至少保留一个 token）。
            ###################################################################
            probs = softmax(logits, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                remove = (cumsum - sorted_probs) >= top_p
                sorted_probs[remove] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
            ###################################################################
            #                             END OF YOUR CODE                            #
            ###################################################################

            ###################################################################
            # TODO: 多项式采样下一个 token 并追加；遇 eot 则停止。
            #   next_id = torch.multinomial(probs, num_samples=1)   # (1, 1)
            #   x = torch.cat([x, next_id], dim=1)
            #   if eot_id is not None and next_id.item() == eot_id: break
            ###################################################################
            next_id = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_id], dim=-1)
            if eot_id is not None and next_id.item() == eot_id:
                break
            ###################################################################
            #                             END OF YOUR CODE                            #
            ###################################################################

    #######################################################################
    # TODO: 把生成的全部 token id 解码回字符串并返回。
    #   return tokenizer.decode(x[0].tolist())
    #######################################################################
    return tokenizer.decode(x[0].tolist())
    #######################################################################
    #                             END OF YOUR CODE                            #
    #######################################################################


def main() -> None:
    """加载 tokenizer 与模型权重，执行一次采样生成并打印结果。"""
    args = parse_args()
    torch.manual_seed(args.seed)

    # ---- 加载 Tokenizer（从序列化的 vocab/merges 文件）----
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens=args.special_tokens)

    # ---- 解析 <|endoftext|> 的 token id（用于提前停止）----
    eot_id = None
    if args.special_tokens:
        eot_bytes = args.special_tokens[0].encode("utf-8")
        for tid, tbytes in tokenizer.vocab.items() if hasattr(tokenizer, "vocab") else []:
            if tbytes == eot_bytes:
                eot_id = tid
                break

    # ---- 构建模型并从 checkpoint 加载权重 ----
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    ).to(args.device)

    # load_checkpoint 需要一个 optimizer 占位以复用通用接口；推理时其状态不重要。
    dummy_optimizer = AdamW(model.parameters(), lr=0.0)
    load_checkpoint(args.checkpoint, model, dummy_optimizer)
    model.eval()

    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        context_length=args.context_length,
        device=args.device,
        eot_id=eot_id,
    )

    print("=" * 72)
    print(text)
    print("=" * 72)


if __name__ == "__main__":
    main()
