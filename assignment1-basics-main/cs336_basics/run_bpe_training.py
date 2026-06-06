"""命令行 BPE 训练脚本：在语料上训练 BPE 并序列化 vocab/merges（讲义 §2.5）。

命令行入口：读取一份文本语料，调用 cs336_basics.tokenizer.train_bpe 训练
字节级 BPE 分词器，然后把得到的 vocab（dict[int, bytes]）与 merges
（list[tuple[bytes, bytes]]）序列化到磁盘，供后续训练 / 解码加载。

本文件是「可运行骨架」：argparse 与 main() 结构完整，序列化细节留 TODO，
运行时会抛 NotImplementedError，属预期。

用法示例：
    python -m cs336_basics.run_bpe_training \\
        --input data/owt_train.txt \\
        --vocab-size 10000 \\
        --special-tokens "<|endoftext|>" \\
        --vocab-out vocab.json --merges-out merges.txt
"""

from __future__ import annotations

import argparse
import time
import base64
import json

from cs336_basics.tokenizer import train_bpe


def parse_args() -> argparse.Namespace:
    """解析命令行参数（语料路径、词表大小、特殊 token、输出路径）。

    返回:
        argparse.Namespace: 含输入语料、vocab_size、special_tokens 与输出文件路径。
    """
    p = argparse.ArgumentParser(description="在语料上训练 BPE 分词器并序列化 (CS336 作业一)")

    p.add_argument("--input", type=str, required=True, help="训练语料文本路径")
    p.add_argument("--vocab-size", type=int, default=10000, help="目标词表大小（含特殊 token 与 256 字节基底）")
    p.add_argument(
        "--special-tokens",
        type=str,
        nargs="*",
        default=["<|endoftext|>"],
        help="特殊 token 列表（永不被拆分）",
    )
    p.add_argument("--vocab-out", type=str, default="vocab.json", help="序列化后的 vocab 输出路径")
    p.add_argument("--merges-out", type=str, default="merges.txt", help="序列化后的 merges 输出路径")

    return p.parse_args()


def save_vocab_and_merges(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_path: str,
    merges_path: str,
) -> None:
    """把训练得到的 vocab 与 merges 序列化到磁盘。

    Args:
        vocab (dict[int, bytes]): token id -> token 字节串 的映射。
        merges (list[tuple[bytes, bytes]]): 按创建顺序排列的 BPE 合并对。
        vocab_path (str): vocab 输出路径。
        merges_path (str): merges 输出路径。

    说明：bytes 不能直接进 JSON / 文本，需要一种可逆编码（如 base64 或
    GPT-2 风格的 byte<->unicode 映射）。Tokenizer.from_files 必须能反序列化它。
    """
    #######################################################################
    # TODO: 序列化 vocab（dict[int, bytes]）到 vocab_path。
    #   bytes 不能直接 JSON 序列化，需选一种可逆编码，例如:
    #     - base64: {str(tid): base64.b64encode(b).decode("ascii") for tid, b in vocab.items()}
    #     - 或 GPT-2 风格 byte->unicode 映射后写 JSON。
    #   关键: 与 Tokenizer.from_files 的反序列化逻辑保持完全一致、可往返。
    #######################################################################
    raw_vocab = {
        str(tid): base64.b64encode(b).decode("ascii")
        for tid, b in vocab.items()
    }
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(raw_vocab, f)
    #######################################################################
    #                             END OF YOUR CODE                            #
    #######################################################################

    #######################################################################
    # TODO: 序列化 merges（list[tuple[bytes, bytes]]）到 merges_path。
    #   常见做法: 每行一条合并，形如 "<token1> <token2>"，token 同样需可逆编码
    #   （base64 或 GPT-2 byte->unicode），并保持合并的创建顺序。
    #   关键: 与 Tokenizer.from_files 读取 merges 的格式严格对应。
    #######################################################################
    with open(merges_path, "w", encoding="utf-8") as f:
        for a, b in merges:
            a_enc = base64.b64encode(a).decode("ascii")
            b_enc = base64.b64encode(b).decode("ascii")
            f.write(f"{a_enc} {b_enc}\n")
    #######################################################################
    #                             END OF YOUR CODE                            #
    #######################################################################


def main() -> None:
    """训练 BPE 分词器并把 vocab/merges 写入磁盘。"""
    args = parse_args()

    print(f"开始在 {args.input} 上训练 BPE，目标词表 {args.vocab_size}，特殊 token {args.special_tokens}")
    t0 = time.time()

    vocab, merges = train_bpe(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
    )

    elapsed = time.time() - t0
    print(f"BPE 训练完成：vocab={len(vocab)} 项，merges={len(merges)} 条，用时 {elapsed:.1f}s")

    save_vocab_and_merges(vocab, merges, args.vocab_out, args.merges_out)
    print(f"已序列化 vocab -> {args.vocab_out}，merges -> {args.merges_out}")


if __name__ == "__main__":
    main()
