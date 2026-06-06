"""把文本语料编码成 uint16 token 的「裸二进制」文件，供 train.py 的 np.memmap 读取。

为什么是裸二进制 + uint16（不是 np.save / .npy）：
    train.py 用 `np.memmap(path, dtype="uint16", mode="r")` 读 token，
    np.memmap 直接把整个文件当成连续的 uint16 数组，没有文件头。
    若用 np.save 写成 .npy，开头会多出几十字节的 header，memmap 会把 header
    当成前几个 token 读进来 —— 数据错位。所以这里必须用
    `ndarray.astype(uint16).tofile(f)` 写裸字节。

内存友好：逐行读 -> 逐行 encode -> 攒够一批就 .tofile 落盘 -> 清空缓冲，
    因此即便语料有几 GB、token 上亿，内存占用也恒定（只跟缓冲区大小成正比）。
    逐行编码与 tokenizer.encode_iterable(fin) 语义等价，这里手写循环是为了
    分批落盘 + 显示进度条。

用法：
    uv run python -m cs336_basics.encode_corpus \\
        --vocab vocab.json --merges merges.txt \\
        --input data/TinyStoriesV2-GPT4-train.txt \\
        --output data/ts_train.bin
    （train / valid 各跑一遍，分别得到 ts_train.bin / ts_valid.bin）

然后训练时：
    uv run python -m cs336_basics.train \\
        --train-data data/ts_train.bin --valid-data data/ts_valid.bin ...
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="将文本语料编码为 uint16 token 二进制文件 (供 np.memmap 读取)")
    p.add_argument("--vocab", type=str, required=True, help="序列化 vocab 文件路径 (run_bpe_training 的 --vocab-out)")
    p.add_argument("--merges", type=str, required=True, help="序列化 merges 文件路径 (run_bpe_training 的 --merges-out)")
    p.add_argument("--input", type=str, required=True, help="输入语料文本 .txt 路径")
    p.add_argument("--output", type=str, required=True, help="输出 token 二进制 .bin 路径 (裸 uint16)")
    p.add_argument(
        "--special-tokens",
        type=str,
        nargs="*",
        default=["<|endoftext|>"],
        help="特殊 token 列表 (须与训练 BPE 时一致；作为文档分隔符整体编码)",
    )
    p.add_argument("--dtype", type=str, default="uint16", help="输出 token 的 numpy dtype (须与 train.py 一致)")
    p.add_argument("--buffer-tokens", type=int, default=1_000_000, help="每攒够多少 token 落一次盘")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = np.dtype(args.dtype)

    # ---- 加载分词器 ----
    print(f"加载分词器: vocab={args.vocab}  merges={args.merges}")
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens=args.special_tokens)

    # ---- 安全检查：所有 token id 必须放得进目标 dtype，否则会静默溢出回绕 ----
    # uint16 上限 65535；TinyStories(vocab=10000)、OWT(vocab=32000) 都安全。
    max_id = max(tokenizer.vocab) if getattr(tokenizer, "vocab", None) else 0
    dtype_max = int(np.iinfo(dtype).max)
    assert max_id <= dtype_max, (
        f"最大 token id {max_id} 超过 {args.dtype} 上限 {dtype_max}，会溢出！请改用更大的 dtype（如 uint32）。"
    )
    print(f"词表最大 id={max_id}，{args.dtype} 上限={dtype_max} ✓")

    # ---- 逐行编码 + 分批落盘 ----
    file_size = os.path.getsize(args.input)
    buf: list[int] = []
    total_tokens = 0

    with (
        open(args.input, "r", encoding="utf-8") as fin,
        open(args.output, "wb") as fout,
        tqdm(total=file_size, unit="B", unit_scale=True, desc=f"编码 {os.path.basename(args.input)}") as pbar,
    ):
        for line in fin:
            buf.extend(tokenizer.encode(line))          # 等价于 encode_iterable 的逐行编码
            pbar.update(len(line.encode("utf-8")))       # 进度按已读字节推进
            if len(buf) >= args.buffer_tokens:
                np.asarray(buf, dtype=dtype).tofile(fout)  # 裸 uint16 追加写
                total_tokens += len(buf)
                buf.clear()
        if buf:                                          # 收尾，写出残余 token
            np.asarray(buf, dtype=dtype).tofile(fout)
            total_tokens += len(buf)

    out_mb = total_tokens * dtype.itemsize / 1e6
    print(f"\n✅ 写出 {total_tokens:,} 个 token -> {args.output}  ({out_mb:.1f} MB, dtype={args.dtype})")

    # ---- 自检：用与 train.py 完全相同的 memmap 调用读回，确认可读、长度对、能解码 ----
    mm = np.memmap(args.output, dtype=dtype, mode="r")
    assert len(mm) == total_tokens, f"回读长度 {len(mm)} != 写出 {total_tokens}，文件可能损坏"
    preview = tokenizer.decode(mm[:50].tolist())
    del mm
    print(f"自检 OK：np.memmap 读回 {total_tokens:,} 个 token；前 50 token 解码预览：")
    print("   " + repr(preview[:200]))


if __name__ == "__main__":
    main()
