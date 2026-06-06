"""把文本语料编码成 uint16 token 的「裸二进制」文件，供 train.py 的 np.memmap 读取。

为什么是裸二进制 + uint16（不是 np.save / .npy）：
    train.py 用 `np.memmap(path, dtype="uint16", mode="r")` 读 token，
    np.memmap 直接把整个文件当成连续的 uint16 数组，没有文件头。
    若用 np.save 写成 .npy，开头会多出几十字节的 header，memmap 会把 header
    当成前几个 token 读进来 —— 数据错位。所以必须用 ndarray.tofile 写裸字节。

并行：大文件按「行边界」切块（保证不切断任何一行），多进程各自逐行 encode、写出
    裸 uint16 分块文件，主进程再「按块序拼接」成最终 .bin。逐行编码与单进程完全一致、
    块按行对齐、按序拼接，因此输出与单进程逐字节相同。进程数按 cgroup 真实配额探测
    （容器内 os.cpu_count 会虚报宿主核数）。小文件自动走串行，避免多进程开销。

用法：
    uv run python -m cs336_basics.encode_corpus \\
        --vocab vocab.json --merges merges.txt \\
        --input data/TinyStoriesV2-GPT4-train.txt \\
        --output data/ts_train.bin
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer import Tokenizer, _resolve_num_processes

# 小于此大小直接串行编码，避免多进程启动开销。
_PARALLEL_MIN_BYTES = 1 << 20  # 1 MiB


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="将文本语料并行编码为 uint16 token 二进制文件 (供 np.memmap 读取)")
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
    p.add_argument("--num-processes", type=int, default=None, help="并行进程数 (默认按 cgroup 真实配额自动探测)")
    return p.parse_args()


def _find_line_boundaries(path: str, desired_num_chunks: int) -> list[int]:
    """把文件切成若干「按行对齐」的字节块边界（每块都是完整的行）。

    在每个均匀猜测位置用 readline() 前进到下一行行首，保证不切断任何一行；
    这样各块逐行编码再按序拼接，结果与单进程逐行编码逐字节一致。
    """
    size = os.path.getsize(path)
    if size == 0:
        return [0, 0]
    boundaries = [i * size // desired_num_chunks for i in range(desired_num_chunks + 1)]
    boundaries[-1] = size
    with open(path, "rb") as f:
        for bi in range(1, len(boundaries) - 1):
            f.seek(boundaries[bi])
            f.readline()  # 吃掉当前残行，前进到下一行行首
            boundaries[bi] = f.tell()
    return sorted(set(boundaries))


def _encode_chunk(args: tuple) -> tuple[int, str, int]:
    """工作进程：读取 [start, end) 的完整行，逐行 encode，写出裸 uint16 分块文件。

    返回 (chunk 下标, 分块文件路径, token 数)。逐行 encode 与单进程一致，块按行对齐，
    故所有分块按下标顺序拼接 == 单进程整体编码输出。
    """
    idx, input_path, start, end, vocab_path, merges_path, special_tokens, dtype_str, part_path, buffer_tokens = args
    tok = Tokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)
    dtype = np.dtype(dtype_str)
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    total = 0
    buf: list[int] = []
    with open(part_path, "wb") as pf:
        for line in io.StringIO(chunk):
            buf.extend(tok.encode(line))
            if len(buf) >= buffer_tokens:
                np.asarray(buf, dtype=dtype).tofile(pf)
                total += len(buf)
                buf.clear()
        if buf:
            np.asarray(buf, dtype=dtype).tofile(pf)
            total += len(buf)
    return idx, part_path, total


def _encode_serial(input_path: str, output: str, tokenizer: Tokenizer, dtype, buffer_tokens: int) -> int:
    """串行逐行编码（小文件用）。返回写出的 token 数。"""
    file_size = os.path.getsize(input_path)
    total, buf = 0, []
    with (
        open(input_path, "r", encoding="utf-8") as fin,
        open(output, "wb") as fout,
        tqdm(total=file_size, unit="B", unit_scale=True, desc=f"编码 {os.path.basename(input_path)}") as pbar,
    ):
        for line in fin:
            buf.extend(tokenizer.encode(line))
            pbar.update(len(line.encode("utf-8")))
            if len(buf) >= buffer_tokens:
                np.asarray(buf, dtype=dtype).tofile(fout)
                total += len(buf)
                buf.clear()
        if buf:
            np.asarray(buf, dtype=dtype).tofile(fout)
            total += len(buf)
    return total


def main() -> None:
    args = parse_args()
    dtype = np.dtype(args.dtype)

    # ---- 加载分词器（主进程用于安全检查与最终自检解码）----
    print(f"加载分词器: vocab={args.vocab}  merges={args.merges}")
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens=args.special_tokens)

    # ---- 安全检查：所有 token id 必须放得进目标 dtype，否则会静默溢出回绕 ----
    max_id = max(tokenizer.vocab) if getattr(tokenizer, "vocab", None) else 0
    dtype_max = int(np.iinfo(dtype).max)
    assert max_id <= dtype_max, (
        f"最大 token id {max_id} 超过 {args.dtype} 上限 {dtype_max}，会溢出！请改用更大的 dtype（如 uint32）。"
    )
    print(f"词表最大 id={max_id}，{args.dtype} 上限={dtype_max} ✓")

    # ---- 并行 / 串行编码 ----
    num_processes = _resolve_num_processes(args.num_processes)
    file_size = os.path.getsize(args.input)
    boundaries: list[int] = []
    if num_processes > 1 and file_size >= _PARALLEL_MIN_BYTES:
        boundaries = _find_line_boundaries(args.input, num_processes)

    if len(boundaries) > 2:
        nchunks = len(boundaries) - 1
        print(f"并行编码: {nchunks} 块 × {min(num_processes, nchunks)} 进程")
        tasks = [
            (i, args.input, boundaries[i], boundaries[i + 1], args.vocab, args.merges,
             args.special_tokens, args.dtype, f"{args.output}.part{i:03d}", args.buffer_tokens)
            for i in range(nchunks)
        ]
        parts: list[tuple[str, int]] = [("", 0)] * nchunks
        with Pool(min(num_processes, nchunks)) as pool:
            for idx, part_path, cnt in tqdm(
                pool.imap_unordered(_encode_chunk, tasks), total=nchunks, desc="编码分块"
            ):
                parts[idx] = (part_path, cnt)
        # 按块序拼接分块文件（裸 uint16 直接字节拼接即可）
        total_tokens = 0
        with open(args.output, "wb") as fout:
            for part_path, cnt in parts:
                with open(part_path, "rb") as pf:
                    shutil.copyfileobj(pf, fout)
                os.remove(part_path)
                total_tokens += cnt
    else:
        total_tokens = _encode_serial(args.input, args.output, tokenizer, dtype, args.buffer_tokens)

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
