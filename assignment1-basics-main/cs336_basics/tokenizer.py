"""BPE 分词器（Byte-Pair Encoding Tokenizer）。

对应讲义 §2「Byte-Pair Encoding (BPE) Tokenizer」。

本模块包含两部分:
  1. ``train_bpe``: 在给定语料上从零训练一个字节级 BPE 分词器,
     产出 ``vocab``（id -> bytes）与 ``merges``（按创建顺序排列的合并规则）。
  2. ``Tokenizer`` 类: 用训练得到的 ``vocab`` / ``merges`` / ``special_tokens``
     对文本做编码（``encode`` / ``encode_iterable``）与解码（``decode``）。

实现要点（详见讲义 §2）:
  - 字节级（byte-level）: 初始词表为 256 个单字节, 外加用户给定的特殊 token。
  - 预分词（pre-tokenization）: 用 GPT-2 的正则把文本切成「预 token」, 合并只发生在
    预 token 内部, 不跨预 token 边界。GPT-2 正则模式见 ``PAT`` 常量。
  - 特殊 token（如 ``<|endoftext|>``）永远作为一个整体, 绝不可被切分或参与合并。
  - 训练大语料时可借助 ``cs336_basics/pretokenization_example.py`` 中的
    ``find_chunk_boundaries`` 在特殊 token 边界处并行预分块, 各块独立统计词频后汇总。
"""

import os
import regex as re
from collections.abc import Iterable, Iterator
from typing import BinaryIO
from collections import defaultdict
import json, base64


# GPT-2 预分词正则模式（见讲义 §2.4）。用 `regex` 模块（非内置 `re`）以支持 \p{L} 等。
# 该模式把文本切成「预 token」: 缩写、字母串、数字串、标点串、空白串等。
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """在给定语料上训练一个字节级 BPE 分词器。

    Args:
        input_path (str | os.PathLike): BPE 训练语料文件路径（UTF-8 文本）。
        vocab_size (int): 最终词表大小上限（含 256 个初始字节与全部特殊 token）。
        special_tokens (list[str]): 特殊 token 列表（如 ``["<|endoftext|>"]"``）;
            这些字符串永远作为单个 token, 不会被切分, 也不参与字节对合并。
        **kwargs: 透传的可选参数（如并行进程数等, 实现时可自行约定/忽略）。

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab: id -> token 字节串 的映射。
            merges: 按创建先后顺序排列的合并规则列表, 每项为 ``(token1, token2)``,
                    表示训练中把 ``token1`` 与 ``token2`` 合并成了新 token。
    """
    ###########################################################################
    # TODO: 实现 BPE 训练（见讲义 §2.2 ~ §2.5）。整体步骤:
    #   1) 初始化词表 vocab: 先放入 256 个单字节 {i: bytes([i]) for i in range(256)},
    #      再把每个 special_token 以其 UTF-8 字节追加为新 id（特殊 token 不可被切分）。
    #   2) 预分词: 读入语料, 先在 special_tokens 处切分文本（用 re.split 加转义后的
    #      "|".join, 保证特殊 token 边界不被跨越）, 再对每段用 GPT-2 正则 PAT 切出预 token。
    #      大文件可用 pretokenization_example.find_chunk_boundaries 在 b"<|endoftext|>"
    #      等特殊 token 边界并行分块, 各块独立统计后汇总词频。
    #   3) 把每个预 token 表示为「字节序列」(初始每个元素是一个长度 1 的 bytes), 统计其出现频次,
    #      形成 词频表: dict[tuple[bytes, ...], int]。
    #   4) 迭代合并: 重复直到 len(vocab) 达到 vocab_size:
    #        a. 统计所有相邻字节对 (a, b) 的总频次（按所在预 token 的词频加权）。
    #        b. 选频次最高的对; 若并列, 选「字典序更大」的对（讲义约定的 tie-break）。
    #        c. 新建 token = a + b, 加入 vocab 与 merges; 在所有预 token 中把相邻的 (a, b)
    #           就地替换为合并后的 token。
    #   提示: 形状/类型 —— vocab: dict[int, bytes]; merges: list[tuple[bytes, bytes]];
    #         词频表的键是 tuple[bytes, ...]。可用 collections.Counter / defaultdict 提速。
    #   提示: tie-break 取最大对可写 max(pair_counts, key=lambda p: (pair_counts[p], p))。
    ###########################################################################
    with open(input_path, encoding="utf-8") as f:
        text = f.read()
    
    vocab = {i: bytes([i]) for i in range(256)}
    for i, special_token in enumerate(special_tokens):
        vocab[256 + i] = special_token.encode("utf-8")
    
    if special_tokens:
        delimiter = "|".join(re.escape(tok) for tok in special_tokens)
        segments = re.split(delimiter, text)
    else:
        segments = [text]
    
    word_freq = defaultdict(int)
    for segment in segments:
        for m in re.finditer(PAT, segment):
            pre_token = m.group()
            key = tuple(bytes([b]) for b in pre_token.encode("utf-8"))
            word_freq[key] += 1
            
    merges = []
    num_merges = vocab_size - len(vocab)    
    for _ in range(num_merges):
        pair_count = defaultdict(int)
        for word, freq in word_freq.items():
            for a, b in zip(word, word[1:]):
                pair_count[(a, b)] += freq
        
        if not pair_count:
            break
        
        best = max(pair_count, key=lambda p: (pair_count[p], p))
        
        a, b = best
        new_token = a + b
        vocab[len(vocab)] = new_token
        merges.append(best)
        word_freq = _merge_in_words(word_freq, best, new_token)
    
    return vocab, merges

def _merge_in_words(word_freq, best, new_token):
    a, b = best
    new_word_freq = defaultdict(int)
    for word, freq in word_freq.items():
        new_word = []
        l = len(word)
        i = 0
        while i < l - 1:
            cur, nxt = word[i], word[i + 1]
            if cur == a and nxt == b:
                new_word.append(new_token)
                i += 2
            else:
                new_word.append(cur)
                i += 1
        if i == l - 1:
            new_word.append(word[i])
        new_word_freq[tuple(new_word)] += freq
    
    return new_word_freq              
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


class Tokenizer:
    """基于 BPE 的字节级分词器（见讲义 §2）。

    用训练好的 ``vocab`` 与 ``merges`` 对文本做编码/解码, 并能正确处理特殊 token。
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """构造分词器。

        Args:
            vocab (dict[int, bytes]): id -> token 字节串 的映射。
            merges (list[tuple[bytes, bytes]]): 按创建顺序排列的合并规则。
            special_tokens (list[str] | None): 特殊 token 列表; 这些字符串编码时
                永远映射为单个 token, 解码时原样还原, 且不参与字节对合并。
        """
        #######################################################################
        # TODO: 保存分词器状态（见讲义 §2）。建议:
        #   - self.vocab = vocab; self.merges = merges
        #   - self.special_tokens = special_tokens or []
        #   - 为加速 encode, 预先构建「反向词表」: bytes -> id, 即
        #       self.byte_to_id = {tok: idx for idx, tok in vocab.items()}
        #   - 为加速合并, 可把 merges 转成「合并优先级」字典:
        #       self.merge_rank = {pair: i for i, pair in enumerate(merges)}（i 越小越先合并）
        #   - 若有 special_tokens, 确保它们的字节串都已在 vocab 中（不在则需新增 id）;
        #     并预编译一个用于「按特殊 token 切分」的正则（注意 re.escape, 且按长度降序
        #     排列以正确处理一个特殊 token 是另一个前缀的情形）。
        #######################################################################
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        
        self.byte_to_id = {b: i for i, b in vocab.items()}
        self.merge_rank = {pair: i for i, pair in enumerate(merges)}
        
        if self.special_tokens:
            specials = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "(" + "|".join(re.escape(t) for t in specials) + ")"
            self.special_pattern = re.compile(pattern)
        else:
            self.special_pattern = None
        
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """从磁盘上的 vocab / merges 文件构造 ``Tokenizer``。

        Args:
            vocab_filepath (str | os.PathLike): 序列化的词表文件路径。
            merges_filepath (str | os.PathLike): 序列化的合并规则文件路径。
            special_tokens (list[str] | None): 特殊 token 列表。

        Returns:
            Tokenizer: 由文件内容构造的分词器实例。
        """
        #######################################################################
        # TODO: 从文件加载 vocab/merges 并返回 cls(vocab, merges, special_tokens)。
        #   - 按你训练脚本采用的序列化格式（如 GPT-2 风格的 json + txt, 或 pickle）反序列化:
        #       vocab: 还原为 dict[int, bytes]（注意把字符串/十六进制等转回 bytes）;
        #       merges: 还原为 list[tuple[bytes, bytes]]（每行两个 token, 解析为 bytes 二元组）。
        #   - 最后:  return cls(vocab, merges, special_tokens)
        #   提示: 保持与 run_bpe_training.py 的写出格式一致, 才能正确读回。
        #######################################################################
        with open(vocab_filepath) as f:
            raw = json.load(f)
            
        vocab = {int(tid): base64.b64decode(s) for tid, s in raw.items()}
        
        merges = []
        with open(merges_filepath) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                a, b = line.split(" ")
                merges.append((base64.b64decode(a), base64.b64decode(b)))
        
        return cls(vocab, merges, special_tokens)
        
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

    def encode(self, text: str) -> list[int]:
        """把一段文本编码为 token id 列表。

        Args:
            text (str): 待编码的输入文本。

        Returns:
            list[int]: 对应的 token id 序列。
        """
        #######################################################################
        # TODO: 实现 BPE 编码（见讲义 §2.5）。步骤:
        #   1) 若存在 special_tokens, 先用预编译正则把 text 切成「特殊段」与「普通段」,
        #      使特殊 token 作为整体保留（普通文本不会跨越特殊 token 边界）。
        #   2) 特殊段: 直接查 self.byte_to_id[special.encode("utf-8")] 得到其单一 id。
        #   3) 普通段: 用 GPT-2 正则 PAT 切成预 token; 每个预 token 先转为字节序列
        #      list[bytes]（每元素 1 字节）, 然后反复按 self.merge_rank 选「优先级最高
        #      （rank 最小）的相邻可合并对」就地合并, 直到无可合并对; 再把每个最终字节段
        #      经 self.byte_to_id 映射为 id。
        #   提示: 形状/类型 —— 返回 list[int]; 中间用 list[bytes] 表示一个预 token。
        #   提示: 选最优对可写 min(候选pair, key=lambda p: self.merge_rank[p])。
        #######################################################################
        ids = []
        
        if self.special_pattern:
            chunks = self.special_pattern.split(text)
        else:
            chunks = [text]
        
        specials_set = set(self.special_tokens)
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in specials_set:
                ids.append(self.byte_to_id[chunk.encode("utf-8")])
                continue
            for m in re.finditer(PAT, chunk):
                parts = [bytes([b]) for b in m.group().encode("utf-8")]
                parts = self._merge(parts)
                ids.extend(self.byte_to_id[p] for p in parts)
        
        return ids   

    def _merge(self, parts):
        while len(parts) >= 2:
            candidates = {pair for pair in zip(parts, parts[1:]) if pair in self.merge_rank}
            if not candidates:
                break
            best = min(candidates, key=lambda p: self.merge_rank[p])
            merged, i = [], 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == best:
                    merged.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        return parts    
    
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """惰性地对一个字符串可迭代对象（如逐行读取的文件句柄）编码。

        适用于无法一次性载入内存的大文件: 逐块产出 token id, 内存占用恒定。

        Args:
            iterable (Iterable[str]): 产出字符串片段的可迭代对象（例如打开的文件对象,
                迭代时按行产出）。

        Yields:
            int: 依次产出的 token id。
        """
        #######################################################################
        # TODO: 实现惰性编码（见讲义 §2.5）。要点:
        #   - 遍历 iterable 的每个字符串片段 chunk, 对其调用 self.encode(chunk),
        #     并用 `yield from` 逐个产出得到的 id（本函数应为生成器, 不要一次性 return 整个列表）。
        #   - 注意: 这里按片段独立编码即可（测试以逐行喂入为主）; 内存复杂度需与单片段成正比,
        #     而非与整个文件成正比。
        #######################################################################
        for chunk in iterable:
            yield from self.encode(chunk)
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

    def decode(self, ids: list[int]) -> str:
        """把 token id 列表解码回文本。

        Args:
            ids (list[int]): 待解码的 token id 序列。

        Returns:
            str: 解码得到的字符串。
        """
        #######################################################################
        # TODO: 实现 BPE 解码（见讲义 §2.5）。步骤:
        #   - 依次用 self.vocab[id] 取出每个 id 对应的 bytes, 并按顺序拼接成一个 bytes 串。
        #   - 对拼接后的 bytes 整体做一次 .decode("utf-8", errors="replace") 还原为字符串
        #     （errors="replace" 用于容错: 单个 token 的字节可能不是合法 UTF-8 起止,
        #      须先拼接全部字节再统一解码, 不可逐 token 解码）。
        #   提示: 形状/类型 —— 输入 list[int], 中间 bytes, 返回 str。
        #######################################################################
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
