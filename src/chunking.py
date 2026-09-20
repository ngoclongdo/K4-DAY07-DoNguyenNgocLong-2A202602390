from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        # Tách câu bằng lookbehind để giữ nguyên dấu câu (. ! ?), không làm mất dấu câu
        # Pattern match khoảng trắng/dòng mới ngay sau dấu câu: (?<=[.!?])\s+ hoặc (?<=\.)\n
        pattern = r'(?<=[.!?])\s+'
        raw_sentences = re.split(pattern, text.strip())
        sentences = [s.strip() for s in raw_sentences if s.strip()]

        if not sentences:
            return []

        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunk_str = " ".join(group).strip()
            if chunk_str:
                chunks.append(chunk_str)

        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not current_text:
            return []

        if len(current_text) <= self.chunk_size:
            return [current_text]

        # Base case 1: Không còn separator nào để thử -> cắt cứng theo chunk_size
        if not remaining_separators:
            return [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        sep = remaining_separators[0]
        next_seps = remaining_separators[1:]

        # Base case 2: sep rỗng "" -> tách từng ký tự rồi cắt theo chunk_size
        if sep == "":
            return [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        # Tách chuỗi theo separator hiện tại
        if sep in current_text:
            splits = current_text.split(sep)
        else:
            # Không tìm thấy separator hiện tại -> thử tiếp separator tiếp theo
            return self._split(current_text, next_seps)

        # Đệ quy xuống sâu cho các mảnh con vẫn vượt quá chunk_size
        refined_splits: list[str] = []
        for s in splits:
            if not s:
                continue
            if len(s) > self.chunk_size:
                refined_splits.extend(self._split(s, next_seps))
            else:
                refined_splits.append(s)

        # Gom lên (Merge): gộp các mảnh nhỏ liền kề cho tới sát chunk_size
        chunks: list[str] = []
        current_chunk = ""
        for piece in refined_splits:
            if not piece:
                continue
            if not current_chunk:
                current_chunk = piece
            else:
                candidate = current_chunk + sep + piece
                if len(candidate) <= self.chunk_size:
                    current_chunk = candidate
                else:
                    chunks.append(current_chunk)
                    current_chunk = piece

        if current_chunk:
            chunks.append(current_chunk)

        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(x * x for x in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    dot_product = _dot(vec_a, vec_b)
    return dot_product / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        fixed_chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=chunk_size // 10)
        sentence_chunker = SentenceChunker(max_sentences_per_chunk=3)
        recursive_chunker = RecursiveChunker(chunk_size=chunk_size)

        fixed_chunks = fixed_chunker.chunk(text)
        sentence_chunks = sentence_chunker.chunk(text)
        recursive_chunks = recursive_chunker.chunk(text)

        def _stats(chunks: list[str]) -> dict:
            count = len(chunks)
            avg_len = sum(len(c) for c in chunks) / count if count > 0 else 0.0
            return {
                "count": count,
                "avg_length": avg_len,
                "chunks": chunks,
            }

        return {
            "fixed_size": _stats(fixed_chunks),
            "by_sentences": _stats(sentence_chunks),
            "recursive": _stats(recursive_chunks),
        }
