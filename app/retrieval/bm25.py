"""BM25 稀疏检索：中文用 jieba 切词，支持按患者过滤。"""

from __future__ import annotations

import re

import jieba
from rank_bm25 import BM25Okapi

from ..ingest.chunker import Chunk

TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fa5]+")
STOPWORDS = {
    "的",
    "了",
    "是",
    "在",
    "和",
    "与",
    "为",
    "有",
    "他",
    "她",
    "它",
    "这",
    "那",
    "什么",
    "多少",
    "哪",
    "年",
    "月",
    "日",
}


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(text.lower()):
        if re.fullmatch(r"[a-z0-9]+", raw):
            tokens.append(raw)
            continue
        for tok in jieba.lcut(raw):
            tok = tok.strip()
            if len(tok) >= 2 and tok not in STOPWORDS:
                tokens.append(tok)
    return tokens or ["_empty"]


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.corpus = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus) if self.corpus else None

    def search(
        self, query: str, top_k: int = 5, patient: str | None = None
    ) -> list[tuple[Chunk, float]]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        out: list[tuple[Chunk, float]] = []
        for i in ranked:
            chunk = self.chunks[i]
            if patient and chunk.patient != patient:
                continue
            out.append((chunk, float(scores[i])))
            if len(out) >= top_k:
                break
        return out
