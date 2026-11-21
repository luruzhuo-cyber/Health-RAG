"""重排：优先用 CrossEncoder 语义重排，未配置模型时退化为词面覆盖度重排。

两种实现都返回同一结构，pipeline 只关心排序结果，不关心用了哪种。
"""

from __future__ import annotations

from ..ingest.chunker import Chunk
from .bm25 import tokenize


class LexicalReranker:
    """无模型时的兜底：按查询词在候选块中的覆盖率重排。"""

    name = "lexical"

    def rerank(
        self, query: str, candidates: list[tuple[Chunk, float]], top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        q_terms = set(tokenize(query))
        if not q_terms:
            return candidates[:top_k]
        scored = []
        for chunk, base in candidates:
            doc_terms = set(tokenize(chunk.text))
            coverage = len(q_terms & doc_terms) / len(q_terms)
            scored.append((chunk, 0.7 * coverage + 0.3 * base))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:top_k]


class CrossEncoderReranker:
    """真正的语义重排，需要 sentence-transformers 与本地模型目录。"""

    name = "cross-encoder"

    def __init__(self, model_path: str):
        from sentence_transformers import (
            CrossEncoder,
        )  # 延迟导入，未安装时不影响其他功能

        self.model = CrossEncoder(
            model_path,
            max_length=512,
            local_files_only=True,
            revision=(
                "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
                if model_path == "BAAI/bge-reranker-v2-m3"
                else None
            ),
        )

    def rerank(
        self, query: str, candidates: list[tuple[Chunk, float]], top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []
        pairs = [(query, chunk.text) for chunk, _ in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(
            zip([c for c, _ in candidates], scores),
            key=lambda kv: float(kv[1]),
            reverse=True,
        )
        return [(chunk, float(score)) for chunk, score in ranked[:top_k]]


def build_reranker(model_path: str = ""):
    if model_path:
        try:
            return CrossEncoderReranker(model_path)
        except Exception:
            pass
    return LexicalReranker()
