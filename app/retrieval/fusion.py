"""结果融合：RRF 只依赖名次，不需要把 BM25 分数与余弦相似度强行归一。"""

from __future__ import annotations

from ..ingest.chunker import Chunk


def rrf_fuse(
    rankings: list[list[tuple[Chunk, float]]], k: int = 60, top_k: int = 5
) -> list[tuple[Chunk, float]]:
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, (chunk, _) in enumerate(ranking, start=1):
            chunks[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [(chunks[cid], score) for cid, score in ordered]
