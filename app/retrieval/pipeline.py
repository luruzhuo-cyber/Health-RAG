"""检索编排：档案过滤 → 单路或混合召回 → 可选重排。"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Settings
from ..index import IndexBundle
from ..ingest.chunker import Chunk
from .fusion import rrf_fuse
from .rerank import build_reranker
from .vector import Embedder


@dataclass
class RetrievalTrace:
    patient: str | None = None
    mode: str = "hybrid"
    used: list[str] = field(default_factory=list)
    rerank: str = ""
    degraded: str = ""


class Retriever:
    def __init__(self, bundle: IndexBundle, settings: Settings):
        self.bundle = bundle
        self.settings = settings
        self._embedder: Embedder | None = None
        self._reranker = None

    def _embed(self, query: str):
        if self._embedder is None:
            self._embedder = Embedder(self.settings)
        return self._embedder.embed([query])

    def resolve_patient(self, query: str) -> str | None:
        """问题里点名了患者就锁定该患者，避免同一主题串到别人档案上。"""
        hits = [p for p in self.bundle.patients if p in query]
        return hits[0] if len(hits) == 1 else None

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int | None = None,
        patient: str | None = None,
        use_rerank: bool = False,
        use_filter: bool = True,
    ) -> tuple[list[tuple[Chunk, float]], RetrievalTrace]:
        if mode not in ("bm25", "vector", "hybrid"):
            raise ValueError("Unsupported retrieval mode")
        top_k = top_k if top_k is not None else self.settings.top_k
        if not 1 <= top_k <= 100:
            raise ValueError("top_k must be 1..100")
        if use_filter and (
            (patient is not None and patient not in self.bundle.patients)
            or (patient is None and self.resolve_patient(query) is None)
        ):
            return [], RetrievalTrace(
                mode=mode, degraded="patient-required-or-ambiguous"
            )
        if patient is None and use_filter:
            patient = self.resolve_patient(query)
        trace = RetrievalTrace(patient=patient, mode=mode)

        rankings: list[list[tuple[Chunk, float]]] = []
        if mode in ("bm25", "hybrid"):
            rankings.append(
                self.bundle.bm25.search(query, top_k=top_k, patient=patient)
            )
            trace.used.append("bm25")
        if mode in ("vector", "hybrid"):
            if self.bundle.vector is not None:
                rankings.append(
                    self.bundle.vector.search(
                        self._embed(query), top_k=top_k, patient=patient
                    )
                )
                trace.used.append("vector")
            else:
                trace.degraded = "no-vector-index"

        if mode == "hybrid" and len(rankings) == 2:
            hits = rrf_fuse(rankings, k=self.settings.rrf_k, top_k=top_k)
        elif rankings:
            hits = rankings[0][:top_k]
        else:
            hits = []

        if use_rerank and hits:
            if self._reranker is None:
                self._reranker = build_reranker(self.settings.rerank_model)
            hits = self._reranker.rerank(query, hits, top_k=top_k)
            trace.rerank = self._reranker.name
            if self._reranker.name == "lexical":
                trace.degraded = (
                    trace.degraded + ";" if trace.degraded else ""
                ) + "lexical-rerank-no-crossencoder"
        return hits, trace


def format_context(hits: list[tuple[Chunk, float]]) -> str:
    lines = []
    for i, (chunk, _) in enumerate(hits, start=1):
        lines.append(
            f"[{i}] 患者：{chunk.patient}｜章节：{chunk.section}｜第 {chunk.page} 页\n{chunk.text}"
        )
    return "\n\n".join(lines)
