"""问答主链路：检索 → 组证据 → 生成 → 引用校验。"""

from __future__ import annotations

from dataclasses import asdict

from ..config import Settings
from ..index import IndexBundle, load_index
from ..llm import ChatClient
from ..retrieval.pipeline import Retriever, format_context
from .guard import ABSTAIN, finalize
from .prompt import build_messages


class Answerer:
    def __init__(self, settings: Settings, bundle: IndexBundle | None = None):
        self.settings = settings
        self.bundle = bundle or load_index(settings)
        self.retriever = Retriever(self.bundle, settings)
        self.client = ChatClient(settings) if settings.has_llm else None

    def _sources(self, hits) -> list[dict]:
        return [
            {
                "index": i,
                "chunk_id": c.chunk_id,
                "patient": c.patient,
                "section": c.section,
                "page": c.page,
                "score": round(s, 4),
                "text": c.text,
            }
            for i, (c, s) in enumerate(hits, start=1)
        ]

    def answer(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int | None = None,
        use_rerank: bool = False,
    ) -> dict:
        hits, trace = self.retriever.search(
            query, mode=mode, top_k=top_k, use_rerank=use_rerank
        )
        base = {"query": query, "trace": asdict(trace), "sources": self._sources(hits)}
        if not hits:
            return {**base, "answer": ABSTAIN, "guard": "no-hit"}
        if self.client is None:
            return {**base, "answer": None, "guard": "no-llm-key"}
        raw = self.client.chat(build_messages(format_context(hits), query))
        text, reason = finalize(raw, len(hits), strict=True)
        return {**base, "answer": text, "guard": reason, "raw": raw}
