"""稠密检索：OpenAI 兼容协议的 embedding + 内存余弦检索。

不同网关（OpenAI、通义、本地 TEI）都遵循 /v1/embeddings，切换只改环境变量。
"""

from __future__ import annotations

import time

import numpy as np

from ..config import Settings
from ..ingest.chunker import Chunk


class Embedder:
    def __init__(self, settings: Settings, batch_size: int = 16):
        self.settings = settings
        self.batch_size = batch_size
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.settings.embed_base_url,
                api_key=self.settings.embed_api_key,
                timeout=self.settings.llm_timeout,
                max_retries=0,
            )
        return self._client

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        if self.settings.embed_backend == "local":
            if not hasattr(self, "_local_model"):
                from sentence_transformers import SentenceTransformer

                self._local_model = SentenceTransformer(
                    self.settings.local_embed_model,
                    local_files_only=True,
                    revision=(
                        "5617a9f61b028005a4858fdac845db406aefb181"
                        if self.settings.local_embed_model == "BAAI/bge-m3"
                        else None
                    ),
                )
            return np.asarray(
                self._local_model.encode(
                    texts, batch_size=self.batch_size, normalize_embeddings=True
                ),
                dtype=np.float32,
            )
        client = self._get_client()
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            for attempt in range(self.settings.llm_retries + 1):
                try:
                    resp = client.embeddings.create(
                        model=self.settings.embed_model, input=batch
                    )
                    vectors.extend([d.embedding for d in resp.data])
                    break
                except Exception as exc:
                    from openai import (
                        APIConnectionError,
                        APITimeoutError,
                        APIStatusError,
                    )

                    transient = isinstance(
                        exc, (APIConnectionError, APITimeoutError)
                    ) or (
                        isinstance(exc, APIStatusError)
                        and (exc.status_code in (408, 429) or exc.status_code >= 500)
                    )
                    if not transient or attempt >= self.settings.llm_retries:
                        raise RuntimeError("Embedding provider unavailable") from None
                    time.sleep(min(8.0, 1.5 * (2**attempt)))
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-9, None)


class VectorIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        if len(chunks) != len(vectors):
            raise ValueError(f"chunk 数与向量数不一致: {len(chunks)} vs {len(vectors)}")
        self.chunks = chunks
        import faiss

        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.faiss = faiss.IndexFlatIP(self.vectors.shape[1])
        self.faiss.add(self.vectors)

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> "VectorIndex":
        return cls(chunks, embedder.embed([c.text for c in chunks]))

    def search(
        self, query_vec: np.ndarray, top_k: int = 5, patient: str | None = None
    ) -> list[tuple[Chunk, float]]:
        q = query_vec.reshape(-1)
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        distances, indices = self.faiss.search(
            np.asarray([q], dtype=np.float32), len(self.chunks)
        )
        ranked = indices[0]
        scores = dict(zip(indices[0], distances[0]))
        out: list[tuple[Chunk, float]] = []
        for i in ranked:
            chunk = self.chunks[int(i)]
            if patient and chunk.patient != patient:
                continue
            out.append((chunk, float(scores[int(i)])))
            if len(out) >= top_k:
                break
        return out
