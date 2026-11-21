"""索引构建与持久化：chunks.jsonl 存元数据，vectors.npy 存稠密向量（可选）。"""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Settings
from .ingest.chunker import Chunk, chunk_segment
from .ingest.loader import list_patients, read_pages, split_segments
from .retrieval.bm25 import BM25Index
from .retrieval.vector import Embedder, VectorIndex


@dataclass
class IndexBundle:
    chunks: list[Chunk]
    patients: list[str]
    bm25: BM25Index
    vector: VectorIndex | None
    rerank_name: str = ""

    @property
    def has_vector(self) -> bool:
        return self.vector is not None


def build_chunks(settings: Settings) -> tuple[list[Chunk], list[str]]:
    pages = read_pages(settings.source_pdf)
    segments = split_segments(pages)
    chunks: list[Chunk] = []
    for seg in segments:
        chunks.extend(
            chunk_segment(seg, settings.chunk_max_chars, settings.chunk_overlap_chars)
        )
    return chunks, list_patients(segments)


def build_index(settings: Settings, with_vectors: bool = False) -> IndexBundle:
    chunks, patients = build_chunks(settings)
    vector = None
    if with_vectors and settings.has_embedding:
        vector = VectorIndex.build(chunks, Embedder(settings))
        vector.model_identity = settings.embedding_identity
    return IndexBundle(
        chunks=chunks, patients=patients, bm25=BM25Index(chunks), vector=vector
    )


def save_index(bundle: IndexBundle, index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)

    def replace_bytes(name, content):
        fd, temporary = tempfile.mkstemp(prefix=".index-", dir=index_dir)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, index_dir / name)
        finally:
            Path(temporary).unlink(missing_ok=True)

    chunks_data = (
        "\n".join(json.dumps(c.to_dict(), ensure_ascii=False) for c in bundle.chunks)
        + "\n"
    ).encode("utf-8")
    vector_sha = ""
    if bundle.vector is not None:
        import io

        buffer = io.BytesIO()
        np.save(buffer, bundle.vector.vectors)
        vector_data = buffer.getvalue()
        vector_sha = hashlib.sha256(vector_data).hexdigest()
        replace_bytes("vectors.npy", vector_data)
    else:
        (index_dir / "vectors.npy").unlink(missing_ok=True)
    replace_bytes("chunks.jsonl", chunks_data)
    metadata = {
        "patients": bundle.patients,
        "chunks": len(bundle.chunks),
        "vector": bundle.has_vector,
        "model_identity": getattr(bundle.vector, "model_identity", ""),
        "vectors_sha256": vector_sha,
        "chunks_sha256": hashlib.sha256(chunks_data).hexdigest(),
    }
    replace_bytes("meta.json", json.dumps(metadata, ensure_ascii=False).encode("utf-8"))


def load_index(settings: Settings) -> IndexBundle:
    index_dir = settings.index_dir
    path = index_dir / "chunks.jsonl"
    if not path.exists():
        return build_index(settings)
    chunks = [
        Chunk(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    meta = (
        json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        if (index_dir / "meta.json").exists()
        else {}
    )
    vector = None
    vec_path = index_dir / "vectors.npy"
    if (
        vec_path.exists()
        and meta.get("vector")
        and settings.has_embedding
        and meta.get("model_identity") == (settings.embedding_identity)
        and meta.get("chunks_sha256") == hashlib.sha256(path.read_bytes()).hexdigest()
        and meta.get("vectors_sha256")
        == hashlib.sha256(vec_path.read_bytes()).hexdigest()
    ):
        vector = VectorIndex(chunks, np.load(vec_path))
    return IndexBundle(
        chunks=chunks,
        patients=sorted({c.patient for c in chunks}),
        bm25=BM25Index(chunks),
        vector=vector,
    )
