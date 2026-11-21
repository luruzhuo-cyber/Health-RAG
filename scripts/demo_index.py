"""Create explicitly fabricated fixtures; never use actual patient files."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import load_settings
from app.index import IndexBundle, save_index
from app.ingest.chunker import Chunk
from app.retrieval.bm25 import BM25Index
from app.retrieval.vector import Embedder, VectorIndex
import argparse

p = argparse.ArgumentParser()
p.add_argument("--with-vectors", action="store_true")
args = p.parse_args()
s = load_settings()
chunks = [
    Chunk(
        chunk_id="demo-a",
        patient="演示甲",
        section="虚构基本信息",
        page=1,
        text="演示甲：虚构学习档案，血型 A 型，身高 170 厘米。",
    ),
    Chunk(
        chunk_id="demo-b",
        patient="演示乙",
        section="虚构基本信息",
        page=2,
        text="演示乙：虚构学习档案，血型 B 型，身高 165 厘米。 ",
    ),
]
v = VectorIndex.build(chunks, Embedder(s)) if args.with_vectors else None
if v is not None:
    v.model_identity = s.embedding_identity
save_index(IndexBundle(chunks, ["演示甲", "演示乙"], BM25Index(chunks), v), s.index_dir)
print("Synthetic demo index ready; ask 演示甲的血型是什么")
