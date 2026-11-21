import pytest

from app.config import load_settings
from app.index import load_index
from app.ingest.chunker import Chunk
from app.retrieval.fusion import rrf_fuse
from app.retrieval.pipeline import Retriever


def _chunk(cid: str, patient: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=cid, patient=patient, section="一、基本信息", page=1, text=text
    )


def test_rrf_prefers_item_in_both_lists():
    a = _chunk("a", "张三九", "血型 O 型")
    b = _chunk("b", "张三九", "过敏史")
    fused = rrf_fuse([[(a, 9.0), (b, 1.0)], [(b, 0.9), (a, 0.8)]], k=60, top_k=2)
    assert [c.chunk_id for c, _ in fused] == ["a", "b"]


def test_rrf_deduplicates():
    a = _chunk("a", "张三九", "血型 O 型")
    fused = rrf_fuse([[(a, 1.0)], [(a, 1.0)]], k=60, top_k=5)
    assert len(fused) == 1


@pytest.fixture(scope="module")
def retriever():
    settings = load_settings()
    return Retriever(load_index(settings), settings)


def test_patient_filter_blocks_cross_record(retriever):
    hits, trace = retriever.search(
        "李四六的血型是什么", mode="bm25", top_k=5, patient="李四六"
    )
    assert trace.patient == "李四六"
    assert hits and all(c.patient == "李四六" for c, _ in hits)


def test_patient_resolution_from_question(retriever):
    assert retriever.resolve_patient("王五的BMI是多少") == "王五"
    assert retriever.resolve_patient("三份档案里谁的BMI最高") is None
