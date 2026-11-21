import pytest

from app.config import load_settings
from app.ingest.chunker import chunk_segment
from app.ingest.loader import Segment, list_patients, read_pages, split_segments


@pytest.fixture(scope="module")
def segments():
    # Explicitly fabricated fixture; no source PDF required.
    return [
        Segment(
            patient=n,
            section="一、基本信息",
            page=i + 1,
            text=f"{n}，虚构测试资料，血型 O 型。",
        )
        for i, n in enumerate(["张三九", "李四六", "王五"])
    ]


def test_three_patients_detected(segments):
    assert list_patients(segments) == ["张三九", "李四六", "王五"]


def test_every_segment_has_metadata(segments):
    for seg in segments:
        assert seg.patient and seg.section and seg.page >= 1


def test_chunk_respects_max_chars(segments):
    for seg in segments:
        for chunk in chunk_segment(seg, max_chars=260, overlap_chars=60):
            assert len(chunk.text) <= 260
            assert chunk.page == seg.page
            assert chunk.patient == seg.patient


def test_overlap_creates_shared_sentence():
    text = "第一句话说明基本情况。" * 12
    seg = Segment(patient="测试", section="一、基本信息", page=1, text=text)
    chunks = chunk_segment(seg, max_chars=80, overlap_chars=30)
    assert len(chunks) > 1
    assert chunks[0].text[-6:] in chunks[1].text
