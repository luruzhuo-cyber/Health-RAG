"""章节内按句切块：控制单块长度，保留句间重叠，避免一句话被截断。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, asdict

from .loader import Segment

SENT_SPLIT = re.compile(r"(?<=[。！？；!?;])")


@dataclass
class Chunk:
    chunk_id: str
    patient: str
    section: str
    page: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENT_SPLIT.split(text)]
    return [p for p in parts if p]


def _make_id(patient: str, section: str, page: int, idx: int, text: str) -> str:
    digest = hashlib.sha1(
        f"{patient}|{section}|{page}|{idx}|{text}".encode("utf-8")
    ).hexdigest()[:8]
    return f"p{page}-{digest}"


def chunk_segment(
    seg: Segment, max_chars: int = 260, overlap_chars: int = 60
) -> list[Chunk]:
    sentences = split_sentences(seg.text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    buf: list[str] = []
    size = 0
    added = False  # 上次切块后是否又拼进了新句子，用来避免只含重叠部分的重复块

    def flush() -> None:
        nonlocal buf, size, added
        if not buf:
            return
        text = "".join(buf)
        chunks.append(
            Chunk(
                chunk_id=_make_id(
                    seg.patient, seg.section, seg.page, len(chunks), text
                ),
                patient=seg.patient,
                section=seg.section,
                page=seg.page,
                text=text,
            )
        )
        # 以句为单位回退 overlap_chars，保证下一块从完整句子开始
        tail: list[str] = []
        acc = 0
        for sent in reversed(buf):
            if acc + len(sent) > overlap_chars:
                break
            tail.insert(0, sent)
            acc += len(sent)
        buf = tail
        size = acc
        added = False

    for sent in sentences:
        if size and size + len(sent) > max_chars:
            flush()
        # 单句本身超长时按窗口硬切，避免出现超过 max_chars 的块
        if len(sent) > max_chars:
            for i in range(0, len(sent), max_chars):
                piece = sent[i : i + max_chars]
                chunks.append(
                    Chunk(
                        chunk_id=_make_id(
                            seg.patient, seg.section, seg.page, len(chunks), piece
                        ),
                        patient=seg.patient,
                        section=seg.section,
                        page=seg.page,
                        text=piece,
                    )
                )
            buf, size, added = [], 0, False
            continue
        buf.append(sent)
        size += len(sent)
        added = True
    if buf and added:
        text = "".join(buf)
        chunks.append(
            Chunk(
                chunk_id=_make_id(
                    seg.patient, seg.section, seg.page, len(chunks), text
                ),
                patient=seg.patient,
                section=seg.section,
                page=seg.page,
                text=text,
            )
        )
    return chunks
