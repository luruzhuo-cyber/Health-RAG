"""PDF 读取与档案结构识别。

原实现把整页文本按空行拼成段落，丢失页码与章节，无法做引用；这里保留
页码、患者、章节三级元数据，供后续过滤与溯源使用。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

CN_NUM = "一二三四五六七八九十"
SECTION_RE = re.compile(rf"^\s*([{CN_NUM}]+)、\s*([\u4e00-\u9fa5]{{2,8}})\s*$", re.M)
RECORD_HEAD = re.compile(r"^\s*健康档案\s*$", re.M)
NAME_BY_ROLE = re.compile(r"患者姓名为\s*([\u4e00-\u9fa5]{2,4})")
NAME_BY_INTRO = re.compile(r"([\u4e00-\u9fa5]{2,4})\s*[，,]\s*[男女][，,]\s*出生于")


@dataclass
class Page:
    number: int
    text: str


@dataclass
class Segment:
    """一段属于同一患者、同一章节的连续文本。"""

    patient: str
    section: str
    page: int
    text: str


def read_pages(pdf_path: str | Path) -> list[Page]:
    doc = pymupdf.open(str(pdf_path))
    return [Page(number=i, text=p.get_text("text")) for i, p in enumerate(doc, start=1)]


def detect_name(block: str) -> str:
    m = NAME_BY_ROLE.search(block) or NAME_BY_INTRO.search(block)
    return m.group(1) if m else ""


def split_segments(pages: list[Page]) -> list[Segment]:
    """跨页维护「当前患者」与「当前章节」：一页可能以别人档案的中段开头，
    姓名与章节都要沿页延续，否则续页开头的文字会被当成标题区丢掉。"""
    segments: list[Segment] = []
    current = ""
    current_section = ""
    for page in pages:
        for part in RECORD_HEAD.split(page.text):
            if not part.strip():
                continue
            head_end = SECTION_RE.search(part)
            head = part[: head_end.start()] if head_end else part
            name = detect_name(head) or detect_name(part[:120])
            if name and name != current:
                current, current_section = name, ""
            if not current:
                continue
            matches = list(SECTION_RE.finditer(part))
            if head.strip() and current_section:
                segments.append(
                    Segment(
                        patient=current,
                        section=current_section,
                        page=page.number,
                        text=head.strip(),
                    )
                )
            for idx, m in enumerate(matches):
                start = m.end()
                end = matches[idx + 1].start() if idx + 1 < len(matches) else len(part)
                body = part[start:end].strip()
                current_section = f"{m.group(1)}、{m.group(2)}"
                if body:
                    segments.append(
                        Segment(
                            patient=current,
                            section=current_section,
                            page=page.number,
                            text=body,
                        )
                    )
    return segments


def list_patients(segments: list[Segment]) -> list[str]:
    seen: list[str] = []
    for seg in segments:
        if seg.patient not in seen:
            seen.append(seg.patient)
    return seen
