"""输出守卫：把「有没有引用」从提示词的软约束变成程序校验。

原实现只在提示词里重复三遍「不要编造」，模型仍然可以照编；这里在返回前
检查引用编号是否落在真实候选范围内，不合规就降级为拒答。
"""

from __future__ import annotations

import re

CITE_RE = re.compile(r"\[(\d+)\]")
ABSTAIN = "资料中未提及该信息。"


def cited_indices(answer: str) -> list[int]:
    return sorted({int(m) for m in CITE_RE.findall(answer)})


def is_abstain(answer: str) -> bool:
    return answer.strip() in (ABSTAIN, ABSTAIN.rstrip("。"))


def validate(answer: str, n_hits: int) -> tuple[bool, str]:
    if is_abstain(answer):
        return True, "abstain"
    if not answer.strip():
        return False, "empty"
    idx = cited_indices(answer)
    if not idx:
        return False, "no-citation"
    if any(i < 1 or i > n_hits for i in idx):
        return False, "citation-out-of-range"
    return True, "ok"


def finalize(answer: str, n_hits: int, strict: bool = True) -> tuple[str, str]:
    """返回（最终回答，处理原因）。strict 模式下不合规输出一律改成拒答。"""
    ok, reason = validate(answer, n_hits)
    if ok:
        return answer.strip(), reason
    if strict:
        return ABSTAIN, reason
    return answer.strip(), reason
