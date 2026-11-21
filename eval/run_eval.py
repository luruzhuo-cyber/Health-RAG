"""检索评测：对同一批标注问题跑多种策略，输出 P@1 / Recall@k / MRR@k。

对比组里 article_filter 是关键的消融项：关掉档案过滤，李四六的问题会被
张三九档案里「紧急联系人李四六」这类片段抢走名次。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from app.config import load_settings
from app.index import load_index
from app.retrieval.pipeline import Retriever

HERE = Path(__file__).resolve().parent


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def load_questions(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def first_hit_rank(hits, must: list[str], patient=None) -> int | None:
    keys = [_norm(m) for m in must]
    for rank, (chunk, _) in enumerate(hits, start=1):
        body = _norm(chunk.text)
        if (not patient or chunk.patient == patient) and any(k in body for k in keys):
            return rank
    return None


def evaluate(
    retriever: Retriever,
    questions: list[dict],
    mode: str,
    top_k: int,
    use_rerank: bool,
    use_filter: bool,
) -> dict:
    ranks: list[int | None] = []
    sizes: list[int] = []
    cross = 0
    traces = []
    for q in questions:
        if not q["answerable"]:
            continue
        patient = q["patient"] if use_filter else None
        hits, trace = retriever.search(
            q["q"],
            mode=mode,
            top_k=top_k,
            patient=patient,
            use_rerank=use_rerank,
            use_filter=use_filter,
        )
        traces.append(
            {"used": trace.used, "rerank": trace.rerank, "degraded": trace.degraded}
        )
        ranks.append(first_hit_rank(hits, q["must_contain"], q["patient"]))
        sizes.append(sum(len(c.text) for c, _ in hits))
        if hits and q["patient"] and any(c.patient != q["patient"] for c, _ in hits):
            cross += 1
    n = len(ranks)
    if not n:
        return {
            "mode": mode,
            "filter": use_filter,
            "n": 0,
            "error": "no-answerable-samples",
        }
    p1 = sum(1 for r in ranks if r == 1) / n
    recall = sum(1 for r in ranks if r) / n
    mrr = sum((1.0 / r) for r in ranks if r) / n
    return {
        "mode": mode,
        "filter": use_filter,
        "n": n,
        "p@1": round(p1, 4),
        "hit@k": round(recall, 4),
        "mrr@k": round(mrr, 4),
        "avg_context_chars": round(sum(sizes) / n, 1),
        "cross_record": round(cross / n, 4),
        "unanswerable_count": sum(not q["answerable"] for q in questions),
        "answer_quality_evaluated": False,
        "effective_backends": traces,
    }


def run(
    modes: list[str],
    use_rerank: bool,
    top_k: int,
    out: str = "",
    questions_path: str = "",
) -> None:
    settings = load_settings()
    bundle = load_index(settings)
    retriever = Retriever(bundle, settings)
    questions = load_questions(
        Path(questions_path) if questions_path else HERE / "demo_questions.jsonl"
    )

    rows = []
    for mode in modes:
        for use_filter in (True, False):
            rows.append(
                evaluate(retriever, questions, mode, top_k, use_rerank, use_filter)
            )

    header = (
        "| 策略 | 档案过滤 | 样本 | P@1 | Hit@%d | MRR@%d | 串档率 | 注入字符 |"
        % (top_k, top_k)
    )
    table = [header, "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r["n"]:
            table.append(
                f"| {r['mode']} | {r['filter']} | 0 | no-answerable-samples | - | - | - | - |"
            )
            continue
        table.append(
            f"| {r['mode']} | {'开' if r['filter'] else '关'} | {r['n']} | {r['p@1']:.2%} | "
            f"{r['hit@k']:.2%} | {r['mrr@k']:.3f} | {r['cross_record']:.2%} | "
            f"{r['avg_context_chars']:.0f} |"
        )
    report = "\n".join(table)

    print(
        f"chunks={len(bundle.chunks)} patients={bundle.patients} vector={bundle.has_vector} rerank={use_rerank}\n"
    )
    print(report)
    if out:
        Path(out).write_text(report + "\n", encoding="utf-8")
        Path(out).with_suffix(".json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nwritten: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="bm25")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default="")
    ap.add_argument("--questions", default="")
    a = ap.parse_args()
    run(a.modes.split(","), a.rerank, a.top_k, a.out, a.questions)
