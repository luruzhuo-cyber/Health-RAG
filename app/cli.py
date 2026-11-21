"""命令行入口：建索引、单问调试、批量评测。"""

from __future__ import annotations

import argparse
import json
import sys

from .config import load_settings
from .index import build_index, save_index


def cmd_ingest(args: argparse.Namespace) -> None:
    settings = load_settings()
    bundle = build_index(settings, with_vectors=args.with_vectors)
    save_index(bundle, settings.index_dir)
    print(
        json.dumps(
            {
                "chunks": len(bundle.chunks),
                "patients": bundle.patients,
                "vector": bundle.has_vector,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    for c in bundle.chunks[:3]:
        print(
            f"- {c.chunk_id} {c.patient}/{c.section}/p{c.page} ({len(c.text)}字) {c.text[:40]}"
        )


def cmd_ask(args: argparse.Namespace) -> None:
    from .generate.answer import Answerer

    settings = load_settings()
    result = Answerer(settings).answer(
        args.query, mode=args.mode, top_k=args.top_k, use_rerank=args.rerank
    )
    print(
        json.dumps(
            {
                "answer": result["answer"],
                "guard": result["guard"],
                "trace": result["trace"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    for s in result["sources"]:
        print(
            f"[{s['index']}] {s['patient']}/{s['section']}/p{s['page']} score={s['score']}"
        )


def cmd_eval(args: argparse.Namespace) -> None:
    from eval.run_eval import run

    run(
        modes=args.modes.split(","),
        use_rerank=args.rerank,
        top_k=args.top_k,
        out=args.out,
        questions_path=args.questions,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="health-rag")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="重建索引")
    p_ingest.add_argument(
        "--with-vectors",
        action="store_true",
        help="同时生成稠密向量（本地模型或已配置的 Embedding API）",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="单条提问")
    p_ask.add_argument("query")
    p_ask.add_argument("--mode", default="hybrid", choices=["bm25", "vector", "hybrid"])
    p_ask.add_argument("--top-k", type=int, default=None)
    p_ask.add_argument("--rerank", action="store_true")
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("eval", help="跑评测集")
    p_eval.add_argument("--modes", default="bm25,hybrid")
    p_eval.add_argument("--rerank", action="store_true")
    p_eval.add_argument("--top-k", type=int, default=5)
    p_eval.add_argument("--out", default="")
    p_eval.add_argument("--questions", default="")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
