"""Fabricated fixtures; tests never load private health records."""

import os
import json
import tempfile
from pathlib import Path

_test_root = tempfile.TemporaryDirectory(prefix="health-rag-test-")
_index = Path(_test_root.name)
_patients = ["张三九", "李四六", "王五"]
_chunks = [
    dict(
        chunk_id=f"fiction-{i}",
        patient=n,
        section="一、基本信息",
        page=i + 1,
        text=f"{n}，虚构测试档案，血型 O 型，BMI 22。",
    )
    for i, n in enumerate(_patients)
]
(_index / "chunks.jsonl").write_text(
    "\n".join(json.dumps(c, ensure_ascii=False) for c in _chunks), encoding="utf-8"
)
(_index / "meta.json").write_text(
    json.dumps({"patients": _patients, "vector": False}), encoding="utf-8"
)
os.environ["RAG_INDEX_DIR"] = str(_index)
os.environ["LLM_API_KEY"] = ""
