"""HTTP 服务：OpenAI 兼容的 /v1/chat/completions，外加健康检查与索引重建。

兼容协议的意义在于前端、小程序、扣子一类编排工具可以直接把 base_url 指过来，
不需要为这个服务单独写 SDK。
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import load_settings
from .generate.answer import Answerer
from .generate.guard import ABSTAIN
from .generate.prompt import build_messages
from .index import build_index, save_index
from .retrieval.pipeline import format_context

settings = load_settings()
app = FastAPI(title="health-rag", version="0.2.0")
_state: dict[str, Any] = {}


def get_answerer() -> Answerer:
    if "answerer" not in _state:
        _state["answerer"] = Answerer(settings)
    return _state["answerer"]


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = Field(default="health-rag")
    messages: list[Message]
    stream: bool = False
    temperature: float = 0.0
    mode: str = "hybrid"  # bm25 | vector | hybrid
    top_k: int | None = None
    rerank: bool = False


class IngestRequest(BaseModel):
    with_vectors: bool = False


@app.get("/health")
def health() -> dict:
    answerer = get_answerer()
    return {
        "status": "ok",
        "chunks": len(answerer.bundle.chunks),
        "patients": answerer.bundle.patients,
        "vector": answerer.bundle.has_vector,
        "llm": settings.has_llm,
    }


@app.post("/admin/ingest")
def ingest(req: IngestRequest) -> dict:
    bundle = build_index(settings, with_vectors=req.with_vectors)
    save_index(bundle, settings.index_dir)
    _state.pop("answerer", None)
    return {
        "chunks": len(bundle.chunks),
        "patients": bundle.patients,
        "vector": bundle.has_vector,
    }


def _last_user(messages: list[Message]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    raise HTTPException(status_code=400, detail="messages 中缺少 user 消息")


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    query = _last_user(req.messages)
    answerer = get_answerer()
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    try:
        result = answerer.answer(
            query, mode=req.mode, top_k=req.top_k, use_rerank=req.rerank
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid retrieval parameters")
    except Exception:
        raise HTTPException(
            status_code=502, detail="Upstream generation or retrieval unavailable"
        )
    if result["answer"] is None:
        raise HTTPException(status_code=503, detail="LLM is not configured")
    if req.stream:

        def sse():
            payload = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": result["answer"]},
                        "finish_reason": None,
                    }
                ],
            }
            yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
            payload["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            payload.update(
                sources=result["sources"], trace=result["trace"], guard=result["guard"]
            )
            yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    if result["answer"] is None:
        raise HTTPException(status_code=503, detail="未配置 LLM_API_KEY，仅完成检索")
    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["answer"]},
                "finish_reason": "stop",
            }
        ],
        "sources": result["sources"],
        "trace": result["trace"],
        "guard": result["guard"],
    }


@app.get("/", include_in_schema=False)
def demo():
    from fastapi.responses import HTMLResponse
    from pathlib import Path

    return HTMLResponse(
        Path(__file__).with_name("demo.html").read_text(encoding="utf-8")
    )
