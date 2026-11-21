"""运行配置：全部从环境变量读取，仓库内不保存任何密钥。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=False)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer")


@dataclass
class Settings:
    # 索引
    index_dir: Path = field(
        default_factory=lambda: Path(
            _env("RAG_INDEX_DIR", str(ROOT / "data" / "index"))
        )
    )
    source_pdf: Path = field(
        default_factory=lambda: Path(
            _env("RAG_SOURCE_PDF", str(ROOT / "data" / "raw" / "健康档案.pdf"))
        )
    )

    # 切块
    chunk_max_chars: int = field(
        default_factory=lambda: _env_int("RAG_CHUNK_MAX_CHARS", 260)
    )
    chunk_overlap_chars: int = field(
        default_factory=lambda: _env_int("RAG_CHUNK_OVERLAP", 60)
    )

    # 检索
    top_k: int = field(default_factory=lambda: _env_int("RAG_TOP_K", 5))
    rrf_k: int = field(default_factory=lambda: _env_int("RAG_RRF_K", 60))
    min_score: float = field(
        default_factory=lambda: float(_env("RAG_MIN_SCORE", "0.05"))
    )

    # 模型（OpenAI 兼容协议，可指向本地网关）
    llm_base_url: str = field(
        default_factory=lambda: _env("LLM_BASE_URL", "http://localhost:3000/v1")
    )
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen-plus"))
    llm_timeout: int = field(default_factory=lambda: _env_int("LLM_TIMEOUT", 30))
    llm_retries: int = field(default_factory=lambda: _env_int("LLM_RETRIES", 2))

    embed_base_url: str = field(default_factory=lambda: _env("EMBED_BASE_URL", ""))
    embed_api_key: str = field(default_factory=lambda: _env("EMBED_API_KEY"))
    embed_model: str = field(
        default_factory=lambda: _env("EMBED_MODEL", "text-embedding-v1")
    )

    # 重排
    rerank_model: str = field(default_factory=lambda: _env("RERANK_MODEL", ""))

    embed_backend: str = field(default_factory=lambda: _env("EMBED_BACKEND", "api"))
    local_embed_model: str = field(
        default_factory=lambda: _env("LOCAL_EMBED_MODEL", "BAAI/bge-m3")
    )

    def __post_init__(self):
        if (
            self.top_k < 1
            or self.top_k > 100
            or self.llm_timeout <= 0
            or not 0 <= self.llm_retries <= 5
        ):
            raise ValueError("Invalid top_k, timeout or retries")
        if (
            self.chunk_max_chars <= 0
            or not 0 <= self.chunk_overlap_chars < self.chunk_max_chars
        ):
            raise ValueError("Invalid chunk size/overlap")
        if self.embed_backend not in ("api", "local"):
            raise ValueError("EMBED_BACKEND must be api or local")

    @property
    def embedding_identity(self):
        return f"{self.embed_backend}|{self.local_embed_model if self.embed_backend == 'local' else self.embed_model}|{self.embed_base_url if self.embed_backend == 'api' else 'local'}"

    @property
    def has_llm(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def has_embedding(self) -> bool:
        return self.embed_backend == "local" or bool(
            self.embed_api_key and self.embed_base_url
        )


def load_settings() -> Settings:
    return Settings()
