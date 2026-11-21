"""下载固定版本模型，自动读取项目 .env；HF_ENDPOINT 可显式选择镜像。"""

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
from huggingface_hub import snapshot_download

cache = Path(os.getenv("HF_HOME", str(ROOT.parent / ".model-cache"))) / "hub"
models = [
    ("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181"),
    ("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"),
]
for repo, revision in models:
    path = snapshot_download(
        repo,
        revision=revision,
        cache_dir=str(cache),
        allow_patterns=[
            "config.json",
            "config_sentence_transformers.json",
            "sentence_bert_config.json",
            "modules.json",
            "1_Pooling/config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
            "pytorch_model.bin",
            "model.safetensors",
        ],
        max_workers=2,
    )
    print(repo, path, flush=True)
