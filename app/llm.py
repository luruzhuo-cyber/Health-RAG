"""OpenAI 兼容的对话客户端：超时、重试、流式。"""

from __future__ import annotations

import time
from typing import Iterator

from .config import Settings


class ChatClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.settings.llm_base_url,
                api_key=self.settings.llm_api_key,
                timeout=self.settings.llm_timeout,
                max_retries=0,
            )
        return self._client

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        client = self._get_client()
        last_err: Exception | None = None
        for attempt in range(self.settings.llm_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=self.settings.llm_model,
                    messages=messages,
                    temperature=temperature,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # 超时、5xx、限流统一重试，4xx 立即抛出
                last_err = exc
                from openai import APIConnectionError, APITimeoutError, APIStatusError

                transient = isinstance(exc, (APIConnectionError, APITimeoutError)) or (
                    isinstance(exc, APIStatusError)
                    and (exc.status_code in (408, 429) or exc.status_code >= 500)
                )
                if not transient or attempt >= self.settings.llm_retries:
                    break
                time.sleep(min(8.0, 1.5 * (2**attempt)))
        raise RuntimeError("LLM provider unavailable") from None

    def stream(self, messages: list[dict], temperature: float = 0.0) -> Iterator[str]:
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.settings.llm_model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for event in resp:
            delta = event.choices[0].delta.content if event.choices else None
            if delta:
                yield delta
