from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health_reports_index():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["chunks"] > 0
    assert "张三九" in body["patients"]


def test_chat_requires_llm_key():
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "张三九的血型是什么"}]},
    )
    # 未配置 LLM_API_KEY 时应明确返回 503，而不是编一个答案
    assert resp.status_code == 503


def test_chat_rejects_empty_messages():
    resp = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "system", "content": "hi"}]}
    )
    assert resp.status_code == 400
