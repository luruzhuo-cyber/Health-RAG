from app.generate.guard import finalize, ABSTAIN


def test_refusal_phrase_does_not_bypass_invalid_citation():
    assert finalize(ABSTAIN + " 另外血型为A[99]", 1)[0] == ABSTAIN


def test_missing_patient_is_safe():
    from app.config import load_settings
    from app.index import load_index
    from app.retrieval.pipeline import Retriever

    s = load_settings()
    hits, trace = Retriever(load_index(s), s).search("血型是什么", mode="bm25")
    assert not hits
    assert trace.degraded == "patient-required-or-ambiguous"


def test_sse_is_guarded(monkeypatch):
    from fastapi.testclient import TestClient
    from app import api
    from app.generate.answer import Answerer
    from app.config import load_settings

    a = Answerer(load_settings())

    class Fake:
        def chat(self, messages):
            return "不合法结论[999]"

    a.client = Fake()
    monkeypatch.setattr(api, "get_answerer", lambda: a)
    c = TestClient(api.app)
    body = {"messages": [{"role": "user", "content": "张三九的血型"}], "stream": True}
    r = c.post("/v1/chat/completions", json=body)
    assert r.status_code == 200
    assert "[999]" not in r.text
    assert ABSTAIN in r.text
    assert "[DONE]" in r.text
    body["stream"] = False
    assert (
        c.post("/v1/chat/completions", json=body).json()["choices"][0]["message"][
            "content"
        ]
        == ABSTAIN
    )


def test_invalid_top_k(monkeypatch):
    from fastapi.testclient import TestClient
    from app.api import app

    assert (
        TestClient(app)
        .post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "张三九"}], "top_k": -1},
        )
        .status_code
        == 400
    )


def test_vectors_removed_on_sparse_rebuild(tmp_path):
    from app.config import load_settings
    from app.index import load_index, save_index

    s = load_settings()
    bundle = load_index(s)
    (tmp_path / "vectors.npy").write_bytes(b"obsolete")
    bundle.vector = None
    save_index(bundle, tmp_path)
    assert not (tmp_path / "vectors.npy").exists()


def test_config_rejects_invalid_values():
    import pytest
    from app.config import Settings

    with pytest.raises(ValueError):
        Settings(top_k=0)
    with pytest.raises(ValueError):
        Settings(chunk_max_chars=10, chunk_overlap_chars=10)


def test_retry_sanitizes_nontransient(monkeypatch):
    import pytest
    from app.config import Settings
    from app.llm import ChatClient

    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(1)
            raise ValueError("secret prompt and API key")

    class Client:
        pass

    fake = Client()
    fake.chat = Client()
    fake.chat.completions = Completions()
    c = ChatClient(Settings())
    c._client = fake
    with pytest.raises(RuntimeError, match="LLM provider unavailable") as e:
        c.chat([])
    assert "secret" not in str(e.value)
    assert len(calls) == 1
