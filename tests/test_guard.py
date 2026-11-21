from app.generate.guard import ABSTAIN, cited_indices, finalize, validate


def test_extracts_citation_indices():
    assert cited_indices("血型为 O 型 [1]，过敏史见 [2][3]。") == [1, 2, 3]


def test_missing_citation_is_rejected():
    ok, reason = validate("张三九的血型是 O 型。", n_hits=3)
    assert not ok and reason == "no-citation"


def test_out_of_range_citation_is_rejected():
    ok, reason = validate("见 [5]。", n_hits=3)
    assert not ok and reason == "citation-out-of-range"


def test_valid_answer_passes():
    ok, reason = validate("血型为 O 型 [1]。", n_hits=3)
    assert ok and reason == "ok"


def test_strict_mode_falls_back_to_abstain():
    text, reason = finalize("张三九的血型是 O 型。", n_hits=3, strict=True)
    assert text == ABSTAIN and reason == "no-citation"


def test_abstain_is_preserved():
    text, reason = finalize(ABSTAIN, n_hits=0)
    assert text == ABSTAIN and reason == "abstain"
