from dataclasses import replace

import pytest

from horizon.analysis import (
    Analysis,
    Classification,
    Entities,
    Impact,
    Maturity,
    TimeHorizon,
    as_attention,
    content_hash,
)


def _sample() -> Analysis:
    return Analysis(
        summary="S",
        signals=("a", "b"),
        classification=Classification(
            "theme", Maturity.EMERGING, Impact.HIGH, TimeHorizon.NEAR
        ),
        entities=Entities(organizations=("O",), technologies=("T",)),
        keywords=("k1", "k2"),
        content_hash="h",
        model_id="m",
        prompt_version="phase2.v1",
        strategy="combined",
        generated_at="2026-01-01T00:00:00+00:00",
        item_date="2026-01-01T00:00:00+00:00",
    )


def test_to_dict_from_dict_roundtrip():
    a = _sample()
    restored = Analysis.from_dict(a.to_dict())
    assert restored == a


def test_to_dict_uses_enum_values():
    d = _sample().to_dict()
    assert d["classification"]["maturity"] == "emerging"
    assert d["classification"]["impact"] == "high"
    assert isinstance(d["signals"], list)


def test_content_hash_normalizes_whitespace():
    assert content_hash("hello   world") == content_hash("hello world\n")
    assert content_hash("hello world") != content_hash("hello there")


def test_from_dict_back_compat_without_item_date():
    # Pre-Phase-3 records lack item_date; it should fall back to generated_at.
    data = _sample().to_dict()
    del data["item_date"]
    restored = Analysis.from_dict(data)
    assert restored.item_date == restored.generated_at


def test_provenance_defaults_and_roundtrip():
    a = _sample()
    assert a.source == "model" and a.approved is False
    d = a.to_dict()
    assert d["source"] == "model" and d["approved"] is False
    assert Analysis.from_dict(d) == a


def test_from_dict_back_compat_without_provenance():
    # Pre-Phase-4 records lack source/approved -> unvetted model output.
    data = _sample().to_dict()
    del data["source"]
    del data["approved"]
    restored = Analysis.from_dict(data)
    assert restored.source == "model" and restored.approved is False


def test_ingestion_provenance_defaults_and_roundtrip():
    # origin/url (Phase-5, §4) default empty for directly-scanned texts and survive
    # a round-trip when set — a separate axis from the training `source`.
    a = _sample()
    assert a.origin == "" and a.url is None
    from dataclasses import replace

    gathered = replace(a, origin="feedX", url="http://x/1")
    d = gathered.to_dict()
    assert d["origin"] == "feedX" and d["url"] == "http://x/1"
    assert d["source"] == "model"  # training-provenance axis untouched
    assert Analysis.from_dict(d) == gathered


def test_from_dict_back_compat_without_ingestion_provenance():
    # Pre-Phase-5 records lack origin/url -> empty / None.
    data = _sample().to_dict()
    del data["origin"]
    del data["url"]
    restored = Analysis.from_dict(data)
    assert restored.origin == "" and restored.url is None


# --- attention signal ---------------------------------------------------------


def test_attention_defaults_to_none_and_roundtrips():
    assert _sample().attention is None
    assert _sample().to_dict()["attention"] is None
    rated = replace(_sample(), attention=0.2875)
    assert Analysis.from_dict(rated.to_dict()) == rated


def test_from_dict_back_compat_without_attention():
    data = _sample().to_dict()
    del data["attention"]  # pre-attention record
    assert Analysis.from_dict(data).attention is None


@pytest.mark.parametrize(
    "value,expected",
    [("nonsense", None), (True, None), (2.0, 1.0), (-1.0, 0.0), (0.5, 0.5)],
)
def test_from_dict_coerces_a_bad_attention(value, expected):
    data = _sample().to_dict()
    data["attention"] = value
    assert Analysis.from_dict(data).attention == expected


def test_as_attention_rejects_nan():
    assert as_attention(float("nan")) is None
