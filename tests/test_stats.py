from horizon import prompts
from horizon.analysis import Analysis
from horizon.training import stats

GOOD = {
    "summary": "A new open model was released.",
    "signals": ["state-of-the-art reasoning"],
    "classification": {
        "theme": "open models",
        "maturity": "emerging",
        "impact": "high",
        "time_horizon": "near",
    },
    "entities": {
        "organizations": [],
        "people": [],
        "technologies": [],
        "locations": [],
    },
    "keywords": ["open-weights"],
}


def _analysis(content_hash="h", **over):
    data = {
        **GOOD,
        **over,
        "content_hash": content_hash,
        "model_id": "fake",
        "prompt_version": prompts.PROMPT_VERSION,
        "strategy": "combined",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "item_date": "2026-01-01T00:00:00+00:00",
    }
    return Analysis.from_dict(data)


def test_script_bucket():
    assert stats._script_bucket("hello world") == "latin"
    assert stats._script_bucket("привіт світ") == "cyrillic"
    assert stats._script_bucket("こんにちは世界") == "cjk"
    assert stats._script_bucket("123 !!!") == "other"


def test_dataset_stats_counts_and_gap():
    analyses = [
        _analysis("a", approved=True),  # ready + approved
        _analysis("b"),  # ready, unapproved
        _analysis("c"),  # trainable but no source
        _analysis("d", summary=""),  # not trainable
    ]
    sources = {"a": "text a", "b": "текст б"}
    s = stats.dataset_stats(analyses, sources, target=10)

    assert s["total"] == 4
    assert s["trainable"] == 3  # a, b, c
    assert s["ready"] == 2  # a, b (c has no source)
    assert s["trainable_no_source"] == 1
    assert s["approved"] == 1
    assert s["gap"] == 8  # target 10 - 2 ready
    # distributions are over the ready set
    assert s["impact"]["high"] == 2
    assert s["scripts"] == {"latin": 1, "cyrillic": 1}


def test_dataset_stats_ready_meets_target_no_gap():
    analyses = [_analysis(f"{i:x}") for i in range(3)]
    sources = {a.content_hash: "text" for a in analyses}
    s = stats.dataset_stats(analyses, sources, target=2)
    assert s["gap"] == 0


def test_format_stats_renders_verdict():
    analyses = [_analysis("a")]
    s = stats.dataset_stats(analyses, {"a": "text a"}, target=5)
    report = stats.format_stats(s)
    assert "Dataset readiness" in report
    assert "need 4 more" in report
