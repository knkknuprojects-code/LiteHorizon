import json

from horizon.training import merge


def test_has_tokenizer_detects_marker_files(tmp_path):
    assert not merge._has_tokenizer(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    assert merge._has_tokenizer(tmp_path)


def test_has_tokenizer_false_for_nondir(tmp_path):
    f = tmp_path / "file"
    f.write_text("x", encoding="utf-8")
    assert not merge._has_tokenizer(f)


def test_read_run_card_returns_none_when_absent(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    assert merge._read_run_card(adapter) is None


def test_read_run_card_reads_sibling(tmp_path):
    # train.py writes run_card.json beside the adapter dir (in output_dir).
    (tmp_path / "run_card.json").write_text(
        json.dumps({"base_model": "m"}), encoding="utf-8"
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    assert merge._read_run_card(adapter) == {"base_model": "m"}


def test_build_merge_card_records_provenance():
    card = merge.build_merge_card(
        "google/gemma-2-2b-it",
        "/runs/x/adapter",
        merged_at="2026-06-30T00:00:00+00:00",
        adapter_run_card={"dataset_hash": "abc"},
    )
    assert card["base_model"] == "google/gemma-2-2b-it"
    assert card["adapter_path"] == "/runs/x/adapter"
    assert card["merged_at"] == "2026-06-30T00:00:00+00:00"
    assert card["adapter_run_card"] == {"dataset_hash": "abc"}


def test_build_merge_card_omits_run_card_when_absent():
    card = merge.build_merge_card("m", "/a", merged_at="t")
    assert "adapter_run_card" not in card


def test_main_wires_args_to_merge_adapter(monkeypatch, tmp_path):
    captured = {}

    def fake_merge(base_model, adapter_path, out_dir, *, device=None):
        captured.update(
            base=base_model, adapter=adapter_path, out=out_dir, device=device
        )
        return out_dir

    monkeypatch.setattr(merge, "merge_adapter", fake_merge)
    rc = merge.main(
        ["--base", "m", "--adapter", str(tmp_path / "a"), "--out", str(tmp_path / "o")]
    )
    assert rc == 0
    assert captured["base"] == "m"
    assert str(captured["adapter"]).endswith("/a")
    assert str(captured["out"]).endswith("/o")
