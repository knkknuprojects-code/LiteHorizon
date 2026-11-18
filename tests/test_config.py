import pytest

from horizon.config import (
    DEFAULT_MODEL_ID,
    GenerationConfig,
    Settings,
    parse_max_memory,
)


def test_default_settings():
    settings = Settings()
    assert settings.model_id == DEFAULT_MODEL_ID
    assert settings.device is None
    assert isinstance(settings.generation, GenerationConfig)


def test_from_env_overrides(monkeypatch):
    monkeypatch.setenv("HORIZON_MODEL_ID", "google/gemma-2-9b-it")
    monkeypatch.setenv("HORIZON_DEVICE", "cuda")
    settings = Settings.from_env()
    assert settings.model_id == "google/gemma-2-9b-it"
    assert settings.device == "cuda"


def test_from_env_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("HORIZON_MODEL_ID", raising=False)
    monkeypatch.delenv("HORIZON_DEVICE", raising=False)
    settings = Settings.from_env()
    assert settings.model_id == DEFAULT_MODEL_ID
    assert settings.device is None


# --- GPU loading / throughput knobs -------------------------------------------


def test_gpu_knobs_default_to_todays_behaviour():
    """Unset knobs must mean "the CPU dev path, unchanged"."""
    settings = Settings()
    assert settings.dtype is None
    assert settings.load_in_4bit is False
    assert settings.device_map is None
    assert settings.max_memory is None
    assert settings.attn_implementation is None
    assert settings.batch_size == 1
    assert settings.max_input_tokens is None


def test_from_env_reads_gpu_knobs(monkeypatch):
    monkeypatch.setenv("HORIZON_DTYPE", "float16")
    monkeypatch.setenv("HORIZON_LOAD_4BIT", "1")
    monkeypatch.setenv("HORIZON_DEVICE_MAP", "auto")
    monkeypatch.setenv("HORIZON_MAX_MEMORY", "0=38GiB,cpu=64GiB")
    monkeypatch.setenv("HORIZON_ATTN", "sdpa")
    monkeypatch.setenv("HORIZON_BATCH_SIZE", "16")
    monkeypatch.setenv("HORIZON_MAX_INPUT_TOKENS", "2048")
    settings = Settings.from_env()
    assert settings.dtype == "float16"
    assert settings.load_in_4bit is True
    assert settings.device_map == "auto"
    assert settings.max_memory == "0=38GiB,cpu=64GiB"
    assert settings.attn_implementation == "sdpa"
    assert settings.batch_size == 16
    assert settings.max_input_tokens == 2048


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "On"])
def test_env_flag_truthy(monkeypatch, value):
    monkeypatch.setenv("HORIZON_LOAD_4BIT", value)
    assert Settings.from_env().load_in_4bit is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "maybe"])
def test_env_flag_falsy(monkeypatch, value):
    monkeypatch.setenv("HORIZON_LOAD_4BIT", value)
    assert Settings.from_env().load_in_4bit is False


def test_parse_max_memory_coerces_gpu_indices_to_int():
    """accelerate ignores a string GPU key, so the int coercion is the point."""
    assert parse_max_memory("0=38GiB,1=38GiB,cpu=64GiB") == {
        0: "38GiB",
        1: "38GiB",
        "cpu": "64GiB",
    }


def test_parse_max_memory_tolerates_spacing_and_empties():
    assert parse_max_memory(" 0 = 38GiB , , cpu=64GiB ") == {
        0: "38GiB",
        "cpu": "64GiB",
    }
    assert parse_max_memory("") == {}


@pytest.mark.parametrize("spec", ["38GiB", "0=", "cpu"])
def test_parse_max_memory_rejects_malformed(spec):
    with pytest.raises(ValueError):
        parse_max_memory(spec)
