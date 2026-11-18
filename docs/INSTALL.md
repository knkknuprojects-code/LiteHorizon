# Installation and environment

The CLI package is named `horizon`; the research repository is called LiteHorizon.

## Baseline prerequisites

- Python **3.12+**, a virtual environment, and adequate disk space.
- Network access and model-specific RAM for **actual local LLM inference**; tests
  do not require downloading model weights.
- Additional authorization is required for gated Gemma weights and paid/hosted APIs.

## Linux / macOS (CPU path)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q tests/test_analysis.py tests/test_extract.py
horizon --help
```

## Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q tests/test_analysis.py tests/test_extract.py
horizon --help
```

If PowerShell activation is restricted, you can call
`.\.venv\Scripts\python.exe -m pip install -e ".[dev]"` and
`.\.venv\Scripts\python.exe -m pytest -q tests/test_analysis.py` directly without
activating it. Run the installed `horizon` command using the virtual environment's
`Scripts` directory or the corresponding Python entry point.

## Optional extras

```bash
python -m pip install -e ".[ingest]"      # RSS/arXiv and Excel input
python -m pip install -e ".[url]"         # cleaner web-page text extraction
python -m pip install -e ".[train]"       # CPU LoRA / evaluation
python -m pip install -e ".[teacher]"     # Gemini SDK
```

Do not install the CUDA-only extras on unsupported machines. The production Gemma 4
GPU environment has distinct requirements: `requirements-gpu.txt` and
`scripts/bootstrap_box.sh` document that path. `pyproject.toml` currently includes
`transformers<5` and `numpy<2` constraints motivated by the historical Intel Mac;
`requirements-gpu.txt` supplies a different GPU stack. These two paths must not be
mixed without explicitly resolving their dependencies.

## Model-free confidence test

```bash
python -m pytest -q --ignore=tests/test_llm.py
```

`test_llm.py` exercises optional model-client dependencies (including a PEFT import in
some test paths). Excluding it is not the same as validating model inference or the
large-model experiment. For a full run, install the needed extras and models first.

## Actual model-based scan

```bash
horizon scan examples/sample.txt \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --max-new-tokens 256 --json
```

This first call may download model weights and can be slow or fail with insufficient
memory. It produces a model-generated, **unreviewed** analysis; no semantic accuracy
claim is implied. Use `--no-save` if you do not want it written to a local JSONL store.

For troubleshooting on the original development host, consult
[the original Ukrainian guide](DEVELOPER_GUIDE_UK.md).
