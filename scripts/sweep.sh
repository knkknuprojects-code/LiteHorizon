#!/usr/bin/env bash
# Probe the card, then sweep the Gemma 4 variants that fit it.
#
#   scripts/sweep.sh                 # fitting variants, batches 1 4 8 16, bf16 + 4-bit
#   scripts/sweep.sh 32              # same, but 32 sampled items per cell
#
# Pick the winner by `s_per_ok_item` (seconds per *validated* analysis), not by raw
# speed: a faster model that needs repair retries costs more per usable analysis.
set -euo pipefail

# Resolve the interpreter rather than requiring an activated venv: these run from cron,
# from a fresh tmux pane, and over `ssh box 'cmd'`, none of which source a profile.
PY_BIN="$(dirname "$0")/../.venv/bin/python"
[ -x "$PY_BIN" ] || PY_BIN="python3"

ITEMS="${1:-8}"
: "${HORIZON_SCRATCH:=${SCRATCH:-$PWD}/horizon}"
RECORDS="${HORIZON_RECORDS:-$HORIZON_SCRATCH/wos-scopus/records}"
BENCH_DIR="$HORIZON_SCRATCH/bench"
mkdir -p "$BENCH_DIR"

export HF_HOME="${HF_HOME:-$HORIZON_SCRATCH/hf}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT="$BENCH_DIR/$(date -u +%Y-%m-%d)-$(hostname -s).jsonl"

echo "=== probe ==="
"$PY_BIN" scripts/bench_models.py --probe

echo
echo "=== sweep -> $OUT ==="
"$PY_BIN" scripts/bench_models.py \
  --device cuda \
  --preset fit \
  --records "$RECORDS" \
  --items "$ITEMS" \
  --batch 1 4 8 16 \
  --quant none 4bit \
  --out "$OUT"

echo
echo "Cells, cheapest validated analysis first:"
"$PY_BIN" - "$OUT" <<'PY'
import json, sys
# Per-model summaries are written as `RESULT {json}`; per-item rows are bare json.
rows = [
    json.loads(line[len("RESULT ") :])
    for line in open(sys.argv[1], encoding="utf-8")
    if line.startswith("RESULT ")
]
cells = [
    (c.get("s_per_ok_item"), r["model"], r["quant"], c["batch"], c["n_ok"], c["n"],
     c.get("gpu_peak_gb"))
    for r in rows for c in r.get("cells", [])
]
for s, model, quant, batch, ok, n, vram in sorted(cells, key=lambda c: (c[0] is None, c[0])):
    print(f"  {s!s:>8} s/ok  {model:<28} {quant:<5} batch={batch:<3} "
          f"yield={ok}/{n}  peak={vram} GB")
PY
