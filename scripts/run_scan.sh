#!/usr/bin/env bash
# Scan one corpus shard into its own dataset root.
#
# Works unchanged under bash, nohup, tmux, `srun`, `sbatch`, or a bare cloud VM —
# the scheduler, if any, wraps this script rather than the script knowing about it.
#
#   scripts/run_scan.sh 2019
#   scripts/run_scan.sh 2019 google/gemma-4-12B-it 16 4bit
#   HORIZON_SOURCE=patents scripts/run_scan.sh 2019
#
# HORIZON_SOURCE picks the gather adapter and, with it, the default records root and the
# per-shard dataset name. It defaults to `wos`, so every existing invocation and the
# scan-loop scripts on the boxes behave exactly as before.
#
# Resuming is just re-running the same command: the store is append-only and
# already-scanned items are skipped, so a preemption or a Ctrl-C costs only the
# in-flight batch.
set -euo pipefail

# Resolve the interpreter rather than requiring an activated venv: these run from cron,
# from a fresh tmux pane, and over `ssh box 'cmd'`, none of which source a profile.
HORIZON_BIN="$(dirname "$0")/../.venv/bin/horizon"
[ -x "$HORIZON_BIN" ] || HORIZON_BIN="horizon"
command -v "$HORIZON_BIN" >/dev/null 2>&1 || [ -x "$HORIZON_BIN" ] || {
  echo "error: no horizon executable — activate the venv or run bootstrap_box.sh" >&2; exit 1; }

YEAR="${1:?usage: run_scan.sh YEAR [MODEL] [BATCH] [none|4bit]}"
# Defaults are the configuration this hardware actually supports, not the largest
# model: gemma-4-12B-it in bf16 needs ~28 GB and would fail at load on a 24 GB card,
# before the batch-halving fallback can help (that covers generation, not loading).
# gemma-4-E4B-it in bf16 at batch 8 measured 17.7 GB with the bare prompt and lands
# near 18.7 with the phase2.v4 rubrics. For 12B pass: run_scan.sh YEAR google/gemma-4-12B-it 8 4bit
MODEL="${2:-google/gemma-4-E4B-it}"
BATCH="${3:-8}"
QUANT="${4:-none}"

SCRATCH_WAS_SET=$([ -n "${HORIZON_SCRATCH:-}" ] && echo 1 || echo 0)
: "${HORIZON_SCRATCH:=${SCRATCH:-$PWD}/horizon}"
# The DATASET ROOT, not the records/ directory inside it: `gather wos --records`
# appends records/ itself. bench_models.py --records means the opposite (the shard
# directory), which is how this was wrong here until a scan actually ran.
SOURCE="${HORIZON_SOURCE:-wos}"
case "$SOURCE" in
  wos)     DEFAULT_RECORDS="$HORIZON_SCRATCH/wos-scopus"; SHARD_PREFIX="wos" ;;
  patents) DEFAULT_RECORDS="$HORIZON_SCRATCH/patents";    SHARD_PREFIX="patents" ;;
  *) echo "error: HORIZON_SOURCE must be wos or patents (got $SOURCE)" >&2; exit 1 ;;
esac
RECORDS_ROOT="${HORIZON_RECORDS:-$DEFAULT_RECORDS}"
[ -d "$RECORDS_ROOT/records" ] || {
  echo "error: no shards under $RECORDS_ROOT/records — pass the dataset root, not the records dir" >&2
  [ "$SCRATCH_WAS_SET" = 1 ] || echo "       (HORIZON_SCRATCH was unset, so this fell back to \$PWD/horizon)" >&2
  exit 1
}
# One dataset root per shard: concurrent appends to a single analyses.jsonl are not
# safe on NFS/Lustre. Merge afterwards with `cat */data/analyses.jsonl`, which is
# valid because the store is append-only with last-write-wins per (hash, version).
DATA_DIR="$HORIZON_SCRATCH/ds/$SHARD_PREFIX-$YEAR"
LOG_DIR="$HORIZON_SCRATCH/logs"
mkdir -p "$DATA_DIR" "$LOG_DIR"

# --- environment hygiene (nothing in the package sets these) ------------------
export HF_HOME="${HF_HOME:-$HORIZON_SCRATCH/hf}"
export HF_HUB_DISABLE_TELEMETRY=1
# Tokenizers' own threads fight the dataloader and skew timings.
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# Materially reduces fragmentation OOMs in a long-lived generate loop.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$(hostname -s)-$$"
LOG="$LOG_DIR/$RUN_ID.log"

quant_flag=()
[ "$QUANT" = "4bit" ] && quant_flag=(--load-4bit)

echo "run_id=$RUN_ID source=$SOURCE year=$YEAR model=$MODEL batch=$BATCH quant=$QUANT" | tee "$LOG"
echo "records=$RECORDS_ROOT data_dir=$DATA_DIR hf_home=$HF_HOME" | tee -a "$LOG"

# --no-corpus: a full pass would write one small JSON per item — an inode and
# metadata-latency problem on a shared filesystem, and for `gather wos` it only
# duplicates the JSONL shards the records already are.
"$HORIZON_BIN" gather "$SOURCE" \
  --records "$RECORDS_ROOT" \
  --years "$YEAR" \
  --data-dir "$DATA_DIR" \
  --scan --no-raw --no-corpus \
  --model "$MODEL" \
  --batch-size "$BATCH" \
  "${quant_flag[@]}" \
  --progress-every 50 \
  2>&1 | tee -a "$LOG"

# tee would otherwise mask a non-zero exit from horizon.
exit "${PIPESTATUS[0]}"
