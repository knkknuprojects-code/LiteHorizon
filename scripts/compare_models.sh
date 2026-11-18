#!/usr/bin/env bash
# Scan the same corpus slice with several models and compare the results pairwise.
#
#   bash scripts/compare_models.sh                 # 2019, 500 items, the three swept models
#   bash scripts/compare_models.sh --dry-run       # print the plan, run nothing
#   YEAR=2021 LIMIT=1000 bash scripts/compare_models.sh
#   RECORDS_ROOT=/data/wos-scopus bash scripts/compare_models.sh
#   MODELS="e4b|google/gemma-4-E4B-it|8| 12b|google/gemma-4-12B-it|16|--load-4bit" \
#     bash scripts/compare_models.sh
#
# Why a script: the throughput sweep answers which configuration is affordable, not
# which is right. Every model here validates 31-32 of 32, so yield says the output
# parses and nothing more. This is the run that compares what they actually say.
#
# One dataset root PER MODEL is a correctness requirement, not tidiness: the store's
# skip index keys on (content_hash, prompt_version) and ignores model_id, so a second
# model pointed at the first one's root would skip every item and write nothing. The
# roots are merged afterwards for the comparison, which is valid because
# list_all_records() keeps overwritten versions -- same key, different model_id is
# exactly what `horizon compare` reads.
#
# Re-running is safe and resumes: each root skips its own already-analysed items.
set -uo pipefail

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

cd "$(dirname "$0")/.."
HORIZON="./.venv/bin/horizon"
[ -x "$HORIZON" ] || HORIZON="horizon"
command -v "$HORIZON" >/dev/null 2>&1 || [ -x "$HORIZON" ] || {
  echo "error: no horizon executable (looked for ./.venv/bin/horizon and \$PATH)" >&2; exit 1; }

SCRATCH_WAS_SET=$([ -n "${HORIZON_SCRATCH:-}" ] && echo 1 || echo 0)
: "${HORIZON_SCRATCH:=${SCRATCH:-$PWD}/horizon}"
# The DATASET ROOT. `gather wos --records` appends records/ itself, unlike
# bench_models.py --records which wants the shard directory.
: "${RECORDS_ROOT:=$HORIZON_SCRATCH/wos-scopus}"
: "${CMP_DIR:=$HORIZON_SCRATCH/cmp}"
: "${YEAR:=2019}"
: "${LIMIT:=500}"
# tag|model_id|batch_size|extra_flags. The batch sizes are each model's measured optimum
# pulled back for memory margin, and pulled back again for the classification rubrics.
# With all four enum fields specified (phase2.v4) the prompt carries ~600 input tokens
# per sequence beyond the bare instructions, and every input token occupies KV cache: 12B
# at its measured optimum of 16 goes from 16.5 GB to roughly 20.7 and out of memory, while
# batch 8 lands at 14.5. E4B at 8 lands at 18.7. See the report's "choice, restated"
# section, and §7.12 on what happens if these are set too high anyway.
# gemma-4-E2B-it is deliberately absent: its output format collapses under 4-bit and it
# does not respond to a written rubric (23% of labels move, the distribution does not).
# Add it back with MODELS=... if that ever needs re-testing.
: "${MODELS:=e4b|google/gemma-4-E4B-it|8| 12b|google/gemma-4-12B-it|8|--load-4bit}"

LOGS="$CMP_DIR/logs"
MERGED="$CMP_DIR/merged"

say() { printf '\n== %s\n' "$*"; }
run() { printf '   $ %s\n' "$*"; [ "$DRY" = 1 ] || eval "$@"; }

[ -d "$RECORDS_ROOT/records" ] || [ "$DRY" = 1 ] || {
  echo "error: no shards under $RECORDS_ROOT/records -- run 'horizon collection-build' first," >&2
  echo "       and pass the dataset root, not the records/ dir inside it" >&2
  [ "$SCRATCH_WAS_SET" = 1 ] || echo "       (HORIZON_SCRATCH was unset, so this fell back to \$PWD/horizon)" >&2
  exit 1; }

say "Plan"
printf '   year=%s limit=%s\n   records=%s\n   out=%s\n' "$YEAR" "$LIMIT" "$RECORDS_ROOT/records" "$CMP_DIR"
for spec in $MODELS; do
  IFS='|' read -r tag model batch extra <<<"$spec"
  printf '   %-5s %-28s batch %-3s %s\n' "$tag" "$model" "$batch" "${extra:-bf16}"
done

run "mkdir -p '$LOGS' '$MERGED/data'"

# --- scan ---------------------------------------------------------------------------
failed=""
for spec in $MODELS; do
  IFS='|' read -r tag model batch extra <<<"$spec"
  say "Scan: $tag ($model, batch $batch ${extra:-bf16})"
  t0=$(date +%s)
  run "'$HORIZON' gather wos \
    --records '$RECORDS_ROOT' --years '$YEAR' --limit '$LIMIT' \
    --data-dir '$CMP_DIR/$tag' --scan --no-raw --no-corpus \
    --model '$model' --batch-size '$batch' ${extra:-} \
    2>&1 | tee '$LOGS/scan-$tag.log'"
  rc=$?
  if [ "$DRY" = 0 ]; then
    t1=$(date +%s)
    if [ "$rc" -ne 0 ]; then
      printf '   FAILED (exit %s) after %ss -- continuing with the rest\n' "$rc" "$((t1-t0))"
      failed="${failed:+$failed }$tag"
    else
      printf '   done in %ss\n' "$((t1-t0))"
    fi
  fi
done

# --- merge --------------------------------------------------------------------------
# Order does not matter: compare pairs on (content_hash, prompt_version) and splits the
# sides by model_id substring.
say "Merge"
parts=""
for spec in $MODELS; do
  IFS='|' read -r tag _ _ _ <<<"$spec"
  f="$CMP_DIR/$tag/data/analyses.jsonl"
  if [ "$DRY" = 1 ] || [ -f "$f" ]; then parts="${parts:+$parts }'$f'"
  else printf '   missing, skipping: %s\n' "$f"; fi
done
[ -n "$parts" ] || { echo "error: nothing to merge" >&2; exit 1; }
run "cat $parts > '$MERGED/data/analyses.jsonl'"
[ "$DRY" = 1 ] || printf '   %s records merged\n' "$(wc -l < "$MERGED/data/analyses.jsonl")"

# --- compare every pair ---------------------------------------------------------------
say "Compare"
tags=()
for spec in $MODELS; do IFS='|' read -r tag _ _ _ <<<"$spec"; tags+=("$tag"); done
for ((i=0; i<${#tags[@]}; i++)); do
  for ((j=i+1; j<${#tags[@]}; j++)); do
    a="${tags[i]}"; b="${tags[j]}"
    # The side selectors are model_id substrings; the tags are chosen to be unique
    # within the model ids (E2B / E4B / 12B).
    A=$(echo "$a" | tr '[:lower:]' '[:upper:]'); B=$(echo "$b" | tr '[:lower:]' '[:upper:]')
    printf '\n   -- %s vs %s --\n' "$A" "$B"
    run "'$HORIZON' compare --data-dir '$MERGED' --baseline '$A' --teacher '$B' | tee '$LOGS/compare-$a-vs-$b.log'"
    run "'$HORIZON' compare --data-dir '$MERGED' --baseline '$A' --teacher '$B' --json > '$LOGS/compare-$a-vs-$b.json'"
  done
done

say "Done"
printf '   logs and json: %s\n' "$LOGS"
[ -n "$failed" ] && printf '   SCANS THAT FAILED: %s -- their side is missing from the comparisons above\n' "$failed"
[ "$DRY" = 1 ] && printf '\n(dry run -- nothing was executed)\n'
exit 0
