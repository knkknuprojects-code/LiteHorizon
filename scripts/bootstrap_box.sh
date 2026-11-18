#!/usr/bin/env bash
# Install Horizon's inference environment on a CUDA box reached over plain SSH.
# Run scripts/survey_box.sh first.
#
#   bash scripts/bootstrap_box.sh              # create .venv and install
#   bash scripts/bootstrap_box.sh --dry-run    # print the plan, change nothing
#
# Unlike survey_box.sh this one WRITES: it creates ./.venv and installs into it. It
# touches nothing outside the repo directory and needs no root.
#
# What it automates is the one decision the manual instructions leave to you — which
# torch wheel matches the driver. Everything else is the manual install
# verbatim, including `pip install -e . --no-deps`: pyproject.toml's Intel-Mac ceilings
# (transformers<5, numpy<2) would otherwise undo the resolution, and Gemma 4 needs
# transformers>=5.5 by construction.
set -euo pipefail

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

cd "$(dirname "$0")/.."
REPO="$PWD"
VENV="${HORIZON_VENV:-$REPO/.venv}"

say()  { printf '\n== %s\n' "$*"; }
do_()  { printf '   $ %s\n' "$*"; [ "$DRY" = 1 ] || eval "$@"; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# --- preflight ---------------------------------------------------------------------
say "Preflight"

PY=""
for c in python3.12 python3 python; do
  if command -v "$c" >/dev/null 2>&1; then
    v=$("$c" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null) || continue
    case "$v" in 3.1[2-9]|3.[2-9]*) PY="$c"; break ;; esac
  fi
done
[ -n "$PY" ] || die "no Python >=3.12 found (pyproject.toml requires it). Ask the admins."
printf '   python      %s (%s)\n' "$($PY -V 2>&1)" "$(command -v "$PY")"

command -v nvidia-smi >/dev/null 2>&1 \
  || die "nvidia-smi not found. bitsandbytes 4-bit is CUDA-only and batched CUDA inference is the point of this box."
printf '   card        %s\n' "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
printf '   driver      %s\n' "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"

# The driver's reported CUDA version, not any installed toolkit, is what the torch
# wheel's bundled runtime has to satisfy.
# torch pulls in triton, which JIT-compiles its own CUDA driver shim as a CPython
# extension the first time a kernel runs. No compiler and no Python.h means every
# generate() raises and every item is recorded as a failure — with the install itself
# looking perfectly healthy, pytest included. Catch it here, not after 3 GB of wheels.
missing_build=""
command -v cc >/dev/null 2>&1 || missing_build="build-essential"
if ! "$PY" -c 'import os,sysconfig;raise SystemExit(0 if os.path.exists(os.path.join(sysconfig.get_paths()["include"],"Python.h")) else 1)' 2>/dev/null; then
  missing_build="${missing_build:+$missing_build }python3-dev"
fi
if [ -n "$missing_build" ]; then
  printf '   toolchain   MISSING: %s\n' "$missing_build"
  printf '               triton compiles a CPython extension at first kernel launch;\n'
  printf '               without these, generation produces zero tokens and every item\n'
  printf '               fails, while the install and the test suite look fine.\n'
  printf '               Fix (needs root):  sudo apt install %s\n' "$missing_build"
else
  printf '   toolchain   cc + Python.h present\n'
fi

CUDA_STR=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' | head -1)
CUDA_MAJ=${CUDA_STR%%.*}
printf '   driver CUDA %s\n' "${CUDA_STR:-unknown}"

INDEX_ARGS=""
if [ -n "${HORIZON_TORCH_INDEX:-}" ]; then
  INDEX_ARGS="--index-url $HORIZON_TORCH_INDEX"
  printf '   torch wheel HORIZON_TORCH_INDEX override -> %s\n' "$HORIZON_TORCH_INDEX"
elif [ -z "${CUDA_MAJ:-}" ]; then
  printf '   torch wheel could not read the driver CUDA version; using the default PyPI wheel\n'
elif [ "$CUDA_MAJ" -ge 12 ] 2>/dev/null; then
  printf '   torch wheel default PyPI wheel (it already bundles CUDA 12.x)\n'
elif [ "$CUDA_MAJ" -eq 11 ] 2>/dev/null; then
  INDEX_ARGS="--index-url https://download.pytorch.org/whl/cu118"
  printf '   torch wheel driver is CUDA 11.x -> installing torch from the cu118 index\n'
else
  die "driver reports CUDA $CUDA_STR. torch>=2.6 (transformers 5's floor) has no wheel for it — the driver needs updating. This is an admin request, not something this script can work around."
fi

# --- install -----------------------------------------------------------------------
say "Virtualenv at $VENV"

# Ubuntu ships the ensurepip *module* in the stdlib but strips its bundled wheels into
# the python3.12-venv / python3-pip-whl packages. So `import ensurepip` succeeds on a
# box where `python3 -m venv` fails — probing it proves nothing, and the only honest
# test is to try. Attempt, then recover without root, since apt would mean going back
# to the admins for one package.
if [ -x "$VENV/bin/pip" ]; then
  printf '   (exists with a working pip — reusing; rm -rf it to start clean)\n'
elif [ "$DRY" = 1 ]; then
  printf '   $ %s -m venv %s\n' "$PY" "$VENV"
  printf '     if ensurepip has no wheels (Ubuntu without python3.12-venv), falls back to:\n'
  printf '     $ %s -m venv --without-pip %s\n' "$PY" "$VENV"
  printf '     $ curl -sSL https://bootstrap.pypa.io/get-pip.py | %s/bin/python\n' "$VENV"
else
  rm -rf "$VENV"
  if "$PY" -m venv "$VENV" >/dev/null 2>&1 && [ -x "$VENV/bin/pip" ]; then
    printf '   created\n'
  else
    printf '   ensurepip has no wheels here (Ubuntu splits them into python3.12-venv).\n'
    printf '   -> retrying --without-pip and bootstrapping pip over the network.\n'
    printf '      With root, `sudo apt install python3.12-venv` is the tidier fix.\n'
    rm -rf "$VENV"
    "$PY" -m venv --without-pip "$VENV" || die "even --without-pip failed; the Python install is broken"
    curl -sSL https://bootstrap.pypa.io/get-pip.py -o "$VENV/get-pip.py" \
      || die "cannot reach bootstrap.pypa.io. With no egress this needs root: apt install python3.12-venv"
    "$VENV/bin/python" "$VENV/get-pip.py" >/dev/null \
      || die "get-pip.py failed"
    printf '   created (pip bootstrapped from bootstrap.pypa.io)\n'
  fi
fi
PIP="$VENV/bin/pip"

# With more than one box analysing one corpus, the two must not differ in library
# versions -- results keyed the same but produced by different transformers builds are
# not one dataset. So prefer the committed lockfile when it exists. Its `-e <path>` line
# is machine-specific and belongs to the separate --no-deps step below, so it is
# filtered out rather than trusted.
LOCK="requirements-gpu.lock.txt"
REQS="requirements-gpu.txt"
if [ "${USE_LOCK:-1}" != "0" ] && [ -f "$LOCK" ]; then
  REQS_SRC="$LOCK (pinned; USE_LOCK=0 to resolve fresh from $REQS instead)"
  REQS_FILE="$VENV/requirements.resolved.txt"
  [ "$DRY" = 1 ] || grep -vE '^\s*(-e |#)' "$LOCK" > "$REQS_FILE"
else
  REQS_SRC="$REQS (floors only -- run pip freeze afterwards and commit the lockfile)"
  REQS_FILE="$REQS"
fi

say "Dependencies"
printf '   from        %s\n' "$REQS_SRC"
do_ "'$PIP' install --upgrade pip"
if [ -n "$INDEX_ARGS" ]; then
  # torch first from the matching index, so the requirements pass cannot pull the
  # default-CUDA wheel over it.
  do_ "'$PIP' install $INDEX_ARGS 'torch>=2.6'"
fi
do_ "'$PIP' install -r '$REQS_FILE'"

say "The package itself, without its dependencies"
do_ "'$PIP' install -e . --no-deps"

# --- verify ------------------------------------------------------------------------
say "Verify (loads no model)"
if [ "$DRY" = 1 ]; then
  do_ "'$VENV/bin/python' -m pytest -q"
elif "$VENV/bin/python" -c 'import pytest' >/dev/null 2>&1; then
  # Not `do_`: a red suite is a finding to report, not a reason to skip the version
  # verification and the lockfile below, which are what the rest of the setup needs.
  "$VENV/bin/python" -m pytest -q || printf '\n   TESTS FAILED — read the output above before trusting this install\n'
else
  printf '   pytest not installed — skipping (add it: %s install pytest)\n' "$PIP"
fi

if [ "$DRY" = 0 ]; then
  "$VENV/bin/python" - <<'PYEOF'
import torch, transformers, numpy
print(f"   torch          {torch.__version__}")
print(f"   transformers   {transformers.__version__}")
print(f"   numpy          {numpy.__version__}")
print(f"   cuda available {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   device         {torch.cuda.get_device_name(0)}")
    print(f"   bf16 supported {torch.cuda.is_bf16_supported()}")
# Gemma 4 is model_type "gemma4"; nothing before transformers 5.5 can load it, and
# finding that out here is much cheaper than finding it out 50 GB into a download.
from packaging.version import Version
ok = Version(transformers.__version__) >= Version("5.5")
print(f"   gemma4-capable {ok}" + ("" if ok else "  <-- transformers<5.5 cannot load Gemma 4"))
try:
    import bitsandbytes  # noqa: F401
    print("   bitsandbytes   importable (4-bit available)")
except Exception as e:
    print(f"   bitsandbytes   NOT importable ({type(e).__name__}) — --load-4bit will fail")
PYEOF
fi

say "Lockfile"
do_ "'$PIP' freeze | grep -vE '^\\s*-e ' > requirements-gpu.lock.txt"
printf '   requirements-gpu.lock.txt written — commit it, it is the real resolution\n'

# --- next --------------------------------------------------------------------------
say "Next"
cat <<'NEXT'
   . .venv/bin/activate
   export HORIZON_SCRATCH=/path/with/200GB      # HF cache alone is 50-75 GB per model
   export HF_HOME="$HORIZON_SCRATCH/hf"

   horizon collection-build --data-dir "$HORIZON_SCRATCH/wos-scopus"   # ~75 s
   shasum "$HORIZON_SCRATCH/wos-scopus/records/2019.jsonl"             # compare to local

   scripts/sweep.sh                              # pick the model/batch/quant cell
   tmux new -s scan                              # survives the SSH session dropping
   for y in $(seq 2015 2026); do scripts/run_scan.sh "$y" MODEL BATCH QUANT; done
NEXT
[ "$DRY" = 1 ] && printf '\n(dry run — nothing was installed)\n'
exit 0
