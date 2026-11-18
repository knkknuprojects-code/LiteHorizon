#!/usr/bin/env bash
# Push the working tree to the GPU box over SSH.
#
#   bash scripts/push_to_box.sh user@host
#   bash scripts/push_to_box.sh user@host --dry-run     # show, transfer nothing
#   BOX_DEST=~/work bash scripts/push_to_box.sh user@host        # non-default destination
#
# There is no git remote yet, so this is how the repo travels. Exclusions live in
# scripts/box-exclude.txt — a file, not a row of --exclude flags, because the flag
# version breaks as soon as a paste wraps it across lines.
#
# On a password-authenticated box this cannot prompt for the password itself when it is
# called from anywhere without a terminal. Open a multiplexed master connection once and
# every later transfer rides it silently:
#
#     # ~/.ssh/config
#     Host gpubox
#         HostName <ip>
#         User <user>
#         ControlMaster auto
#         ControlPath ~/.ssh/cm-%C
#         ControlPersist 8h
#
#     ssh gpubox            # type the password once; the socket lives 8h after logout
#     bash scripts/push_to_box.sh gpubox
#
# Re-run it after every local edit: rsync sends only what changed, so the second run
# costs seconds. .git/ and out/ are excluded (see box-exclude.txt); pass
# --include '.git/***' to send history anyway.
set -euo pipefail

DEST="${1:-}"
if [ -z "$DEST" ] || [ "$DEST" = "-h" ]; then
  sed -n '2,17p' "$0"; exit "$([ -z "$DEST" ] && echo 2 || echo 0)"
fi
shift

cd "$(dirname "$0")/.."
EXCLUDES="scripts/box-exclude.txt"
TARGET="${BOX_DEST:-~/Horizon}"

[ -f "$EXCLUDES" ] || { echo "missing $EXCLUDES" >&2; exit 1; }

echo "-> $DEST:$TARGET"
echo "   excluding per $EXCLUDES (secrets, .venv, generated data; collection/ IS sent)"
echo

# -a preserves modes so the scripts stay executable on the far side; --partial lets an
# interrupted 216 MB transfer resume instead of restarting. -i itemizes every change, so
# a run that transferred nothing cannot be mistaken for a run that updated something.
rsync -avi --progress --partial \
  --exclude-from="$EXCLUDES" \
  "$@" \
  ./ "$DEST:$TARGET/"

# A stale copy on the box already cost one debugging round: the box kept running an old
# bootstrap_box.sh and reproduced an error that had been fixed locally. Print a canary
# so "did my change actually arrive?" is one command to answer, not a guess.
echo
echo "Canary — these must match what the box reports:"
for f in scripts/bootstrap_box.sh scripts/run_scan.sh; do
  [ -f "$f" ] && printf '  %s  %s\n' "$( (shasum "$f" 2>/dev/null || sha1sum "$f") | awk '{print $1}')" "$f"
done
echo
echo "On the box:"
echo "  cd ${TARGET/#\~/~} && sha1sum scripts/bootstrap_box.sh scripts/run_scan.sh"
echo "  bash scripts/bootstrap_box.sh --dry-run"
