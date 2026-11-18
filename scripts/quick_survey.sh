#!/usr/bin/env bash
# Zero-dependency box survey, small enough to paste into an SSH session.
#
#   bash quick_survey.sh              # -> ./horizon-survey.txt
#   bash quick_survey.sh -o out.txt
#
# The companion to scripts/survey_box.sh, not a replacement: that one is thorough
# (Kubernetes objects, Kubeflow CRDs, SLURM, container runtime, per-variant VRAM
# verdict) but assumes the repo already reached the box. This one covers the moment
# before that is true — you are sitting in a fresh SSH session with nothing copied
# yet, and the answers here decide whether copying is even worth it.
#
# Read-only: installs nothing, downloads nothing, needs no root. Every probe degrades
# to a dash rather than aborting, because learning which halves are missing is the job.
set -uo pipefail

OUT="horizon-survey.txt"
[ "${1:-}" = "-o" ] && [ -n "${2:-}" ] && OUT="$2"

{
echo "Horizon quick survey  $(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "===== host"
hostname; uname -srm; grep PRETTY /etc/os-release 2>/dev/null
echo "cpus: $(nproc 2>/dev/null)"
awk '/MemTotal/{printf "ram: %.0f GiB\n",$2/1048576}' /proc/meminfo 2>/dev/null
# Every figure below would be the pod's, not the node's — worth knowing before reading them.
grep -qa 'docker\|kubepods\|containerd' /proc/1/cgroup 2>/dev/null && echo "NOTE: inside a container"

# The driver's CUDA version decides the torch wheel; VRAM decides which Gemma 4
# variants can load at all. This is the block everything else hangs off.
echo "===== gpu"
nvidia-smi 2>&1 | head -12
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv 2>&1
echo "-- who is already on the card --"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1

echo "===== python"
for c in python3.12 python3; do command -v "$c" >/dev/null && echo "$c -> $($c -V 2>&1)"; done
python3 -c "import os,sysconfig;print('PEP668 externally-managed:',os.path.exists(sysconfig.get_paths()['stdlib']+'/EXTERNALLY-MANAGED'))" 2>&1
# `import venv` and even `import ensurepip` both succeed on Ubuntu boxes where
# `python3 -m venv` fails — Debian strips ensurepip's bundled wheels into a separate
# package. Only actually creating one tells the truth, so do that in a temp dir.
_vt=$(mktemp -d 2>/dev/null) && { python3 -m venv "$_vt/v" >/dev/null 2>&1 \
  && echo "venv creation: ok" \
  || echo "venv creation: FAILS — needs 'apt install python3.12-venv' (bootstrap_box.sh works around it without root)"; rm -rf "$_vt"; }

# torch's triton JIT-compiles a CPython extension at the first kernel launch, so a
# box without these generates zero tokens while looking perfectly installed.
echo "===== build toolchain (triton needs it at runtime)"
command -v cc >/dev/null && echo "cc: $(cc --version 2>/dev/null | head -1)" || echo "cc: MISSING — apt install build-essential"
python3 -c "import os,sysconfig;p=os.path.join(sysconfig.get_paths()['include'],'Python.h');print('Python.h:', p if os.path.exists(p) else 'MISSING — apt install python3-dev')" 2>&1

echo "===== ml stack already present?"
for m in torch transformers numpy accelerate bitsandbytes sentence_transformers; do
  python3 -c "import $m;print('$m', $m.__version__)" 2>/dev/null || echo "$m --"
done
python3 -c "import torch;print('cuda available:',torch.cuda.is_available(),'| torch cuda:',torch.version.cuda)" 2>/dev/null

# A bf16 checkpoint is 50-75 GB and $HOME is often quota-capped, so where HF_HOME
# lands is a decision, not a detail.
echo "===== disk"
df -hT 2>/dev/null | grep -vE 'tmpfs|overlay' || df -h
echo "-- inodes --"; df -i "$HOME" 2>/dev/null | tail -1
echo "-- candidate scratch paths --"
for p in "$HOME" /scratch /data /mnt "${SCRATCH:-}" "${HORIZON_SCRATCH:-}"; do
  [ -n "$p" ] && [ -d "$p" ] && printf '%-20s %-14s writable=%s\n' \
    "$p" "$(df -h "$p" 2>/dev/null | awk 'NR==2{print $4" free"}')" "$([ -w "$p" ] && echo yes || echo NO)"
done
du -sh "${HF_HOME:-$HOME/.cache/huggingface}" 2>/dev/null

# An air-gapped box is workable but changes the whole plan, so find out now.
echo "===== network"
for u in https://pypi.org/simple/ https://huggingface.co/api/models/gpt2 https://download.pytorch.org/whl/; do
  printf '%-46s %s\n' "$u" "$(curl -sSL -m 8 -o /dev/null -w '%{http_code}' "$u" 2>&1)"
done
echo "proxy: ${HTTPS_PROXY:-${https_proxy:-none}}"

echo "===== other tooling"
for c in docker podman nvidia-ctk kubectl k3s microk8s minikube sinfo git rsync tmux screen curl; do
  command -v "$c" >/dev/null && echo "$c: $(command -v "$c")"
done
} 2>&1 | tee "$OUT"

printf '\nSaved to: %s\n' "$OUT"
