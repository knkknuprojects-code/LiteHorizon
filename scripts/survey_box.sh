#!/usr/bin/env bash
# Survey the box before installing anything.
#
#   bash scripts/survey_box.sh                 # print to stdout, also save a copy
#   bash scripts/survey_box.sh -o report.txt   # explicit destination
#
# Read-only by design: it inspects, downloads nothing, installs nothing, and needs no
# root. Every probe degrades to "not found" rather than aborting, because the whole
# point is to learn which halves of the toolchain exist. The saved copy is what gets
# pasted back to whoever is answering the open questions (driver version, StorageClass,
# GPU quota) — this is the method section of whatever report the run produces.
#
# Deliberately absent: no `pip install`, no `docker pull`, no `kubectl apply`. A survey
# that mutates the box cannot be run by someone who is not yet sure they are allowed to.

set -uo pipefail   # no -e: a missing tool is data, not a failure

OUT=""
while getopts ":o:h" opt; do
  case "$opt" in
    o) OUT="$OPTARG" ;;
    h) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "usage: $0 [-o report.txt]" >&2; exit 2 ;;
  esac
done
[ -n "$OUT" ] || OUT="horizon-survey-$(date -u +%Y%m%dT%H%M%SZ)-$(hostname -s 2>/dev/null || echo host).txt"

have() { command -v "$1" >/dev/null 2>&1; }
# Some probes (quota on a dead NFS mount, nvidia-smi on a wedged driver) hang forever.
run() { if have timeout; then timeout 15 "$@" 2>&1; else "$@" 2>&1; fi; }
hdr() { printf '\n===== %s =====\n' "$*"; }
kv()  { printf '%-28s %s\n' "$1" "$2"; }
miss() { printf '  (not found: %s)\n' "$1"; }

survey() {

printf 'Horizon box survey  %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# --- host --------------------------------------------------------------------------
hdr "Host"
kv "hostname" "$(hostname 2>/dev/null)"
kv "kernel"   "$(uname -srm 2>/dev/null)"
if [ -r /etc/os-release ]; then
  # shellcheck disable=SC1091
  kv "os" "$(. /etc/os-release; echo "$PRETTY_NAME")"
else
  kv "os" "$(uname -s)"
fi
kv "cpus" "$( (nproc 2>/dev/null) || (sysctl -n hw.ncpu 2>/dev/null) || echo '?')"
if [ -r /proc/meminfo ]; then
  kv "ram" "$(awk '/MemTotal/{printf "%.0f GiB", $2/1048576}' /proc/meminfo)"
fi
kv "uptime" "$(uptime 2>/dev/null | sed 's/^ *//')"
# A container hides the real node from every probe below; say so up front.
if [ -f /.dockerenv ] || grep -qa 'docker\|kubepods\|containerd' /proc/1/cgroup 2>/dev/null; then
  kv "NOTE" "this appears to be INSIDE a container — figures may be the pod's, not the node's"
fi

# --- GPU ---------------------------------------------------------------------------
# The single most important block: driver version decides whether stock PyPI torch
# (bundled CUDA 12.x) works or a cu121 index is needed, and VRAM decides which Gemma 4
# variants are even loadable.
hdr "GPU / driver"
if have nvidia-smi; then
  run nvidia-smi
  echo
  run nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv
  echo
  echo "-- persistence / ECC / MIG --"
  run nvidia-smi --query-gpu=persistence_mode,ecc.mode.current,mig.mode.current --format=csv
  echo
  echo "-- who is already using the card --"
  run nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
else
  miss "nvidia-smi"
  echo "  Without it nothing here can run: bitsandbytes 4-bit is CUDA-only and the"
  echo "  whole point of this box is batched CUDA inference."
fi
have nvcc && { echo; echo "-- nvcc (toolkit, not required: torch wheels bundle CUDA) --"; run nvcc --version; }

# --- python ------------------------------------------------------------------------
hdr "Python"
for c in python3 python3.12 python3.13 python; do
  have "$c" && kv "$c" "$("$c" -V 2>&1)"
done
have python3 && {
  # Neither `import venv` nor `import ensurepip` predicts this: Debian keeps the
  # modules and strips ensurepip's bundled wheels. Creating one is the only real test.
  _vt=$(mktemp -d 2>/dev/null)
  if [ -n "${_vt:-}" ] && python3 -m venv "$_vt/v" >/dev/null 2>&1; then
    kv "venv creation" "ok"
  else
    kv "venv creation" "FAILS — apt install python3.12-venv (bootstrap_box.sh works around it)"
  fi
  rm -rf "${_vt:-/nonexistent}"
  # PEP 668: Ubuntu 24.04 marks the system Python externally-managed, so a venv is
  # mandatory rather than good manners.
  py_ext=$(python3 -c 'import sysconfig,os;p=sysconfig.get_paths()["stdlib"];print("yes" if os.path.exists(os.path.join(p,"EXTERNALLY-MANAGED")) else "no")' 2>/dev/null)
  kv "externally-managed (PEP 668)" "${py_ext:-?}"
}
have pip3 && kv "pip3" "$(run pip3 --version | head -1)"
echo
# triton (pulled in by torch) JIT-compiles a CPython extension at the first kernel
# launch. Missing either of these means generation raises on every item while the
# install, and even pytest, look healthy.
echo "-- build toolchain (triton compiles at runtime; both are required) --"
kv "cc" "$(command -v cc >/dev/null && cc --version 2>/dev/null | head -1 || echo 'MISSING — apt install build-essential')"
kv "Python.h" "$(python3 -c 'import os,sysconfig;p=os.path.join(sysconfig.get_paths()["include"],"Python.h");print(p if os.path.exists(p) else "MISSING — apt install python3-dev")' 2>&1)"
echo
echo "-- ML stack already present? (informational; we install our own in a venv) --"
have python3 && run python3 - <<'PYEOF'
for m in ("torch", "transformers", "numpy", "accelerate", "bitsandbytes", "sentence_transformers"):
    try:
        mod = __import__(m)
        print(f"  {m:24s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m:24s} -- ({type(e).__name__})")
try:
    import torch
    print(f"  torch.cuda.is_available   {torch.cuda.is_available()}")
    print(f"  torch.version.cuda        {torch.version.cuda}")
except Exception:
    pass
PYEOF

# --- container runtime -------------------------------------------------------------
hdr "Container runtime"
for c in docker podman nerdctl ctr crictl; do
  have "$c" && kv "$c" "$(run "$c" --version | head -1)"
done
have docker && {
  echo "-- can this user talk to the daemon? --"
  run docker info --format '{{.ServerVersion}} runtimes={{.Runtimes}}' | head -3
}
# Without the NVIDIA container toolkit a container cannot see the card at all.
kv "nvidia-container-toolkit" "$(have nvidia-ctk && run nvidia-ctk --version | head -1 || echo 'not found')"

# --- kubernetes --------------------------------------------------------------------
hdr "Kubernetes"
if have kubectl; then
  kv "kubectl" "$(run kubectl version --client -o yaml | awk '/gitVersion/{print $2; exit}')"
  echo "-- reachable cluster? --"
  if run kubectl cluster-info >/dev/null 2>&1; then
    run kubectl cluster-info | head -5
    kv "current context" "$(run kubectl config current-context)"
    kv "namespace" "$(run kubectl config view --minify -o jsonpath='{..namespace}')"
    echo
    echo "-- nodes (allocatable GPU is what actually matters) --"
    run kubectl get nodes -o custom-columns='NAME:.metadata.name,STATUS:.status.conditions[-1].type,CPU:.status.allocatable.cpu,MEM:.status.allocatable.memory,GPU:.status.allocatable.nvidia\.com/gpu,KUBELET:.status.nodeInfo.kubeletVersion'
    echo
    echo "-- is the NVIDIA device plugin running? (no plugin => nvidia.com/gpu is never schedulable) --"
    run kubectl get pods -A -o wide | grep -i -E 'nvidia|gpu-operator|device-plugin' | head -10
    echo
    echo "-- StorageClasses (we need one that supports ReadWriteMany) --"
    run kubectl get storageclass -o custom-columns='NAME:.metadata.name,PROVISIONER:.provisioner,DEFAULT:.metadata.annotations.storageclass\.kubernetes\.io/is-default-class,BINDING:.volumeBindingMode'
    echo "  NOTE: RWX support is a provisioner property and is NOT shown above."
    echo "  NFS/CephFS/Longhorn provisioners usually do; local-path / most block CSI do not."
    echo
    echo "-- quotas and limits in this namespace --"
    run kubectl get resourcequota,limitrange 2>&1 | head -10
    echo
    echo "-- existing horizon objects --"
    run kubectl get pvc,job,pod -l app=horizon 2>&1 | head -20
  else
    echo "  kubectl is installed but no cluster is reachable from here."
  fi
  echo
  echo "-- single-node distributions --"
  for c in k3s minikube microk8s kind k0s; do have "$c" && kv "$c" "present"; done
else
  miss "kubectl"
fi

# --- kubeflow ----------------------------------------------------------------------
hdr "Kubeflow"
if have kubectl && run kubectl cluster-info >/dev/null 2>&1; then
  echo "-- CRDs --"
  crds=$(run kubectl get crd -o name 2>/dev/null | grep -i -E 'kubeflow|notebook|pytorchjob|pipeline|katib|inferenceservice')
  if [ -n "$crds" ]; then echo "$crds"; else echo "  none — this looks like plain Kubernetes, not Kubeflow"; fi
  echo
  echo "-- namespaces --"
  run kubectl get ns 2>/dev/null | grep -i -E 'kubeflow|NAME' | head -10
else
  echo "  (skipped: no reachable cluster)"
fi

# --- slurm -------------------------------------------------------------------------
hdr "SLURM"
if have sinfo; then run sinfo; run scontrol show config 2>/dev/null | grep -i -E 'MaxJobCount|SchedulerType' | head -5
else miss "sinfo (fine — a single box does not need a scheduler)"; fi

# --- storage -----------------------------------------------------------------------
# A bf16 gemma-4 checkpoint is 50-75 GB and $HOME is often quota-capped, so where the
# HF cache lands is a real decision, not a detail.
hdr "Storage"
echo "-- filesystems --"
# -T (fstype) is GNU-only; BSD/macOS df rejects it.
if df -hT >/dev/null 2>&1; then run df -hT; else run df -h; fi
echo
echo "-- candidate scratch paths --"
for p in "${SCRATCH:-}" "${HORIZON_SCRATCH:-}" "$HOME" /scratch /data /mnt /var/tmp; do
  [ -n "$p" ] && [ -d "$p" ] && printf '%-24s %s  writable=%s\n' "$p" "$(df -h "$p" 2>/dev/null | awk 'NR==2{print $4" free of "$2}')" "$( [ -w "$p" ] && echo yes || echo NO)"
done
echo
echo "-- inodes (269k items would create 269k files without --no-corpus) --"
run df -i "$HOME" 2>/dev/null
have quota && { echo; echo "-- quota --"; run quota -s; }
echo
kv "HF_HOME" "${HF_HOME:-<unset — defaults to ~/.cache/huggingface>}"
for d in "${HF_HOME:-$HOME/.cache/huggingface}"; do
  [ -d "$d" ] && kv "existing HF cache" "$(run du -sh "$d" | awk '{print $1}')"
done

# --- network -----------------------------------------------------------------------
# An air-gapped box is workable but changes the whole plan (internal mirror + weights
# staged by hand), so it must be known before anything is ordered.
hdr "Network egress"
if have curl; then
  for u in https://pypi.org/simple/ https://files.pythonhosted.org/ https://huggingface.co/ https://huggingface.co/api/models/gpt2 https://download.pytorch.org/whl/; do
    code=$(curl -sSL -m 8 -o /dev/null -w '%{http_code}' "$u" 2>/dev/null)
    printf '%-46s %s\n' "$u" "${code:-FAILED}"
  done
  kv "proxy env" "${HTTPS_PROXY:-${https_proxy:-none}}"
else
  miss "curl"
fi

# --- verdict -----------------------------------------------------------------------
hdr "Verdict"
vram_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
if [ -n "${vram_mib:-}" ] && [ "$vram_mib" -gt 0 ] 2>/dev/null; then
  vram_gb=$(( vram_mib / 1024 ))
  kv "VRAM (largest card)" "${vram_gb} GiB"
  echo
  echo "  Which Gemma 4 variants fit, by weights alone."
  echo "  These EXCLUDE the KV cache, which at batch 32 is 32x its batch-1 size and is"
  echo "  what will actually OOM you. Treat this as a shortlist for the sweep, never as"
  echo "  a result: the number to trust is the gpu_peak_gb that bench_models.py measures."
  printf '    %-28s %-10s %-10s\n' VARIANT bf16 4-bit
  fits() { [ "$1" -le "$vram_gb" ] && echo "fits" || echo "no"; }
  printf '    %-28s %-10s %-10s\n' "gemma-4-E2B-it   (5B)"  "$(fits 12)" "$(fits 4)"
  printf '    %-28s %-10s %-10s\n' "gemma-4-E4B-it   (8B)"  "$(fits 18)" "$(fits 6)"
  printf '    %-28s %-10s %-10s\n' "gemma-4-12B-it"         "$(fits 28)" "$(fits 9)"
  printf '    %-28s %-10s %-10s\n' "gemma-4-26B-A4B-it"     "$(fits 62)" "$(fits 18)"
  printf '    %-28s %-10s %-10s\n' "gemma-4-31B-it"         "$(fits 74)" "$(fits 22)"
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  if [ -n "$cc" ]; then
    echo
    kv "compute capability" "$cc"
    # bf16 needs Ampere (8.0+); older cards must run --dtype float16.
    case "$cc" in
      8.*|9.*|1*) echo "    bf16: supported.  flash_attention_2: supported (Ampere+)." ;;
      *)          echo "    bf16: NOT supported on this card — use --dtype float16." ;;
    esac
  fi
fi
echo
echo "  Open questions this survey is meant to answer, in priority order:"
echo "    1. driver version -> stock PyPI torch, or the cu121 index?"
echo "    2. RWX StorageClass -> can 12 pods + the Notebook share one PVC?"
echo "       (single box, single GPU: parallelism is 1, so RWO may be enough — but"
echo "        the Notebook and the Job still overlap, so check it anyway)"
echo "    3. free space on the chosen scratch path -> >=200 GiB for the HF cache?"
echo "    4. egress to huggingface.co and pypi.org -> or is a mirror needed?"

}

survey 2>&1 | tee "$OUT"
printf '\nSaved to: %s\n' "$OUT"
