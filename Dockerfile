# Horizon inference image for a CUDA Kubernetes/Kubeflow cluster.
#
#   docker build -t <registry>/horizon:<tag> .
#   docker push  <registry>/horizon:<tag>
#
# The image carries the *dependencies*; the working copy is expected on the shared PVC
# and mounted over /opt/horizon at run time. `pip install -e` records a path rather than
# copying files, so overlaying that path with the PVC checkout yields live code with no
# rebuild — which is what keeps the sweep's edit/rerun loop usable on k8s at all. The
# baked copy underneath is the fallback for a frozen run with no mount.

# Must match the node's driver: the torch wheel bundles CUDA 12.x and needs a driver
# that supports it. Ask the admins for the driver version, then pin this by digest once
# the first build succeeds. If the driver is older than 12.x, drop to a cu121 base and
# install torch from https://download.pytorch.org/whl/cu121.
ARG CUDA_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu24.04
FROM ${CUDA_IMAGE}

# Ubuntu 24.04 ships Python 3.12 as `python3` — exactly the floor pyproject.toml sets.
ENV DEBIAN_FRONTEND=noninteractive
# build-essential and python3-dev are not optional: torch pulls in triton, which
# JIT-compiles its CUDA driver shim as a CPython extension at the first kernel launch.
# Without them the image installs cleanly, passes the test suite, and then generates
# zero tokens — every item failing with "Failed to find C compiler" or "Python.h: No
# such file or directory" buried under a caught exception.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev build-essential \
        git ca-certificates bash procps rsync \
    && rm -rf /var/lib/apt/lists/*

# A venv rather than --break-system-packages: Ubuntu 24.04 marks the system Python
# externally-managed (PEP 668) and pip refuses to write into it.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /opt/horizon

# Dependencies in their own layer: this is the expensive one (torch alone is GBs) and a
# source edit must not invalidate it.
COPY requirements-gpu.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-gpu.txt

# --no-deps for the same reason the box does it by hand: pyproject.toml's Intel-Mac
# ceilings (transformers<5, numpy<2) would undo the resolution above.
COPY . .
RUN pip install --no-cache-dir -e . --no-deps

# Progress goes to stderr and the machine-readable summary to stdout; unbuffered keeps
# `kubectl logs -f` live instead of arriving in block-sized bursts when the pod ends.
ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD ["bash"]
