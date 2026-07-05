#!/usr/bin/env bash
# Export CUDA library paths for onnxruntime-gpu in WSL + PyTorch cu13 venv.
set -euo pipefail

VENV="${VENV:-${VIRTUAL_ENV:-}}"
if [[ -z "${VENV}" && -f ".venv/bin/activate" ]]; then
  VENV="$(pwd)/.venv"
fi
if [[ -z "${VENV}" ]]; then
  echo "Activate your venv first, or set VENV=/path/to/.venv" >&2
  return 1 2>/dev/null || exit 1
fi

PYVER="$(find "${VENV}/lib" -maxdepth 1 -type d -name 'python3.*' | head -1 | xargs basename)"
SITE="${VENV}/lib/${PYVER}/site-packages"
CUDA_LIB="${SITE}/nvidia/cu13/lib"
if [[ ! -d "${CUDA_LIB}" ]]; then
  echo "CUDA libs not found at ${CUDA_LIB}. Install torch with CUDA first." >&2
  return 1 2>/dev/null || exit 1
fi

export LD_LIBRARY_PATH="${CUDA_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
echo "LD_LIBRARY_PATH includes ${CUDA_LIB}"
