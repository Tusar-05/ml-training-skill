#!/usr/bin/env bash
# Creates a uv environment with the correct PyTorch build for the detected accelerator.
# Usage: bash setup_env.sh <project-dir> [cpu|cuda|rocm] [--python 3.12] [--dry-run]

set -u

PROJECT_DIR=""
TARGET="auto"
PYVER="3.12"
DRY_RUN=0

usage() {
  cat >&2 <<'EOF'
usage: bash setup_env.sh <project-dir> [cpu|cuda|rocm|auto] [--python <ver>] [--dry-run]
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --python)     PYVER="${2:-3.12}"; shift 2 ;;
    --python=*)   PYVER="${1#*=}"; shift ;;
    --dry-run)    DRY_RUN=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    cpu|cuda|rocm|auto) TARGET="$1"; shift ;;
    -*)           echo "unknown option: $1" >&2; usage; exit 1 ;;
    *)            [ -z "$PROJECT_DIR" ] && PROJECT_DIR="$1"; shift ;;
  esac
done

if [ -z "$PROJECT_DIR" ]; then
  usage
  exit 1
fi

detect_target() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
    echo cuda
  elif command -v rocminfo >/dev/null 2>&1; then
    echo rocm
  else
    echo cpu
  fi
}

[ "$TARGET" = auto ] && TARGET="$(detect_target)"

case "$TARGET" in
  cuda) TORCH_INDEX="https://download.pytorch.org/whl/cu128"; EXTRA="torchvision torchaudio" ;;
  rocm) TORCH_INDEX="https://download.pytorch.org/whl/rocm6.2"; EXTRA="torchvision torchaudio" ;;
  cpu)  TORCH_INDEX="https://download.pytorch.org/whl/cpu";   EXTRA="torchvision torchaudio" ;;
  *) echo "unknown target: $TARGET (expected cpu|cuda|rocm|auto)" >&2; exit 1 ;;
esac

cat <<PLAN
ML environment plan
  project dir : $PROJECT_DIR
  target      : $TARGET
  python      : $PYVER (uv-managed)
  torch index : $TORCH_INDEX
  extras      : $EXTRA
PLAN

if [ "$DRY_RUN" -eq 1 ]; then
  echo "dry run: no changes made"
  exit 0
fi

command -v uv >/dev/null 2>&1 || { echo "uv not found - install uv first" >&2; exit 1; }
mkdir -p "$PROJECT_DIR"

cd "$PROJECT_DIR" || exit 1
uv venv --python "$PYVER" .venv || exit 1
# shellcheck disable=SC1091
. .venv/bin/activate

uv pip install numpy pandas scikit-learn || exit 1
# Torch must come from the matching index; the default PyPI wheel is often the wrong build.
uv pip install torch $EXTRA --index-url "$TORCH_INDEX" || exit 1

echo
echo "--- verification ---"
python - <<'PY'
import torch
ok = torch.cuda.is_available()
print("torch        :", torch.__version__)
print("cuda available:", ok)
if ok:
    print("device       :", torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print("compute cap  :", cap)
    if cap[0] >= 12:
        print("note         : Blackwell (sm_120+) - requires CUDA 12.8+ builds")
else:
    print("device       : CPU (fine for sklearn and small CNNs)")
PY
echo
echo "Next: activate with 'source $PROJECT_DIR/.venv/bin/activate', or use 'uv run'."
