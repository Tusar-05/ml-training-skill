#!/usr/bin/env bash
# ML training environment probe: reports accelerator, VRAM, and usable Python.
# Usage: bash probe_gpu.sh

set -u

section() { printf '\n== %s ==\n' "$1"; }

section "CPU"
if [ -r /proc/cpuinfo ]; then
  model=$(awk -F': ' '/^model name/{print $2; exit}' /proc/cpuinfo)
  printf '  %s (%s threads)\n' "${model:-unknown}" "$(nproc 2>/dev/null || echo '?')"
else
  printf '  /proc/cpuinfo unavailable\n'
fi

section "RAM"
if [ -r /proc/meminfo ]; then
  awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{printf "  %.1f GiB total, %.1f GiB available\n", t/1048576, a/1048576}' /proc/meminfo
fi

section "GPU (NVIDIA)"
if command -v nvidia-smi >/dev/null 2>&1; then
  if gpu_info=$(nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader 2>/dev/null) && [ -n "$gpu_info" ]; then
    printf '  %s\n' "$gpu_info"
    vram_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "${vram_mb:-}" ] && printf '  VRAM: %s MiB (~%.1f GiB usable for training)\n' "$vram_mb" "$(awk -v v="$vram_mb" 'BEGIN{print (v/1024)*0.9}')"
  else
    printf '  nvidia-smi present but cannot talk to the driver\n'
    printf '  -> driver/module not loaded, or the dGPU is disabled in firmware (BIOS)\n'
  fi
else
  printf '  nvidia-smi not installed (no NVIDIA driver)\n'
fi

section "GPU (PCI enumeration)"
if command -v lspci >/dev/null 2>&1; then
  lspci 2>/dev/null | grep -Ei 'vga|3d|display' | sed 's/^/  /' || printf '  no display devices found\n'
  if ! lspci 2>/dev/null | grep -qiE 'nvidia'; then
    printf '  NOTE: no NVIDIA PCI device visible -> if a card exists, firmware has it disabled\n'
  fi
else
  printf '  lspci not installed\n'
fi

section "GPU (ROCm)"
if command -v rocminfo >/dev/null 2>&1; then
  rocminfo 2>/dev/null | grep -m2 'Name:' | sed 's/^/  /'
else
  printf '  rocminfo not installed (no ROCm)\n'
fi

section "Python"
py=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then py="$cand"; break; fi
done
if [ -n "$py" ]; then
  pyver=$("$py" -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])' 2>/dev/null || echo unknown)
  printf '  %s -> %s\n' "$py" "$pyver"
  case "$pyver" in
    3.14*|3.15*|3.16*|4.*)
      printf '  WARNING: %s is newer than deep-learning wheel support (use 3.11-3.13)\n' "$pyver" ;;
  esac
  torch_state=$("$py" -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>/dev/null || true)
  if [ -n "$torch_state" ]; then
    printf '  torch -> %s\n' "$torch_state"
  else
    printf '  torch -> not installed\n'
  fi
else
  printf '  no python interpreter found\n'
fi
command -v uv >/dev/null 2>&1 && printf '  uv -> %s\n' "$(uv --version 2>/dev/null)"

section "Suggested tier"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits >/dev/null 2>&1; then
  vram_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  if awk -v v="${vram_mb:-0}" 'BEGIN{exit !(v>=1000)}'; then
    if   awk -v v="$vram_mb" 'BEGIN{exit !(v<=6000)}';  then printf '  NVIDIA 4-6 GiB -> classic CNNs, frozen-backbone transfer, LoRA <=1B\n'
    elif awk -v v="$vram_mb" 'BEGIN{exit !(v<=12000)}'; then printf '  NVIDIA 8-12 GiB -> detection/YOLO, BERT-class FT, QLoRA 7B (short ctx)\n'
    elif awk -v v="$vram_mb" 'BEGIN{exit !(v<=26000)}'; then printf '  NVIDIA 16-24 GiB -> QLoRA 13-34B, SDXL LoRA, full FT <=3B\n'
    else printf '  NVIDIA 40 GiB+ -> full FT 7-13B, small-model pretraining\n'; fi
  fi
elif command -v rocminfo >/dev/null 2>&1; then
  printf '  ROCm present -> verify the specific GPU is supported before relying on it\n'
else
  printf '  CPU-only -> scikit-learn/XGBoost, small CNNs on subsets, tiny fine-tunes (slow)\n'
  printf '  For anything bigger: cloud GPU (Colab / Kaggle) or enable a discrete card\n'
fi
