# ml-training-skill

An [opencode](https://opencode.ai) skill for training AI/ML models correctly:
hardware triage, environment setup, dataset discipline, reproducible training,
crash triage, resume, and export.

## What's inside

| Path | Purpose |
|---|---|
| `SKILL.md` | The skill entry point: routing table, non-negotiable rules, 7-step procedure |
| `references/hardware-and-vram.md` | Probe commands, torch build matrix, VRAM math, feasibility tiers, 8 GB model zoo |
| `references/dataset-playbook.md` | Acquisition, splits (group/time-aware), leakage checks, balance, licensing |
| `references/training-recipes.md` | Runnable skeletons: sklearn baseline, CNN, HF Trainer, QLoRA, audio classification, export/quantize, error analysis |
| `references/error-cookbook.md` | Real training error strings → cause → exact fix |
| `references/monitoring-and-resume.md` | Backgrounding long runs, live monitoring, checkpoint resume, crash triage |
| `scripts/probe_gpu.sh` | Detects accelerator (NVIDIA/ROCm/CPU), VRAM, and usable Python versions |
| `scripts/setup_env.sh` | Creates a `uv` environment with the correct PyTorch build for the detected hardware |
| `scripts/monitor_run.sh` | Live view of a run directory: metrics tail, checkpoints, log, PID, GPU |

## Install (opencode)

```bash
git clone https://github.com/Tusar-05/ml-training-skill.git
cp -r ml-training-skill ~/.config/opencode/skills/ml-training
```

Restart opencode; the skill then triggers on requests like "train a model",
"fine-tune", "LoRA", "CUDA out of memory", "resume checkpoint", or "export onnx".

## Quick start

```bash
bash scripts/probe_gpu.sh                      # what can this machine actually train?
bash scripts/setup_env.sh ~/my-project cpu     # or cuda / rocm; --dry-run to preview
bash scripts/monitor_run.sh runs/<name>        # live view while training
```

See `SKILL.md` for the full procedure.
