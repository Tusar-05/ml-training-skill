---
name: ml-training
description: "Trains, fine-tunes, and evaluates AI/ML models on the local machine or a cloud fallback. Use when the user asks to train a model, fine-tune a pretrained model, do LoRA/QLoRA, set up a training environment, pick a model size that fits their GPU, fix CUDA/torch/VRAM errors, debug OOM during training, triage a crashed training run, resume from a checkpoint, choose hyperparameters, evaluate whether a model actually learned, export or quantize a trained model (ONNX/int8), or build audio/image/text classifiers. Covers classical ML (scikit-learn/XGBoost), small neural networks, transfer learning, parameter-efficient fine-tuning of transformers, and audio deepfake/spoof detection. Triggers: train model, fine-tune, finetune, LoRA, QLoRA, PEFT, epochs, batch size, OOM, CUDA out of memory, torch, PyTorch, dataset splits, leakage, overfitting, model card, checkpoint, resume, VRAM, no kernel image, training crash, export onnx, quantize, wav2vec2, spectrogram."
---

# Training AI/ML Models

A repeatable procedure for training a model correctly: know the hardware, pin the
environment, scope the run to what fits, protect the data, train reproducibly,
survive crashes, then prove it worked.

## When to use

Use this skill whenever the deliverable is a **trained model artifact** (weights,
metrics, a model card) or the task is to make one exist: training from scratch,
fine-tuning, PEFT/LoRA, hyperparameter work, diagnosing a failed run, resuming a
killed run, or exporting a trained model for deployment.

## When NOT to use

- Pure inference with an existing hosted API — no training involved.
- Data analysis or visualization with no model — use a data tool instead.
- Prompt engineering without weight updates.
- Scraping or labeling data only — that is a data-engineering task.

## Routing table (read the right file first)

| User request | Go to |
|---|---|
| "does X fit on my machine / which model size" | `references/hardware-and-vram.md` |
| "train / fine-tune / LoRA / set up env" | Step 2 + `references/training-recipes.md` |
| "get the dataset / split it / is there leakage" | `references/dataset-playbook.md` |
| "training crashed / OOM / error message / no kernel image" | `references/error-cookbook.md` |
| "resume the run / monitor progress / how long left" | `references/monitoring-and-resume.md` |
| "export / quantize / deploy the model" | `references/training-recipes.md` §6 |
| "audio deepfake / spoof detection / spectrograms" | `references/training-recipes.md` §5 |
| "is the model actually good / error analysis" | Step 6 below + `references/training-recipes.md` §7 |

## Non-negotiable rules

1. **Never claim a model "works" without held-out metrics and a saved artifact.**
   A printed loss curve is not evidence. Evidence = eval number on data the model
   never saw + an inference smoke test on a real sample + artifact path.
2. **Never tune on the test set.** Val for decisions, test exactly once at the end.
3. **Always seed everything** and save the config next to the weights.
4. **Always report a baseline** (majority class, random, or the untuned pretrained
   model). "95% accuracy" means nothing without a baseline.
5. **Do not silently move the user's workload to the cloud.** If the run does not
   fit locally, say so and ask before handing off to Colab/Kaggle.
6. **Never start a long run without a stated time/VRAM estimate.** Tell the user
   "this is ~40 min on your 8 GB card" before burning their compute. Quote from
   measured first-step time when possible, not from hope.
7. **Long runs go to the background with a log file; never block a session on them.**
   Launch with `nohup`/`tee` (or tmux), report the PID and log path, then poll.
   See `references/monitoring-and-resume.md`.
8. **Save enough state to resume.** A checkpoint that only holds weights cannot
   resume training properly — optimizer and scheduler state must be in it too.

## Procedure

### Step 1 — Hardware triage (always first)

Run the probe, don't guess:

```bash
bash scripts/probe_gpu.sh
```

It reports accelerator (NVIDIA / ROCm / CPU), VRAM, and usable Python versions.
Interpret the tier with `references/hardware-and-vram.md`. State the tier and its
implications to the user in one sentence before proceeding.

Common traps this catches: a disabled dGPU that is invisible to `lspci`; a Python
version newer than the deep-learning wheels support; a GPU present but with no
working driver (torch installs fine and then reports `cuda unavailable`).

On this laptop (Ryzen 7 250 / 16 GB / RTX 5060 disabled in BIOS) the expected answer
today is **CPU-only**; re-probe after any BIOS change.

### Step 2 — Environment

```bash
bash scripts/setup_env.sh <project-dir> [cpu|cuda|rocm]
```

With no argument it auto-detects. Key points:

- Use `uv` with a **wheel-supported Python** (3.11–3.13). Do not use a bleeding-edge
  interpreter — deep-learning wheels lag by months.
- Install torch from the **matching index URL**, not default PyPI, so the build
  matches the driver. Blackwell (RTX 50-series) needs a CUDA 12.8+ build — an older
  build fails at runtime with `no kernel image is available for execution`.
- Verify before training: `torch.cuda.is_available()` must be `True` and the device
  name must be the expected card. A silent `False` means the wrong build — reinstall,
  do not proceed.

### Step 3 — Scope the run to the hardware

Decide the task class and check it fits **before** writing a training loop:
classical baseline → small network → transfer learning → PEFT/LoRA. See
`references/hardware-and-vram.md` for the feasibility table, realistic timing
expectations, and a model zoo that fits 8 GB, and
`references/training-recipes.md` for the code skeleton per class.

If it does not fit:
1. Reduce: smaller model, LoRA instead of full fine-tune, lower sequence length,
   gradient accumulation instead of a bigger batch, gradient checkpointing, frozen
   backbone.
2. If still too big, prepare a cloud handoff (self-contained notebook + pinned
   requirements + estimated GPU-hours) and ask the user to choose.

### Step 4 — Data discipline

Full procedure in `references/dataset-playbook.md`. Minimum bar:

- Acquire from a source with a license and a dataset card (HF `load_dataset`,
  torchvision/torchaudio, or clearly-owned local files).
- Split **train / val / test** before any preprocessing that touches labels.
- Group-aware splits: rows from one speaker/patient/subject must not straddle
  splits (critical for audio deepfake data — otherwise the model memorizes voices).
- Time-aware splits for temporal data; never shuffle across time.
- Check for duplicates and near-duplicates across splits (leakage).
- Report class balance; if imbalanced, use macro-F1/AUROC and class weights — not
  raw accuracy.
- Record provenance: source, license, size, split seed, and label definitions.

### Step 5 — Train reproducibly

Required in every run:

- Seeded RNG (python, numpy, framework), deterministic flags where practical
- Hyperparameters in a config file saved into the run directory
- Checkpoint **best and last** with full state (model + optimizer + scheduler +
  epoch + RNG); early stopping on the val metric
- Metrics logged per epoch/step to `metrics.csv`, not just stdout
- Mixed precision on GPU (bf16 preferred on modern cards) unless it destabilizes
- Launch in the background with `tee`'d logs; track the PID
  (`references/monitoring-and-resume.md`)

### Step 6 — Verify and report

Before declaring done:

1. Eval on the held-out set, once.
2. Compare against the baseline from Step 3.
3. Inference smoke test: load the saved artifact **fresh in a new process** and
   predict on one real sample.
4. Error analysis, not just a headline number: per-class metrics, confusion
   matrix, and 3–5 example failures (`references/training-recipes.md` §7).
5. Write `MODEL_CARD.md` into the run directory: task, data provenance, splits and
   seeds, hyperparameters, val/test metrics, baseline comparison, hardware and
   wall-clock, known limitations.
6. Report honestly, including negative results. "Val loss plateaued; macro-F1 0.71
   vs 0.50 baseline" is a real result — say it that way.

### Step 7 — Export (only if asked)

If the user wants to deploy or share the model: set `model.eval()`, export to a
portable format (TorchScript / ONNX / safetensors), quantize if latency matters,
and re-verify the exported artifact against the original on the same inputs.
Recipe: `references/training-recipes.md` §6.

## Run directory contract

Every training run writes to a self-describing directory:

```
runs/<run-name>/
  config.yaml          # exact hyperparameters + seeds + dataset revision
  metrics.csv          # per-step/epoch train + val metrics
  checkpoints/
    best.pt            # full state: model+opt+sched+epoch+rng
    last.pt
  logs/
    train.log          # tee'd stdout/stderr of the run
  MODEL_CARD.md        # written on completion
```

## Anti-patterns (refuse to do these)

- Training without a validation split, then reporting training loss as performance
- Empty or fake results; fabricating metrics to satisfy an expectation
- Tuning hyperparameters against the test set
- Launching an unbounded run with no estimate, timeout, or checkpointing
- Blocking the session for hours waiting on a foreground training process
- Saving weights only (no optimizer state) and calling the checkpoint "resumable"
- Downloading datasets or model weights without checking the license or
  accepting gated-model terms (use `HF_TOKEN` when required)
- Random-splitting audio/medical data so the same speaker/patient lands in train
  and test
- "Fixing" OOM by blindly lowering batch size repeatedly instead of computing the
  budget (see `references/hardware-and-vram.md`)
- Deleting or weakening an eval to make a number look better
- Declaring success from train loss or from an eval on training data

## Reference files

- `references/hardware-and-vram.md` — probe commands, this machine's profile,
  torch build/index matrix, VRAM math, feasibility tiers, timing expectations,
  small-GPU model zoo
- `references/training-recipes.md` — runnable skeletons: baseline, CNN, HF
  Trainer, QLoRA, audio classification, export/quantize, error analysis
- `references/dataset-playbook.md` — acquisition, splits, leakage, balance,
  integrity checks, licensing
- `references/error-cookbook.md` — real error strings → cause → exact fix
- `references/monitoring-and-resume.md` — backgrounding, live monitoring,
  resume-from-checkpoint, crash triage
- `scripts/probe_gpu.sh` — accelerator + Python detection
- `scripts/setup_env.sh` — uv environment creation with correct torch build
- `scripts/monitor_run.sh` — live view of a run directory (metrics + checkpoints + GPU)
