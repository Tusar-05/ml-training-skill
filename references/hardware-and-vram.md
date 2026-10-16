# Hardware, VRAM Budgets, and Accelerator Setup

## 1. Probing the machine

```bash
bash scripts/probe_gpu.sh          # this skill's probe: accelerator + python
nvidia-smi                          # NVIDIA driver + card + live VRAM (if installed)
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
rocminfo | grep -m2 Name            # AMD compute (ROCm)
lspci | grep -Ei 'vga|3d|display'   # raw PCI: shows what the firmware exposed
python3 -c "import torch;print(torch.__version__, torch.cuda.is_available())"
```

Reading the results:

| Symptom | Meaning |
|---|---|
| `nvidia-smi: command not found` | No NVIDIA driver installed — install before any CUDA work |
| `nvidia-smi` fails "couldn't communicate with the NVIDIA driver" | Driver installed but kernel module not loaded / card disabled |
| Card absent from `lspci` | **Firmware/BIOS has it disabled** — software cannot fix this |
| `torch.cuda.is_available()` is `False` on a machine with a card | Wrong torch build (CPU wheel or mismatched CUDA) — reinstall from the correct index |
| `compute_cap` = 12.0 | Blackwell (RTX 50-series): requires CUDA 12.8+ builds |

Hybrid laptops: leaving the dGPU in BIOS hybrid mode and letting the iGPU drive the
display usually gives the cleanest setup — the dGPU stays available for compute
without touching the desktop/compositor configuration.

## 1b. This machine's profile (Lenovo 83JG)

| Component | Value | Training implication |
|---|---|---|
| CPU | AMD Ryzen 7 250, 8c/16t | Usable for classical ML and small CNNs; transformer FT is hours-scale |
| RAM | 16 GB (+ swap) | Cap DataLoader `num_workers` ≤ 4; don't hold full datasets in memory |
| iGPU | Radeon 780M (RDNA3) | ROCm does **not** support this iGPU; treat as CPU |
| dGPU | RTX 5060 Laptop, 8 GB GDDR7, **disabled in BIOS** | Invisible to `lspci`; enable in BIOS to unlock CUDA |
| dGPU compute cap | 12.0 (Blackwell) | Needs torch with sm_120 → CUDA 12.8+ / torch 2.7+ builds only |
| Kernel | 7.2.5-3-omarchy (Arch) | After enabling dGPU: `nvidia-open-dkms` already installed; reboot, then re-probe |

Expected probe result **today**: CPU-only tier. After enabling the dGPU in BIOS and
rebooting: NVIDIA 8–12 GiB tier (~7.2 GB usable after display headroom on hybrid).
Re-run `scripts/probe_gpu.sh` after any BIOS/driver change; do not assume.

## 2. Torch build selection

Install from the index that matches the accelerator; default PyPI wheels are the
CPU-only or one specific CUDA build and frequently the wrong one.

| Target | Index URL |
|---|---|
| CPU | `https://download.pytorch.org/whl/cpu` |
| NVIDIA, Blackwell (sm_120, RTX 50-series) | `https://download.pytorch.org/whl/cu128` or newer |
| NVIDIA, older (sm_75–sm_90) | matching `cu12x` index for the installed driver |
| AMD ROCm | `https://download.pytorch.org/whl/rocm6.x` |

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

ROCm caveat: RDNA3 **integrated** GPUs (e.g. Radeon 780M) are not officially
supported by ROCm. Unofficial `HSA_OVERRIDE_GFX_VERSION` hacks are unreliable —
treat iGPU acceleration as a gamble and prefer CPU or a discrete card.

Post-install verification is mandatory:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

## 3. Memory math (do this instead of guessing)

Per-parameter bytes:

| Precision | Bytes/param |
|---|---|
| fp32 | 4 |
| bf16 / fp16 | 2 |
| int8 | 1 |
| nf4 / 4-bit | ~0.5 |

Full fine-tuning with AdamW and mixed precision, rule of thumb:

```
total ≈ params × (2 weights + 2 grads + 8 optimizer) + activations
      ≈ params × 12 bytes  + activations
```

| Model | Full FT (bf16 + AdamW) | LoRA / QLoRA (4-bit base) |
|---|---|---|
| 110M (BERT-base) | ~1.5–2 GB | ~0.5 GB |
| 1–3B | 12–40 GB | 2–4 GB |
| 7–8B | 80–110 GB | 5–7 GB |
| 13B | 160–200 GB | 8–10 GB |

Activations scale with `batch × sequence_length × hidden_size × layers`, so sequence
length is the biggest lever at fixed VRAM. Reduce it before reducing the model.

## 4. Fitting a job into a given budget

| Lever | Effect | Cost |
|---|---|---|
| LoRA / QLoRA instead of full FT | 5–20× less memory | Slightly lower ceiling |
| Gradient accumulation | Bigger effective batch at fixed memory | Slower per step |
| Gradient checkpointing | Large activation savings | ~20–30% slower |
| Shorter sequence / smaller input | Linear-to-quadratic savings | Possible accuracy loss |
| Frozen backbone | Trains only the head | Limited adaptation |
| bf16 mixed precision | ~Halves weight+grad memory | Not for all ops |

## 5. Feasibility tiers

| Tier | Fits in | Realistic tasks |
|---|---|---|
| CPU only | 8–12 GB system RAM | scikit-learn, XGBoost, small CNNs on subsets, tiny transformer fine-tunes (slow) |
| 4–6 GB VRAM | small GPU | classic CNNs, transfer learning with frozen backbone, LoRA on ≤1B |
| 8–12 GB VRAM | mid GPU | YOLO/detection training, BERT-class full FT, QLoRA 7B (short context), SD1.5 LoRA |
| 16–24 GB VRAM | large GPU | QLoRA 13–34B, SDXL LoRA, full FT ≤3B, Whisper-medium FT |
| 40–80 GB+ | datacenter | Full FT 7–13B, small-model pretraining |
| Multi-node | cluster | LLM pretraining from scratch |

If the job lands more than one tier above the hardware, stop and offer the user the
choice: shrink the job, or move it to rented/cloud GPUs (Colab, Kaggle's weekly free
GPU quota, or a paid instance). Never quietly downgrade the task and present it as
equivalent — report what was actually trained.

## 5b. Realistic timing expectations

Always **measure the first 100 steps** and extrapolate; these are planning numbers
only, for a quiet machine (background load changes everything):

| Job | CPU-only (this laptop) | 8 GB GPU (once dGPU enabled) |
|---|---|---|
| scikit-learn / XGBoost (10⁴–10⁵ rows) | seconds–minutes | n/a (CPU job) |
| Small CNN, MNIST-scale | ~2–5 min total | seconds–minutes total |
| ResNet-18, CIFAR-10 (50k imgs, 20–50 epochs) | hours (use a subset) | ~10–30 min |
| BERT-base/DistilBERT text FT (short seq, few epochs) | many hours | ~10–40 min |
| Whisper-small / Wav2Vec2-base audio FT | hours | ~15–60 min |
| QLoRA on a 7B LLM (short ctx) | not practical | ~1–3 h (tight VRAM) |

Rules of thumb: quote an estimate before starting; re-quote after step 100; if the
extrapolated runtime exceeds what the user expects, stop and re-scope.

## 5c. Model zoo that fits 8 GB VRAM

Pretrained starting points that comfortably train (full FT or LoRA) on the 8 GB
tier. Prefer these over inventing architectures.

| Domain | Model id | Fit / notes |
|---|---|---|
| Vision backbone | `resnet18`, `efficientnet-b0`, `convnext-tiny` | Full FT, batch 32–64, 224px |
| Detection | `yolov8n` / `yolov8s` (ultralytics) | Nano/small only at 8 GB |
| Text classification | `distilbert-base-uncased`, `microsoft/deberta-v3-small` | Full FT, seq 256, batch 16–32 |
| Audio classification | `facebook/wav2vec2-base`, `microsoft/wavlm-base-plus` | FT head + last layers, 16 kHz mono, ≤10 s clips |
| ASR / deepfake-adjacent | `openai/whisper-small` | LoRA or FT with `transformers` Seq2Seq; medium is tight |
| Small LLM (QLoRA) | `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`, `meta-llama/Llama-3.2-1B-Instruct` | LoRA r=16, short ctx (512–1024) |
| 7B LLM (QLoRA, tight) | `Qwen/Qwen2.5-7B-Instruct` | nf4 + gradient checkpointing, ctx ≤ 1024 |

If a chosen model OOMs, walk the lever table (§4) before switching models.

## 6. Laptop-specific notes

Laptops throttle: check sustained clocks and power limits
(`nvidia-smi -q -d POWER,CLOCK`) rather than trusting the first 30 seconds of
throughput. Plan for a slower steady-state rate when quoting a training-time
estimate, and watch temperatures on long runs. Do not run a multi-hour job on a
laptop without confirming the user wants to keep the machine busy that long.
