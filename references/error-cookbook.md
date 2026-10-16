# Error Cookbook — real error strings → cause → fix

Match the error string, apply the fix. If the error is not listed, search the log
tail for the **first** Python traceback (later errors are often cascade noise).

## Import / environment errors

| Error contains | Cause | Fix |
|---|---|---|
| `Torch not compiled with CUDA enabled` | CPU wheel installed on a CUDA machine | Reinstall from the CUDA index: `uv pip install torch --index-url https://download.pytorch.org/whl/cu128` (or the index matching the driver) |
| `no kernel image is available for execution on the device` | Torch build too old for this GPU's compute capability (classic on Blackwell sm_120 / RTX 50-series) | Upgrade torch to a build that includes sm_120 (CUDA 12.8+, torch 2.7+); do not try to work around it |
| `CUDA driver version is insufficient for CUDA runtime version` | Driver older than the torch build expects | Update the NVIDIA driver, or install a torch build matching the installed driver (check `nvidia-smi` header vs `torch.version.cuda`) |
| `libcudnn.so...: cannot open shared object file` / `libtorch_cuda.so: cannot open shared object file` | Broken/partial install, or env mismatch (venv not activated) | `uv pip uninstall torch && uv pip install torch --index-url <correct-index>`; confirm the venv is activated |
| `CUDA setup failed unexpectedly` (bitsandbytes) | bitsandbytes wheel built for a different CUDA/torch than installed | Reinstall matching versions: `uv pip install -U bitsandbytes` from source or the matching release; verify `torch.version.cuda` first |
| `Your setup doesn't support bfloat16` | GPU predates Ampere (no bf16 hardware) | Use fp16 with `torch.amp.GradScaler` instead of bf16 |
| `ModuleNotFoundError: No module named 'X'` | Missing dependency or wrong venv | Install into the project venv (`uv pip install X`), not system python |

## CUDA runtime errors

| Error contains | Cause | Fix |
|---|---|---|
| `CUDA out of memory. Tried to allocate ...` (GPU VRAM) | Batch/sequence/model too large | Apply the ladder in order: halve batch → gradient accumulation to keep effective batch → `model.gradient_checkpointing_enable()` → shorter `max_length` → smaller model/LoRA. Compute the budget (`hardware-and-vram.md` §3) instead of retrying blindly |
| `CUDA error: out of memory` during eval | Eval also needs VRAM; `no_grad` isn't enough if batch is huge | Lower eval batch size (eval batch can differ from train batch) |
| `Found no NVIDIA driver on your system` | Card absent/disabled (this laptop: dGPU off in BIOS) or driver not loaded | `lspci | grep -i nvidia` to check firmware exposure; if absent, enable in BIOS. Software cannot fix a firmware-disabled card |
| `torch.cuda.is_available()` is `False` with no exception | Usually wrong torch build; sometimes kernel module not loaded after kernel update | `lsmod | grep nvidia`; if loaded, wrong build → reinstall from correct index; if not, `sudo modprobe nvidia` or reboot (dkms rebuild) |
| `CUDA error: device-side assert triggered` | Almost always an illegal memory access from bad input — most common: **label index ≥ num_labels** | Print `y.min(), y.max()` and `logits.shape[-1]`; remap labels to `0..C-1`. Re-run with `CUDA_LAUNCH_BLOCKING=1` to localize |
| `CUDA error: an illegal memory access was encountered` | Corrupt tensor or bad kernel; sometimes OOM cascade | Set `CUDA_LAUNCH_BLOCKING=1`, re-run to get the real line; check inputs for NaN/garbage |
| `CUDA error: unknown error` after suspend/resume or GPU reset | Driver state corrupted | Restart the process; if persistent, reboot. Save checkpoints frequently on laptops |

## Process / system errors

| Symptom | Cause | Fix |
|---|---|---|
| Process prints `Killed` (no traceback) | Linux **OOM killer** hit system RAM, not VRAM | Reduce `num_workers` in DataLoader (each worker copies data), avoid loading whole dataset into memory (`dataset = load_dataset(...)` then `.map` lazily), close other apps; check `dmesg | grep -i -e oom -e kill` |
| `Cannot re-initialize CUDA in forked subprocess` | DataLoader workers + CUDA: workers forked after CUDA init | `DataLoader(..., num_workers=0)` on Windows/small runs, or set `torch.multiprocessing.set_start_method("spawn")` **before** CUDA init, or create workers before moving the model to GPU |
| Hang at start of training, 100% CPU, no output | DataLoader bottleneck or deadlock (`num_workers` too high on small machines) | Try `num_workers=0` first; then 2; `persistent_workers=True` for multi-epoch; on this 16 GB laptop keep it ≤4 |
| `No space left on device` | Checkpoints + logs + HF cache filling disk | `du -sh ~/.cache/huggingface runs/*`; delete stale runs; set `HF_HOME` to a bigger disk; keep `save_total_limit=2` |
| `Killed` only under `torch.compile` | Inductor codegen memory spike | Drop `torch.compile`, or compile after warmup; on a new GPU/driver wait for support |

## Data / Hugging Face errors

| Error contains | Cause | Fix |
|---|---|---|
| `401 Unauthorized` / `GatedRepoError` / `Repository not found` | Gated model/dataset needs authentication | `uv pip install -U huggingface_hub && hf auth login`, or export `HF_TOKEN=...`; request access on the model page |
| `Couldn't reach` / `Connection error` / `Offline mode is enabled` | No network or offline cache | Check network; for air-gapped use `local_files_only=True` with a pre-populated `HF_HOME` |
| `DatasetNotFoundError` / `Can't load dataset` | Dataset moved, renamed, or script-based repo deprecated | Look up the dataset page for the new id; prefer Parquet/CSV data files over legacy loading scripts |
| `Couldn't find any data file` (CSV/arrow path) | Relative path resolved from wrong cwd | Use absolute paths; remember background jobs may start in a different directory |
| `ValueError: ... expected input ...` after tokenizer change | Mixing tokenizer of model A with weights of model B | Always load tokenizer and model from the **same** checkpoint id |

## Training-instability errors (no exception, just bad numbers)

| Symptom | Cause | Fix |
|---|---|---|
| Loss is `nan` from step 1 | LR too high, bad init, or NaN already in data | Drop LR 10×; scan inputs for NaN (`torch.isnan(x).any()`); check labels are valid indices |
| Loss goes `nan` after a few steps | bf16/fp16 overflow, LR spike, or exploding gradients | Switch to fp16+GradScaler or fp32 for the unstable layer; lower LR; keep `clip_grad_norm_(..., 1.0)` |
| Loss plateaus at the majority-class value | Labels imbalanced or shuffled wrong (x/y misaligned) | Verify a batch: `print(x.shape, y[:8])`; use macro-F1 not accuracy; check stratification |
| Train loss ↓ but val flat/worse from epoch 1 | Overfitting or leakage in reverse (val harder than train) | Early stopping; augmentation; smaller model; confirm split is sane (`dataset-playbook.md`) |
| Val metric identical to baseline forever | Model not training (frozen params, `eval()` mode left on, wrong `requires_grad`) | `model.train()` each epoch; print `sum(p.numel() for p in model.parameters() if p.requires_grad)` — must be > 0 |

## Diagnosis order for any crash

1. Read the **first** traceback in `logs/train.log` (not the last line).
2. `tail -20 runs/<name>/metrics.csv` — did it die at a specific epoch boundary
   (checkpoint/eval) or mid-epoch (data/CUDA)?
3. `dmesg | grep -i -e oom -e kill` — system OOM killer vs Python exception.
4. Reproduce with `CUDA_LAUNCH_BLOCKING=1` and `num_workers=0` — localizes CUDA
   async errors and DataLoader races.
5. Only then change code. One variable at a time.
