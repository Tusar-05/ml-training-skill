# Monitoring and Resume — never block, never lose a run

## 1. Launch a long run in the background

Foreground training blocks the whole session. Always background it with a log:

```bash
cd <project-dir>
nohup uv run python train.py > runs/<name>/logs/train.log 2>&1 &
echo "PID $!" | tee runs/<name>/logs/pid.txt
```

Or inside tmux (survives SSH drop, easy attach):

```bash
tmux new -s train -d "cd <project-dir> && uv run python train.py 2>&1 | tee runs/<name>/logs/train.log"
tmux attach -t train      # Ctrl-b d to detach
```

Report to the user immediately: PID, log path, and the time estimate from Step 3.
Then **poll** — do not wait in the foreground.

## 2. Polling commands

```bash
bash scripts/monitor_run.sh runs/<name>          # one-shot snapshot (below)
tail -f runs/<name>/logs/train.log               # live log
watch -n 10 'tail -3 runs/<name>/metrics.csv'    # live metrics
nvidia-smi                                       # GPU util/VRAM while it runs
ps -p "$(cat runs/<name>/logs/pid.txt | awk '{print $2}')" -o pid,etime,cmd   # still alive?
```

`scripts/monitor_run.sh` prints: last metrics row, epoch count, checkpoint
timestamps and sizes, whether the PID is alive, and GPU state when `nvidia-smi`
works. Use `--watch 10` to refresh every 10 s.

## 3. Stopping safely

- Polite stop (lets the script run its checkpoint-on-exit path, if implemented):
  `kill -INT <pid>` (Ctrl-C equivalent).
- Hard stop: `kill <pid>` (TERM), then `kill -9 <pid>` only if it refuses.
- In tmux: `tmux kill-session -t train`.
- Implement checkpoint-on-SIGTERM in the training script when runs are long:

```python
import signal
def _save_and_exit(signum, frame):
    save_checkpoint(path="runs/<name>/checkpoints/last.pt")   # full state
    raise SystemExit(143)
signal.signal(signal.SIGTERM, _save_and_exit)
```

## 4. Resume from checkpoint

Checkpoints must hold **full training state**, not just weights:

```python
def save_checkpoint(path, model, opt, sched, epoch, best_metric, rng=None):
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict() if sched is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, path)

def load_checkpoint(path, model, opt, sched=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    if sched is not None and ckpt.get("scheduler"):
        sched.load_state_dict(ckpt["scheduler"])
    torch.set_rng_state(ckpt["torch_rng"].cpu())
    if ckpt.get("cuda_rng") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([r.cpu() for r in ckpt["cuda_rng"]])
    return ckpt["epoch"], ckpt["best_metric"]
```

Wiring into the training loop:

```python
start_epoch, best = 1, float("inf")
if resume_from and Path(resume_from).exists():
    start_epoch, best = load_checkpoint(resume_from, model, opt, sched)
    print(f"resumed from {resume_from} at epoch {start_epoch}")

for epoch in range(start_epoch, num_epochs + 1):
    ...
```

Also pass `resume_from="runs/<name>/checkpoints/last.pt"` when relaunching.
Hugging Face Trainer handles this natively: `Trainer(..., resume_from_checkpoint="runs/<name>")`.

## 5. Crash triage (in this order)

1. `tail -50 runs/<name>/logs/train.log` — find the **first** traceback.
2. `tail -5 runs/<name>/metrics.csv` — which epoch/step did it reach?
3. `dmesg | grep -i -e oom -e kill` — system OOM killer (RAM) vs Python exception
   (VRAM/logic). A bare `Killed` in the log with no traceback is the OOM killer.
4. Match the message in `references/error-cookbook.md` — apply its fix, don't
   improvise.
5. Relaunch with `resume_from=.../last.pt` so the crashed hours are not lost.

## 6. Time budgeting

- Before launch: estimate total wall-clock from measured step time:
  `sec_per_step × steps_per_epoch × epochs`. Put the number in the user report.
- Hard ceiling: wrap the command in `timeout` for untrusted/new scripts so a
  runaway cannot occupy the machine all night:
  `timeout 4h uv run python train.py ...` — note `timeout` sends SIGTERM, so the
  checkpoint-on-signal handler above is what makes this safe.
- On a laptop: warn the user the machine will be busy and may throttle/heat;
  do not start multi-hour jobs without explicit approval.
