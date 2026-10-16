# Training Recipes (runnable skeletons)

Skeletons, not complete programs: fill in the dataset and paths. Every recipe
assumes the Step 4/5 discipline from `SKILL.md` — splits first, seeded, checkpoints
saved, metrics logged, baseline reported.

## 0. Device selection and seeding (put in every script)

```python
import os, random
import numpy as np
import torch

SEED = 42
def seed_all(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = best_device()
print("device:", device, "| name:",
      torch.cuda.get_device_name(0) if device.type == "cuda" else "-")
seed_all()
```

bf16 is the safe default on modern NVIDIA cards; fall back to fp16 (with a GradScaler)
if you hit instability, and to fp32 on CPU:

```python
amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
```

## 1. Classical baseline (works on CPU; do this first)

Always cheaper than a neural net and often enough. It is also the baseline every
later number gets compared against.

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.dummy import DummyClassifier

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=SEED)

baseline = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
print("baseline majority-class:", baseline.score(X_te, y_te))

clf = HistGradientBoostingClassifier(random_state=SEED).fit(X_tr, y_tr)
print("val:", clf.score(X_va, y_va))
print(classification_report(y_te, clf.predict(X_te)))
```

## 2. Small CNN (PyTorch, device-agnostic)

Includes the pieces people skip and then regret: val loop, best/last checkpoint,
metrics file, early stopping.

```python
import csv, copy, time
from pathlib import Path
from torch.utils.data import DataLoader

run = Path("runs/cnn-run"); (run / "checkpoints").mkdir(parents=True, exist_ok=True)
early_stop_patience, best_val, bad_epochs = 5, float("inf"), 0

model = MyCNN().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = torch.nn.CrossEntropyLoss()

def evaluate(loader):
    model.eval(); correct = total = 0; loss_sum = 0.0
    with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype,
                                         enabled=device.type == "cuda"):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x); loss = loss_fn(logits, y)
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item(); total += y.size(0)
    return loss_sum / total, correct / total

with open(run / "metrics.csv", "w", newline="") as f:
    log = csv.writer(f); log.writerow(["epoch", "train_loss", "val_loss", "val_acc", "sec"])
    for epoch in range(1, 31):
        t0 = time.time(); model.train(); running = 0.0; seen = 0
        for x, y in train_loader:                        # train_loader: shuffle=True
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=amp_dtype,
                                enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * y.size(0); seen += y.size(0)

        val_loss, val_acc = evaluate(val_loader)
        log.writerow([epoch, running / seen, val_loss, val_acc, round(time.time() - t0, 2)])
        # full-state checkpoint: model + optimizer + rng so the run is truly resumable
        ckpt = {"model": model.state_dict(), "optimizer": opt.state_dict(),
                "epoch": epoch, "best_metric": best_val,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        torch.save(ckpt, run / "checkpoints" / "last.pt")
        if val_loss < best_val:
            best_val, bad_epochs = val_loss, 0
            torch.save({**ckpt, "model": copy.deepcopy(model).state_dict()},
                       run / "checkpoints" / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= early_stop_patience:
                print("early stop at epoch", epoch); break
```

Then load `best.pt` fresh and run the inference smoke test on one real sample before
reporting anything.

To resume a killed run, reload all state and start from the saved epoch (full
snippets in `references/monitoring-and-resume.md`):

```python
ckpt = torch.load(run / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"]); opt.load_state_dict(ckpt["optimizer"])
torch.set_rng_state(ckpt["torch_rng"].cpu())
start_epoch = ckpt["epoch"] + 1; best_val = ckpt["best_metric"]
```

## 3. Transformer fine-tune (HF `Trainer`)

Use for text classification, audio classification, and similar with a frozen-then-tuned
backbone. For 8 GB VRAM this is comfortable up to BERT-base scale.

```python
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from datasets import load_dataset

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
ds = load_dataset("csv", data_files="data.csv").map(
    lambda b: tok(b["text"], truncation=True, max_length=256), batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=N_LABELS)

args = TrainingArguments(
    output_dir="runs/hf-run", seed=SEED,
    eval_strategy="epoch", save_strategy="epoch",
    save_total_limit=2, load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    num_train_epochs=3, per_device_train_batch_size=16,
    gradient_accumulation_steps=2,            # grow effective batch without VRAM
    learning_rate=2e-5, bf16=torch.cuda.is_bf16_supported(),
    report_to="none",
)
Trainer(model=model, args=args, train_dataset=ds["train"],
        eval_dataset=ds["validation"], compute_metrics=macro_f1).train()
```

Keep a held-out `test` split untouched by the `Trainer`; evaluate it once afterwards.

## 4. QLoRA on a ~7B model (8–12 GB VRAM, tight but doable)

```bash
uv pip install transformers peft accelerate bitsandbytes datasets
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                       bnb_4bit_compute_dtype=torch.bfloat16,
                       bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained("some/model-7b", quantization_config=q,
                                             device_map={"": 0})
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora)
model.print_trainable_parameters()   # expect ~0.1-2% of params
```

Add `gradient_checkpointing_enable()` and keep sequence length short if it OOMs.
Prefer the `trl` `SFTTrainer` over hand-rolling the loop for instruction data.

## 5. Audio classification (deepfake / spoof detection)

Audio data has two extra traps: **sample-rate consistency** and **speaker leakage**
(the same voice in train and test makes the model memorize speakers, not spoofs).
Always split with `GroupShuffleSplit` on speaker/recording-id first — see
`references/dataset-playbook.md` §2.

```python
from datasets import load_dataset
from transformers import (AutoProcessor, AutoModelForAudioClassification,
                          TrainingArguments, Trainer)

MODEL = "facebook/wav2vec2-base"           # fits 8 GB VRAM; whisper-small for ASR-adjacent
proc = AutoProcessor.from_pretrained(MODEL)

def prepare(batch):
    # one rate, mono, duration cap for every clip
    import torchaudio, torch
    wav, sr = torchaudio.load(batch["path"])
    wav = wav.mean(0, keepdim=True)                       # mono
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.squeeze(0)[: 16000 * 10]                    # cap at 10 s
    inputs = proc(wav, sampling_rate=16000, return_tensors="pt")
    batch["input_values"] = inputs.input_values[0]
    return batch

ds = load_dataset("audio_folder", data_dir="data/") \
        .cast_column("audio", Audio(decode=False))        # keep paths, load lazily
ds = ds.map(prepare)                                      # then group-split as above

model = AutoModelForAudioClassification.from_pretrained(MODEL, num_labels=N_CLASSES)
args = TrainingArguments("runs/audio-run", seed=SEED, eval_strategy="epoch",
                         per_device_train_batch_size=4, learning_rate=1e-4,
                         num_train_epochs=5, bf16=torch.cuda.is_bf16_supported(),
                         report_to="none")
Trainer(model=model, args=args, train_dataset=ds["train"],
        eval_dataset=ds["validation"], compute_metrics=macro_f1).train()
```

Lighter CPU-friendly alternative: log-mel spectrograms + a small CNN
(`torchaudio.transforms.MelSpectrogram` → 2D conv). Slower to converge but trains
on this laptop's CPU where wav2vec2 FT would take hours.

## 6. Export and quantize (only when deploying)

Verify the exported artifact matches the original before shipping it.

```python
model = MyModel(); model.load_state_dict(torch.load("runs/x/checkpoints/best.pt",
                                                    map_location="cpu")["model"])
model.eval().cpu()
dummy = torch.randn(1, *INPUT_SHAPE)

# TorchScript
ts = torch.jit.trace(model, dummy); ts.save("model.pt")
# ONNX (requires: uv pip install onnx onnxruntime)
torch.onnx.export(model, dummy, "model.onnx", input_names=["x"],
                  output_names=["logits"], dynamic_axes={"x": {0: "batch"}})

# int8 dynamic quantization for transformer/linear-heavy models (~4× smaller, CPU-fast)
from torch.ao.nn.quantized import dynamic
qmodel = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
torch.save(qmodel.state_dict(), "model_int8.pt")
```

Parity check (must pass before declaring export done):

```python
with torch.no_grad():
    ref, exp = model(dummy), torch.jit.load("model.pt")(dummy)
assert torch.allclose(ref, exp, atol=1e-4), f"export drift: {(ref-exp).abs().max()}"
```

For transformers, prefer `save_pretrained()` + `safetensors`; ship the tokenizer /
processor alongside the weights — weights alone are not deployable.

## 7. Error analysis (do this before writing the model card)

A single headline metric hides everything that matters. Produce all four:

```python
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

y_pred = np.argmax(logits, 1)
print(classification_report(y_true, y_pred, digits=3))     # per-class recall/F1
print(confusion_matrix(y_true, y_pred))                    # which classes get mixed up

# worst-K examples: the model is most wrong where data is most suspicious
probs = torch.softmax(torch.tensor(logits), 1).numpy()
conf = 1 - probs[np.arange(len(y_true)), y_true]           # 1 - true-class prob
for i in np.argsort(conf)[-5:][::-1]:
    print(f"TRUE={y_true[i]} PRED={y_pred[i]} p={probs[i,y_pred[i]]:.2f}",
          snippet_or_path[i])
```

Read the worst examples by hand. Typical findings: mislabeled rows (fix the
dataset, retrain), one class systematically missed (class weights / more data),
confusions that follow a real sub-pattern (report it as a model-card limitation).
Calibration, if probabilities drive decisions: compare `probs` against accuracy
bins; a model at 70% acc but 95% confidence is overconfident — say so.

## 8. Logging and reproducibility hygiene

- Write the resolved config next to the run: dump `args.to_dict()`, the seed, model
  id, dataset revision/hash, and package versions into `config.yaml`.
- Log to file as well as stdout (`tee`), so a crashed run still has its history.
- `uv pip freeze > requirements.txt` (or the `uv` lock) inside the run directory.
- Set a wall-clock timeout for experiments so a runaway job cannot occupy the machine
  indefinitely (`references/monitoring-and-resume.md` §6).
