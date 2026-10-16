# Dataset Playbook — acquire, split, protect, report

Data mistakes cannot be fixed by a better model. Do this **before** any training.

## 1. Acquisition

| Source | How | Notes |
|---|---|---|
| Hugging Face | `load_dataset("<org>/<name>")` | Check the dataset card: license, size, label definition, intended use |
| Local CSV / JSONL | `load_dataset("csv", data_files=...)` | Convert to HF dataset early for consistent splitting |
| Images on disk | `torchvision.datasets.ImageFolder(root)` | Layout: `root/<class>/*.jpg`; validate no empty class dirs |
| Audio on disk | `torchaudio.datasets` or custom `Dataset` | Store sample rate + duration; resample on load to a fixed rate |
| Torchvision built-ins | `MNIST`, `CIFAR10`, ... | Fine for smoke tests; **not** representative of real tasks |

Rules:

- **Never** fetch a dataset whose license you have not read. For "gated" repos,
  authenticate (`hf auth login` / `HF_TOKEN`) and confirm the user may use it.
- Record the exact dataset id + revision/hash in `config.yaml` so the run can be
  reproduced when the upstream dataset later changes.
- Sample 20–50 random items and **look at them** (print text, play audio, view
  images) before training. Catches label-swapped columns and corrupt files early.

## 2. Splits

Always three splits: **train** (fit), **val** (all decisions), **test** (once, at
the end). Typical ratios: 70/15/15 or 80/10/10.

```python
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=42)
```

**Group-aware split** (audio speakers, patients, devices, writers) — mandatory when
multiple rows share an entity. The model must never see a speaker in both train
and test:

```python
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, te_idx = next(gss.split(X, y, groups=speaker_ids))   # groups = one id per row
```

**Time-aware split** (anything temporal): sort by timestamp, cut by time — never
shuffle across time. Future data must not leak into training for forecasting.

## 3. Leakage checks (run these; they catch the classic silent wins)

| Check | How | What it catches |
|---|---|---|
| Exact duplicates across splits | Hash rows (`hashlib` on file bytes / normalized text) and intersect train vs test hash sets | Copy-paste leakage; the same file in both splits |
| Near-duplicates | `difflib.SequenceMatcher` ratio > 0.9 on text; embedding cosine > 0.95 on embeddings; perceptual hash on images | Slightly edited copies |
| Group straddling | `set(train_groups) & set(test_groups)` must be empty | Same speaker/patient/device on both sides |
| Target leakage | List every feature; ask "could this feature only exist after the label is known?" | Fraud timestamps, post-outcome fields |
| Preprocessing fit on all data | `StandardScaler`/`PCA`/vocabulary must be fit on **train only** | Subtle statistical leakage inflating metrics |

If a leakage check finds hits: drop the offending rows from the minority split,
re-split with groups, and **re-report**. A metric computed on leaked data is void.

## 4. Class balance and metrics

- Print the label histogram for each split. If one class > 90%, accuracy is
  meaningless.
- Use **macro-F1** (or AUROC for binary) as the primary metric; never accuracy
  alone on imbalanced data.
- Remedies, in order: class weights (`CrossEntropyLoss(weight=...)`), resampling
  (train only — never resample val/test), focal loss, or threshold tuning on val.
- Keep val/test at their natural distribution so the reported number reflects
  reality.

## 5. Integrity checks

```python
# label range
assert y.min() >= 0 and y.max() < num_labels, "label out of range -> device-side assert"

# NaN / inf in features
assert not torch.isnan(X).any() and not torch.isinf(X).any(), "NaN/inf in features"

# audio sanity
sr = torchaudio.load(path)[1]
assert sr == target_sr, f"sample rate mismatch: {sr} != {target_sr}"
```

- Filter zero-length / corrupted files during dataset build, not during training
  (a corrupt file mid-run kills the job at hour 3).
- For audio: convert to mono, resample to one rate (e.g. 16 kHz), cap duration
  (e.g. 10 s) — mismatched rates cause silent garbage features.
- For text: strip empty strings; dedupe; record language if mixed.

## 6. Provenance to record (into `config.yaml` / `MODEL_CARD.md`)

- Source: dataset id + revision, or file paths + collection method
- License and any access restrictions
- Size: rows/bytes per split; label histogram
- Split method: ratios, seed, group/time key used
- Preprocessing: resampling, normalization constants, tokenizer id + version
- Known issues: imbalance, missing labels, suspected noise

## 7. Memory pragmatics on a 16 GB laptop

- Do not `list(dataset)` a multi-GB corpus into RAM. Stream/map lazily
  (`load_dataset(..., streaming=True)` for huge sets; `.map(..., batched=True)`
  with `keep_in_memory=False`).
- ImageFolder: decode on `__getitem__`, not at init.
- Audio: store paths, not waveforms; load + resample in the worker.
- Cache location: `HF_HOME` on the disk with space; check `df -h` before a big
  download.
