# Trained Brand Classifier + Temporal Consistency — Design

**Date:** 2026-06-19
**Status:** Approved (design), pending implementation plan

## Goal

Cut the false-alarm flood in the live spotter. A real run produced dozens of
bogus catches in 40 minutes ("Porsche" every few seconds, a "Bugatti", and the
**same track caught as both Ferrari and Aston Martin**). Two complementary fixes:

1. **Phase 1 — Temporal consistency + higher floor** (no data, immediate relief).
2. **Phase 2 — A trained CLIP-feature classifier** (web-downloaded data), deployed
   only if it measurably beats the current zero-shot scorer.

Goal metric: **precision** — far fewer false catches, accepting the occasional miss.

## Problem evidence (from the live run)

- Brand misfires: ordinary cars scored as supercars (esp. "Porsche").
- **Temporal flicker:** one car flips brand frame-to-frame — a trained model alone
  won't fix this.
- Stream stutter (`Waiting for stream 0`) fragments one car into many short track
  ids, each firing its own catch — multiplying the flood.

## Decisions (locked during brainstorming)

- Tier **C**: train a real model (not just tuning).
- Training data: **download from the web** (~150–250 images/class).
- Model: **CLIP-feature classifier** — a small head (logistic regression) on top of
  the frozen CLIP image embedding we already compute on the GPU.
- **Plus** temporal consistency, because flicker is a big part of the problem.

---

## Phase 1 — Temporal consistency + higher floor (do first)

**Temporal consistency.** A tracked car must be classified as the *same* supercar
brand on **K consecutive fresh classifications** (default `CONFIRM_STREAK = 3`)
before it counts as a catch. A non-supercar classification, or a *different* brand,
**resets** the streak. "Fresh" means an actual CLIP run — cached verdicts (the
classify-every-N-frames throttle) do NOT advance the streak, so the count reflects
real re-checks, not repeated frames.

- Lives in `catch.py`: `CatchTracker` gains a `confirm_streak` parameter and
  per-track streak state. Its `update(...)` returns a `CatchAction` only once a
  track's streak reaches the threshold (then the existing best-shot/improvement
  logic applies). Pure and unit-testable.

**Higher confidence floor.** Raise `MIN_CONFIDENCE` default `0.70 → 0.85` so weak
guesses don't even enter the streak. Editable in the config block.

This phase needs no data and ships immediately. Expected to remove the large
majority of the flood on its own.

---

## Phase 2 — Trained CLIP-feature classifier

**Key idea:** keep CLIP as the frozen "eye"; train a small classifier head on CLIP
embeddings of web photos. Data-efficient, robust to messy images, retrains in
seconds, reuses the existing GPU setup.

### Pipeline
```
1. DOWNLOAD    brands + "ordinary car"  ->  data/<class>/*.jpg   (~150-250 each)
2. EMBED       each image -> CLIP vector (cached to disk)
3. TRAIN+EVAL  logistic-regression head; stratified hold-out test set;
               COMPARE accuracy head-to-head vs the current zero-shot scorer
4. INTEGRATE   TrainedBrandClassifier(crop) -> (label, confidence), behind a
               USE_TRAINED toggle, feeding the SAME temporal-consistency gate
```

### Components (small, isolated, testable)

- **`clip_encoder.py` → `CLIPEncoder`** — loads CLIP once (CUDA/MPS/CPU, same
  fallback as today), `.embed(crop_bgr) -> float vector`. Refactored *out* of the
  current `BrandClassifier`, so the zero-shot scorer, the training step, and the
  trained classifier all share one CLIP. `BrandClassifier` becomes `CLIPEncoder` +
  text-prompt features (behavior unchanged).

- **`download_data.py`** — for each class, pull ~N images from the web (via
  `icrawler`'s Bing crawler) into `data/<class>/`. Prints a count per class and a
  reminder to **manually delete junk** before training. Re-runnable/resumable.

- **`train_classifier.py`** — embeds every image (caches vectors to
  `data/embeddings.npz` keyed by file, so re-runs are instant), does a stratified
  train/test split, trains a scikit-learn `LogisticRegression` head, and prints:
  overall accuracy, per-class precision/recall, a confusion matrix, **and the
  zero-shot baseline's accuracy on the same held-out images** (the gate). Saves
  `models/brand_head.joblib` (head + class names + CLIP config) — small, committed.

- **`trained_brain.py` → `TrainedBrandClassifier`** — `CLIPEncoder` + the loaded
  head → `(label, confidence)` via `predict_proba`. A decision helper mirrors the
  existing `pick_best`/`Verdict`: `is_supercar` iff argmax is a brand (not
  "ordinary car") and its probability clears `MIN_CONFIDENCE`. Drop-in for the
  spotter/dashboard, and it feeds the same `CatchTracker` temporal gate.

- **Integration** — `USE_TRAINED = True/False` in the spotter/dashboard config.
  On → `TrainedBrandClassifier`; off → today's zero-shot `BrandClassifier`. Lets
  you A/B them live.

### Classes
Supercar brands (Ferrari, Lamborghini, Porsche, McLaren, Aston Martin, Audi R8,
Corvette, Maserati — editable) **plus an `ordinary car` class** (downloaded from
common cars: Corolla, Civic, generic sedan/SUV/hatchback). The `ordinary car`
class is what lets the head confidently say "not a supercar" — the crux of
precision. Ultra-rare brands (Bugatti, Koenigsegg) are excluded by default to
avoid near-guaranteed false positives.

### The accuracy gate (non-negotiable)
`train_classifier.py` reports trained accuracy vs zero-shot accuracy on the same
held-out photos. If trained doesn't win, keep zero-shot (`USE_TRAINED = False`) and
iterate on data. The decision is made from the printed numbers, not on faith.

## Testing strategy

- **Temporal consistency** (`catch.py`): unit tests — a track needs K agreeing
  fresh verdicts to fire; a different brand or a non-supercar resets the streak;
  cached repeats don't advance it; two tracks are independent; `reset()` clears.
- **Trained decision logic**: unit-test the head-probabilities → `(label,
  is_supercar, confidence)` helper (ordinary wins → not a catch; brand over floor →
  catch) with synthetic probability vectors — no model load.
- **CLIPEncoder / TrainedBrandClassifier**: one integration test (marked, may load
  the model) asserting `.embed` shape and that inference returns a normalized
  distribution over the classes.
- **download_data / train_classifier**: I/O and ML — smoke-tested (small run),
  not unit-tested. The accuracy-gate comparison is the real check.

## Dependencies & git
- New: `icrawler` (image download), `scikit-learn` (the head), `joblib` (save).
- `data/` (images + `embeddings.npz`) is **gitignored**; `models/brand_head.joblib`
  is small (~KB) and **committed**.

## Risks
- Web images are noisy → the manual cleanup step + the accuracy gate are the
  guardrails.
- A linear head may not beat CLIP zero-shot if data is too thin → the gate catches
  that; fallback is more/better data, a small MLP head, or staying zero-shot.
- `icrawler` web scraping is brittle (rate limits, junk) → re-runnable, and counts
  are printed so shortfalls are visible.

## Out of scope (later, if ever)
- Fine-tuning CLIP itself or a from-scratch YOLO detector.
- Harvesting training data from the user's own camera (possible Phase 3 refinement).
- Fixing the iPhone Continuity stream stutter (separate camera issue).
