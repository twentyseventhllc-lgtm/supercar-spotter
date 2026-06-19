# Supercar Spotter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Watch a street video feed, find every car with YOLO, name the brand with zero-shot CLIP, and save one labeled photo per caught supercar.

**Architecture:** Two-stage, zero-shot, no training. YOLO detects + tracks cars (stock COCO `car` class). Each tracked car crop is scored by CLIP against a text word-list of supercar brands vs "ordinary car" negatives. A per-track deduper saves one best shot per car. Three small modules: pure decision logic (`brain.pick_best`), pure dedup (`catch.CatchTracker`), CLIP wrapper (`brain.BrandClassifier`), wired by `spotter.py`.

**Tech Stack:** Python, Ultralytics YOLO (yolo11n), open_clip_torch (ViT-B-32), OpenCV, pytest.

## Global Constraints

- Python 3.10+ (uses `X | None` type hints, dataclasses).
- No model training — zero-shot only.
- `brain.py` must NOT import torch / open_clip at module top level. Heavy imports go *inside* `BrandClassifier`, so importing `pick_best` / `Verdict` stays fast for tests.
- Default fast `pytest` run must NOT download any model. Model-dependent tests are marked `integration` and excluded by default.
- COCO car class index is `2`. Confidence/probability values are floats in `[0, 1]`.
- Friendly config-block-at-top style in `spotter.py`, matching the existing `detect.py`.

---

### Task 1: Project setup — deps, layout, import smoke test

**Files:**
- Create: `requirements.txt`
- Create: `conftest.py` (empty — makes repo root importable by pytest)
- Create: `pytest.ini`
- Create: `tests/test_smoke.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a working Python env; `tests/` runnable via `python -m pytest`.

- [ ] **Step 1: Write `requirements.txt`**

```
ultralytics>=8.3.0
open-clip-torch>=2.24.0
opencv-python>=4.9.0
numpy>=1.24
Pillow>=10.0
pytest>=8.0
```

- [ ] **Step 2: Create empty `conftest.py` at repo root**

```python
# Empty on purpose: presence makes pytest add the repo root to sys.path
# so tests can `import brain`, `import catch`, etc.
```

- [ ] **Step 3: Write `pytest.ini` (register marker, skip integration by default)**

```ini
[pytest]
markers =
    integration: tests that load the real CLIP model (downloads weights). Run with: python -m pytest -m integration
addopts = -m "not integration"
```

- [ ] **Step 4: Write the import smoke test `tests/test_smoke.py`**

```python
def test_core_libs_import():
    import ultralytics  # noqa: F401
    import open_clip    # noqa: F401
    import cv2          # noqa: F401
```

- [ ] **Step 5: Install deps**

Run: `python -m pip install -r requirements.txt`
Expected: completes (pulls torch transitively; large download is normal).

- [ ] **Step 6: Run smoke test**

Run: `python -m pytest tests/test_smoke.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt conftest.py pytest.ini tests/test_smoke.py
git commit -m "chore: project setup, deps, pytest config"
```

---

### Task 2: `pick_best` — the verdict decision (pure, TDD)

**Files:**
- Create: `brain.py`
- Create: `tests/test_pick_best.py`

**Interfaces:**
- Consumes: nothing heavy (no torch).
- Produces:
  - `Verdict` dataclass: `label: str`, `confidence: float`, `is_supercar: bool`.
  - `pick_best(scores: dict[str, float], brands: list[str], negatives: list[str], margin: float) -> Verdict`.
    Picks the highest-scoring brand label; `is_supercar` is True iff `(best_brand_prob - best_negative_prob) >= margin`. If no brand label is present in `scores`, returns `Verdict("", 0.0, False)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pick_best.py
from brain import pick_best, Verdict

BRANDS = ["Ferrari", "Lamborghini"]
NEGS = ["ordinary car", "sedan"]

def test_clear_supercar_is_caught():
    scores = {"Ferrari": 0.7, "Lamborghini": 0.05, "ordinary car": 0.1, "sedan": 0.05}
    v = pick_best(scores, BRANDS, NEGS, margin=0.15)
    assert v == Verdict(label="Ferrari", confidence=0.7, is_supercar=True)

def test_normie_car_not_caught():
    scores = {"Ferrari": 0.2, "ordinary car": 0.6}
    v = pick_best(scores, BRANDS, NEGS, margin=0.15)
    assert v.label == "Ferrari" and v.confidence == 0.2 and v.is_supercar is False

def test_within_margin_not_caught():
    scores = {"Lamborghini": 0.5, "ordinary car": 0.45}
    v = pick_best(scores, BRANDS, NEGS, margin=0.15)
    assert v.label == "Lamborghini" and v.is_supercar is False

def test_no_brand_in_scores():
    scores = {"ordinary car": 0.9}
    v = pick_best(scores, BRANDS, NEGS, margin=0.15)
    assert v == Verdict(label="", confidence=0.0, is_supercar=False)

def test_no_negatives_present_uses_zero_floor():
    scores = {"Ferrari": 0.3}
    v = pick_best(scores, BRANDS, NEGS, margin=0.15)
    assert v.is_supercar is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pick_best.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'brain'` (or `ImportError`).

- [ ] **Step 3: Write minimal implementation in `brain.py`**

```python
# brain.py
# NOTE: keep torch / open_clip imports OUT of module top level (see Task 4).
from dataclasses import dataclass


@dataclass
class Verdict:
    label: str
    confidence: float
    is_supercar: bool


def pick_best(scores, brands, negatives, margin):
    brand_scores = {label: scores[label] for label in brands if label in scores}
    if not brand_scores:
        return Verdict(label="", confidence=0.0, is_supercar=False)

    best_label = max(brand_scores, key=brand_scores.get)
    best_brand = brand_scores[best_label]

    neg_scores = [scores[label] for label in negatives if label in scores]
    best_neg = max(neg_scores) if neg_scores else 0.0

    is_supercar = (best_brand - best_neg) >= margin
    return Verdict(label=best_label, confidence=best_brand, is_supercar=is_supercar)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_pick_best.py -v`
Expected: all 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add brain.py tests/test_pick_best.py
git commit -m "feat: pick_best verdict logic (brand vs normie by margin)"
```

---

### Task 3: `CatchTracker` — one catch per car (pure, TDD)

**Files:**
- Create: `catch.py`
- Create: `tests/test_catch.py`

**Interfaces:**
- Consumes: `brain.Verdict`.
- Produces:
  - `CatchAction` dataclass: `track_id: int`, `label: str`, `confidence: float`.
  - `CatchTracker` class:
    - `update(track_id: int, verdict: Verdict) -> CatchAction | None` — returns a `CatchAction` when this track should be saved (first qualifying supercar frame, or a later higher-confidence frame for the same track); otherwise `None`. Non-supercar verdicts always return `None`.
    - `reset() -> None` — clears all per-track state (call between separate clips so track IDs don't collide).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_catch.py
from brain import Verdict
from catch import CatchTracker, CatchAction

def sc(conf):       # a supercar verdict
    return Verdict(label="Ferrari", confidence=conf, is_supercar=True)

def normie(conf):   # a non-supercar verdict
    return Verdict(label="Ferrari", confidence=conf, is_supercar=False)

def test_normie_never_catches():
    t = CatchTracker()
    assert t.update(1, normie(0.9)) is None

def test_first_supercar_frame_catches():
    t = CatchTracker()
    assert t.update(1, sc(0.6)) == CatchAction(track_id=1, label="Ferrari", confidence=0.6)

def test_higher_confidence_same_track_recatches():
    t = CatchTracker()
    t.update(1, sc(0.6))
    assert t.update(1, sc(0.8)) == CatchAction(track_id=1, label="Ferrari", confidence=0.8)

def test_lower_confidence_same_track_is_ignored():
    t = CatchTracker()
    t.update(1, sc(0.8))
    assert t.update(1, sc(0.5)) is None

def test_two_tracks_are_independent():
    t = CatchTracker()
    assert t.update(1, sc(0.6)) is not None
    assert t.update(2, sc(0.6)) is not None

def test_reset_clears_state():
    t = CatchTracker()
    t.update(1, sc(0.6))
    t.reset()
    assert t.update(1, sc(0.6)) is not None  # track 1 is fresh again
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_catch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'catch'`.

- [ ] **Step 3: Write minimal implementation in `catch.py`**

```python
# catch.py
from dataclasses import dataclass
from brain import Verdict


@dataclass
class CatchAction:
    track_id: int
    label: str
    confidence: float


class CatchTracker:
    def __init__(self):
        self._best: dict[int, float] = {}

    def update(self, track_id: int, verdict: Verdict) -> CatchAction | None:
        if not verdict.is_supercar:
            return None
        prev = self._best.get(track_id)
        if prev is not None and verdict.confidence <= prev:
            return None
        self._best[track_id] = verdict.confidence
        return CatchAction(track_id=track_id, label=verdict.label, confidence=verdict.confidence)

    def reset(self) -> None:
        self._best.clear()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_catch.py -v`
Expected: all 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add catch.py tests/test_catch.py
git commit -m "feat: CatchTracker dedup (one best shot per tracked car)"
```

---

### Task 4: `BrandClassifier` — the CLIP wrapper (integration-tested)

**Files:**
- Modify: `brain.py` (add `BrandClassifier`; keep heavy imports inside the class)
- Create: `tests/test_brand_classifier.py`

**Interfaces:**
- Consumes: `numpy`, `torch`, `open_clip`, `PIL`, `cv2` (all lazy-imported inside the class).
- Produces:
  - `BrandClassifier(labels: list[str], model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", template="a photo of {}")`.
  - `.score(crop_bgr) -> dict[str, float]` — takes an OpenCV BGR crop (numpy `HxWx3`), returns a dict mapping each label to a softmax probability. Keys equal `labels`; values sum to ~1.0.

- [ ] **Step 1: Write the failing (integration) test**

```python
# tests/test_brand_classifier.py
import numpy as np
import pytest

pytestmark = pytest.mark.integration  # downloads CLIP weights; run with -m integration

def test_score_returns_normalized_distribution_over_labels():
    from brain import BrandClassifier
    labels = ["a sports car", "a tree", "a sandwich"]
    clf = BrandClassifier(labels=labels, template="{}")
    crop = np.zeros((120, 200, 3), dtype=np.uint8)  # any valid BGR image
    scores = clf.score(crop)
    assert set(scores.keys()) == set(labels)
    assert all(0.0 <= v <= 1.0 for v in scores.values())
    assert abs(sum(scores.values()) - 1.0) < 1e-3
```

- [ ] **Step 2: Run the integration test to verify it fails**

Run: `python -m pytest tests/test_brand_classifier.py -m integration -v`
Expected: FAIL with `ImportError: cannot import name 'BrandClassifier' from 'brain'`.

- [ ] **Step 3: Add `BrandClassifier` to `brain.py`**

Append to `brain.py`:

```python
class BrandClassifier:
    """Zero-shot brand scorer. Lazy-imports heavy deps so `pick_best` stays light."""

    def __init__(self, labels, model_name="ViT-B-32",
                 pretrained="laion2b_s34b_b79k", template="a photo of {}"):
        import torch
        import open_clip

        self._torch = torch
        self.labels = list(labels)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()
        tokenizer = open_clip.get_tokenizer(model_name)

        prompts = [template.format(label) for label in self.labels]
        tokens = tokenizer(prompts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        self._text_features = text_features

    def score(self, crop_bgr):
        import cv2
        from PIL import Image

        torch = self._torch
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        image = self.preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ self._text_features.T
            probs = logits.softmax(dim=-1).squeeze(0).tolist()

        return {label: float(p) for label, p in zip(self.labels, probs)}
```

- [ ] **Step 4: Run the integration test to verify it passes**

Run: `python -m pytest tests/test_brand_classifier.py -m integration -v`
Expected: PASS (first run downloads the CLIP weights — slow, normal).

- [ ] **Step 5: Confirm the default fast suite still ignores it**

Run: `python -m pytest -v`
Expected: smoke + pick_best + catch tests run; the integration test is deselected.

- [ ] **Step 6: Commit**

```bash
git add brain.py tests/test_brand_classifier.py
git commit -m "feat: BrandClassifier zero-shot CLIP scorer"
```

---

### Task 5: `spotter.py` — wire it all together + run on a clip

**Files:**
- Create: `spotter.py`
- Create: `clips/` (gitignored; holds test footage)
- Create: `catches/` (gitignored; created at runtime if missing)

**Interfaces:**
- Consumes: `ultralytics.YOLO`, `cv2`, `brain.BrandClassifier`, `brain.pick_best`, `catch.CatchTracker`.
- Produces: a runnable script. On a catch, writes `catches/<ts>_track<id>_<label>.jpg` (annotated) and appends a row to `catches/log.csv`.

- [ ] **Step 1: Write `spotter.py`**

```python
# spotter.py — Supercar Spotter
# ─── config ───────────────────────────────────────────────────
SOURCE     = "clips/test.mp4"   # a video file, a folder of clips, or 0 for webcam
BRANDS     = ["Ferrari", "Lamborghini", "Porsche", "McLaren", "Lotus",
              "Aston Martin", "Bugatti", "Maserati", "Audi R8", "Corvette"]
NEGATIVES  = ["ordinary car", "sedan", "SUV", "van", "truck"]
MARGIN     = 0.15      # how much a brand must beat the best normie score to count
MIN_BOX_AREA = 4000    # px²; skip tiny far-away cars
CONF       = 0.30      # YOLO detection confidence floor
CAR_CLASSES = [2]      # COCO 'car'
OUT_DIR    = "catches"
SHOW       = True      # live preview window
# ──────────────────────────────────────────────────────────────

import csv
import glob
import os
import time

import cv2
from ultralytics import YOLO

from brain import BrandClassifier, pick_best
from catch import CatchTracker

LABELS = BRANDS + NEGATIVES
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def list_sources(source):
    if isinstance(source, int):
        return [source]
    if os.path.isdir(source):
        files = sorted(
            f for f in glob.glob(os.path.join(source, "*"))
            if f.lower().endswith(VIDEO_EXTS)
        )
        return files or []
    return [source]


def save_catch(out_dir, frame, action):
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    safe_label = action.label.replace(" ", "_")
    path = os.path.join(out_dir, f"{ts}_track{action.track_id}_{safe_label}.jpg")
    cv2.imwrite(path, frame)
    with open(os.path.join(out_dir, "log.csv"), "a", newline="") as fh:
        csv.writer(fh).writerow([ts, action.track_id, action.label,
                                 f"{action.confidence:.3f}", path])
    print(f"📸 caught {action.label} ({action.confidence:.0%}) → {path}")


def run():
    classifier = BrandClassifier(labels=LABELS)
    tracker = CatchTracker()

    for src in list_sources(SOURCE):
        tracker.reset()  # fresh track IDs per clip
        model = YOLO("yolo11n.pt")
        stream = model.track(source=src, classes=CAR_CLASSES, conf=CONF,
                             persist=True, stream=True, verbose=False)

        for result in stream:
            frame = result.orig_img.copy()
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                _show(frame)
                continue

            for box, tid in zip(boxes, boxes.id.int().tolist()):
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
                    continue
                crop = result.orig_img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                verdict = pick_best(classifier.score(crop), BRANDS, NEGATIVES, MARGIN)
                color = (0, 215, 255) if verdict.is_supercar else (120, 120, 120)
                label = (f"{verdict.label} {verdict.confidence:.0%}"
                         if verdict.is_supercar else "car")
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                action = tracker.update(tid, verdict)
                if action is not None:
                    save_catch(OUT_DIR, frame, action)

            if not _show(frame):
                return


def _show(frame):
    if not SHOW:
        return True
    cv2.imshow("Supercar Spotter", frame)
    return cv2.waitKey(1) & 0xFF != ord("q")  # False => quit


if __name__ == "__main__":
    run()
    cv2.destroyAllWindows()
```

- [ ] **Step 2: Get a test clip**

Place any short street/traffic video at `clips/test.mp4` (a supercar street clip is ideal; any traffic clip proves the pipeline). If you have none yet, a clip with normal cars is fine to confirm it runs and correctly *ignores* normies.

Run: `ls clips/test.mp4`
Expected: the file exists.

- [ ] **Step 3: Run the spotter on the clip**

Run: `python spotter.py`
Expected: a window opens showing the clip with boxes; normal cars are grey/`car`, exotics get a colored brand label; `catches/` fills with a `.jpg` per caught supercar and a `log.csv`. Press `q` to quit.

- [ ] **Step 4: Sanity-check the catches**

Run: `ls catches/ && cat catches/log.csv`
Expected: snapshots only for exotic-looking cars; one file per distinct car (not dozens of the same one). If too many false catches, raise `MARGIN`; if it misses obvious exotics, lower `MARGIN`.

- [ ] **Step 5: Commit**

```bash
git add spotter.py
git commit -m "feat: spotter main loop (YOLO track + CLIP + catch saving)"
```

---

## Notes for the implementer

- **Going live:** once the clips look good, set `SOURCE = 0` in `spotter.py` to run on the webcam pointed out the window.
- **Tuning knobs:** `MARGIN` (strictness), `MIN_BOX_AREA` (ignore distant cars), `BRANDS`/`NEGATIVES` (vocabulary).
- **Performance:** if the live window lags with several cars, the cheap fix is to classify a track only every Nth frame; not needed for v1 light street traffic.
- **Phase 2 (not in this plan):** Telegram ping with the caught photo; custom-trained model for sharper brand accuracy.
