# Two-Tier Catching + All-Cars Safety Net — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Catch far more cars by bucketing into identified / unidentified tiers (instead of "confident brand or nothing"), plus an optional all-cars safety net so nothing is missed.

**Architecture:** `pick_best` now reports `is_exotic` (beats ordinary by margin — the catch trigger) and `is_identified` (also clears the name floor). `CatchTracker` streaks on `is_exotic` and tags each catch with a `tier`. The spotter/dashboard draw gold (identified) / cyan (unidentified) / grey (car), save by tier (`catches/` for cool catches, `catches/all/` for the firehose), and push only the cool tiers.

**Tech Stack:** Python, existing brain/catch/spotter/dashboard/notify modules, pytest.

## Global Constraints

- Python 3.10+ (`X | None`, dataclasses). Run via `.venv/bin/python -m pytest`.
- `Verdict` fields become `label, confidence, is_exotic, is_identified` (the old `is_supercar` is removed). `pick_best`'s floor param is renamed `identify_floor`.
- `CatchAction` gains `tier: str` (`"identified"` or `"unidentified"`).
- Catch trigger = `is_exotic` for `CONFIRM_STREAK` consecutive FRESH checks (not same-brand). Tier = `identified` if any check on the track was `is_identified`, else `unidentified`.
- Config: `IDENTIFY_FLOOR` (replaces `MIN_CONFIDENCE`), `MARGIN`, `CONFIRM_STREAK=2`, `SAVE_ALL_CARS=True`. `CONFIRM_STREAK`/`IDENTIFY_FLOOR` single-sourced in `spotter.py`; `dashboard.py` imports them.
- Output: identified → `catches/<ts>_track<id>_<Brand>.jpg`; unidentified → `catches/<ts>_track<id>_unidentified.jpg`; all-cars → `catches/all/<ts>_track<id>_car.jpg` (the crop). `log.csv` row = `ts, track_id, tier, label, confidence, path`.
- Colors (dashboard already defines them): GOLD identified, CYAN unidentified, GREY car.
- Notifications: identified + unidentified push (60s cooldown); all-cars silent.

---

### Task 1: `pick_best` reports `is_exotic` + `is_identified`

**Files:**
- Modify: `brain.py` (`Verdict`, `pick_best`)
- Test: `tests/test_pick_best.py` (rewrite)

**Interfaces:**
- Produces:
  - `Verdict(label: str, confidence: float, is_exotic: bool, is_identified: bool)`.
  - `pick_best(scores, brands, negatives, margin, identify_floor=0.0) -> Verdict` —
    `is_exotic = (best_brand - best_negative) >= margin`; `is_identified = is_exotic and best_brand >= identify_floor`. No brand present → `Verdict("", 0.0, False, False)`.

- [ ] **Step 1: Rewrite `tests/test_pick_best.py`**

```python
from brain import pick_best, Verdict

BRANDS = ["Ferrari", "Lamborghini"]
NEGS = ["ordinary car", "sedan"]

def test_confident_brand_is_identified():
    scores = {"Ferrari": 0.8, "ordinary car": 0.1}
    v = pick_best(scores, BRANDS, NEGS, margin=0.15, identify_floor=0.70)
    assert v == Verdict(label="Ferrari", confidence=0.8, is_exotic=True, is_identified=True)

def test_exotic_but_below_floor_is_unidentified():
    # Looks clearly more exotic than ordinary, but not confident enough to name.
    scores = {"Ferrari": 0.55, "ordinary car": 0.1}
    v = pick_best(scores, BRANDS, NEGS, margin=0.15, identify_floor=0.70)
    assert v.is_exotic is True and v.is_identified is False and v.label == "Ferrari"

def test_ordinary_is_neither():
    scores = {"Ferrari": 0.2, "ordinary car": 0.6}
    v = pick_best(scores, BRANDS, NEGS, margin=0.15, identify_floor=0.70)
    assert v.is_exotic is False and v.is_identified is False

def test_within_margin_not_exotic():
    scores = {"Lamborghini": 0.5, "ordinary car": 0.45}
    v = pick_best(scores, BRANDS, NEGS, margin=0.15, identify_floor=0.70)
    assert v.is_exotic is False

def test_no_brand_in_scores():
    v = pick_best({"ordinary car": 0.9}, BRANDS, NEGS, margin=0.15, identify_floor=0.70)
    assert v == Verdict(label="", confidence=0.0, is_exotic=False, is_identified=False)

def test_default_floor_zero_identifies_any_exotic():
    scores = {"Ferrari": 0.3, "ordinary car": 0.05}
    v = pick_best(scores, BRANDS, NEGS, margin=0.15)
    assert v.is_exotic is True and v.is_identified is True
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_pick_best.py -q`
Expected: FAIL (`Verdict` has no `is_exotic` / unexpected `is_supercar`).

- [ ] **Step 3: Rewrite `Verdict` + `pick_best` in `brain.py`**

Replace the `Verdict` dataclass and `pick_best` function with:

```python
@dataclass
class Verdict:
    label: str
    confidence: float
    is_exotic: bool       # beats the best "ordinary car" score by `margin`
    is_identified: bool   # is_exotic AND confidence >= identify_floor


def pick_best(scores, brands, negatives, margin, identify_floor=0.0):
    brand_scores = {label: scores[label] for label in brands if label in scores}
    if not brand_scores:
        return Verdict(label="", confidence=0.0, is_exotic=False, is_identified=False)

    best_label = max(brand_scores, key=brand_scores.get)
    best_brand = brand_scores[best_label]

    neg_scores = [scores[label] for label in negatives if label in scores]
    best_neg = max(neg_scores) if neg_scores else 0.0

    is_exotic = (best_brand - best_neg) >= margin
    is_identified = is_exotic and best_brand >= identify_floor
    return Verdict(label=best_label, confidence=best_brand,
                   is_exotic=is_exotic, is_identified=is_identified)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_pick_best.py -q`
Expected: all 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add brain.py tests/test_pick_best.py
git commit -m "feat: pick_best reports is_exotic + is_identified (two-tier)"
```

---

### Task 2: `CatchTracker` streaks on exotic + tags a tier

**Files:**
- Modify: `catch.py` (`CatchAction`, `CatchTracker`)
- Test: `tests/test_catch.py` (rewrite)

**Interfaces:**
- Consumes: `brain.Verdict` (`label, confidence, is_exotic, is_identified`).
- Produces:
  - `CatchAction(track_id: int, label: str, confidence: float, tier: str)`.
  - `CatchTracker(confirm_streak: int = 1)`, `update(track_id, verdict, fresh=True) -> CatchAction | None`, `reset()`.
  - Behavior: streak counts consecutive FRESH `is_exotic` verdicts (a non-exotic fresh verdict resets to 0). At/after `confirm_streak`, emit. Tier = `"identified"` if any check was `is_identified` (label = that brand, confidence = its best), else `"unidentified"` (label = `"unidentified"`, confidence = best exotic confidence). Re-emit on a strictly-better shot or an unidentified→identified upgrade.

- [ ] **Step 1: Rewrite `tests/test_catch.py`**

```python
from brain import Verdict
from catch import CatchTracker, CatchAction

def ex(conf, ident=False, label="Ferrari"):    # an exotic verdict
    return Verdict(label=label, confidence=conf, is_exotic=True, is_identified=ident)

def ordinary(conf=0.1):                          # a non-exotic verdict
    return Verdict(label="Ferrari", confidence=conf, is_exotic=False, is_identified=False)

def test_exotic_streak_required():
    t = CatchTracker(confirm_streak=2)
    assert t.update(1, ex(0.5)) is None
    assert t.update(1, ex(0.5)) == CatchAction(track_id=1, label="unidentified",
                                               confidence=0.5, tier="unidentified")

def test_identified_when_confident():
    t = CatchTracker(confirm_streak=2)
    assert t.update(1, ex(0.9, ident=True)) is None
    a = t.update(1, ex(0.9, ident=True))
    assert a.tier == "identified" and a.label == "Ferrari" and a.confidence == 0.9

def test_unidentified_when_exotic_but_not_named():
    t = CatchTracker(confirm_streak=1)
    a = t.update(1, ex(0.5, ident=False))
    assert a.tier == "unidentified" and a.label == "unidentified"

def test_non_exotic_resets_streak():
    t = CatchTracker(confirm_streak=2)
    assert t.update(1, ex(0.6)) is None            # streak 1
    assert t.update(1, ordinary()) is None          # reset
    assert t.update(1, ex(0.6)) is None             # streak 1 again, not caught

def test_unidentified_upgrades_to_identified():
    t = CatchTracker(confirm_streak=1)
    first = t.update(1, ex(0.5, ident=False))
    assert first.tier == "unidentified"
    upgraded = t.update(1, ex(0.8, ident=True))
    assert upgraded is not None and upgraded.tier == "identified" and upgraded.label == "Ferrari"

def test_cached_verdict_does_not_advance_streak():
    t = CatchTracker(confirm_streak=2)
    assert t.update(1, ex(0.6), fresh=True) is None
    assert t.update(1, ex(0.6), fresh=False) is None   # cached, no advance
    assert t.update(1, ex(0.6), fresh=True) is not None # 2nd fresh -> catch

def test_two_tracks_independent():
    t = CatchTracker(confirm_streak=1)
    assert t.update(1, ex(0.5)) is not None
    assert t.update(2, ex(0.5)) is not None

def test_reset_clears_state():
    t = CatchTracker(confirm_streak=1)
    t.update(1, ex(0.5))
    t.reset()
    assert t.update(1, ex(0.5)) is not None
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_catch.py -q`
Expected: FAIL (`CatchAction` has no `tier`; old streak logic mismatch).

- [ ] **Step 3: Rewrite `catch.py`**

```python
from dataclasses import dataclass
from brain import Verdict


@dataclass
class CatchAction:
    track_id: int
    label: str
    confidence: float
    tier: str   # "identified" or "unidentified"


class CatchTracker:
    def __init__(self, confirm_streak: int = 1):
        self.confirm_streak = confirm_streak
        self._tracks: dict[int, dict] = {}

    def update(self, track_id: int, verdict: Verdict,
               fresh: bool = True) -> CatchAction | None:
        st = self._tracks.get(track_id)
        if st is None:
            st = {"count": 0, "best_conf": 0.0, "brand": None, "brand_conf": 0.0,
                  "emitted_conf": None, "emitted_tier": None}
            self._tracks[track_id] = st

        if fresh:
            if verdict.is_exotic:
                st["count"] += 1
                st["best_conf"] = max(st["best_conf"], verdict.confidence)
                if verdict.is_identified and verdict.confidence > st["brand_conf"]:
                    st["brand"] = verdict.label
                    st["brand_conf"] = verdict.confidence
            else:
                st["count"] = 0

        if st["count"] < self.confirm_streak:
            return None

        if st["brand"] is not None:
            tier, label, conf = "identified", st["brand"], st["brand_conf"]
        else:
            tier, label, conf = "unidentified", "unidentified", st["best_conf"]

        upgraded = st["emitted_tier"] == "unidentified" and tier == "identified"
        if st["emitted_conf"] is not None and conf <= st["emitted_conf"] and not upgraded:
            return None
        st["emitted_conf"] = conf
        st["emitted_tier"] = tier
        return CatchAction(track_id=track_id, label=label, confidence=conf, tier=tier)

    def reset(self) -> None:
        self._tracks.clear()
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_catch.py -q`
Expected: all 8 PASS.

- [ ] **Step 5: Commit**

```bash
git add catch.py tests/test_catch.py
git commit -m "feat: CatchTracker streaks on exotic + tags identified/unidentified tier"
```

---

### Task 3: Output helpers (`save_catch` tier + `save_all_car`) and notify title

**Files:**
- Modify: `spotter.py` (`save_catch`, add `save_all_car`)
- Modify: `notify.py` (`notify_catch` title by tier)
- Test: `tests/test_spotter.py`, `tests/test_notify.py`

**Interfaces:**
- Consumes: `catch.CatchAction` (now with `.tier`).
- Produces:
  - `save_catch(out_dir, frame, action) -> path` — file `<ts>_track<id>_<safe_label>.jpg` in `out_dir`; log row `ts, track_id, tier, label, confidence:.3f, path`.
  - `save_all_car(out_dir, crop, track_id) -> path` — file `<ts>_track<id>_car.jpg` in `out_dir/all/`. No log.
  - `notify_catch(action, image_path)` title = `"<brand> <conf>%"` if identified, else `"Unidentified exotic"`.

- [ ] **Step 1: Update the save tests in `tests/test_spotter.py`**

Replace the two `save_catch` tests and add a `save_all_car` test:

```python
def test_save_catch_names_file_and_logs_tier(tmp_path):
    import numpy as np
    from catch import CatchAction
    import spotter
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    action = CatchAction(track_id=7, label="Ferrari", confidence=0.83, tier="identified")
    spotter.save_catch(str(tmp_path), frame, action)
    jpgs = list(tmp_path.glob("*.jpg"))
    assert len(jpgs) == 1 and "track7" in jpgs[0].name and "Ferrari" in jpgs[0].name
    row = (tmp_path / "log.csv").read_text().strip().split(",")
    assert "identified" in row and "Ferrari" in row and "7" in row

def test_save_all_car_writes_to_all_subfolder(tmp_path):
    import numpy as np
    import spotter
    crop = np.zeros((20, 30, 3), dtype=np.uint8)
    path = spotter.save_all_car(str(tmp_path), crop, track_id=12)
    assert (tmp_path / "all").is_dir()
    assert path.endswith(".jpg") and "track12" in path and "_car" in path
    assert len(list((tmp_path / "all").glob("*.jpg"))) == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_spotter.py -q`
Expected: FAIL (`CatchAction` needs `tier` already exists from Task 2; `save_all_car` undefined; log lacks tier).

- [ ] **Step 3: Update `save_catch` and add `save_all_car` in `spotter.py`**

Replace `save_catch` and add `save_all_car` right after it:

```python
def save_catch(out_dir, frame, action):
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    safe_label = action.label.replace(" ", "_")
    path = os.path.join(out_dir, f"{ts}_track{action.track_id}_{safe_label}.jpg")
    cv2.imwrite(path, frame)
    with open(os.path.join(out_dir, "log.csv"), "a", newline="") as fh:
        csv.writer(fh).writerow([ts, action.track_id, action.tier, action.label,
                                 f"{action.confidence:.3f}", path])
    print(f"📸 caught [{action.tier}] {action.label} ({action.confidence:.0%}) → {path}")
    return path


def save_all_car(out_dir, crop, track_id):
    all_dir = os.path.join(out_dir, "all")
    os.makedirs(all_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(all_dir, f"{ts}_track{track_id}_car.jpg")
    cv2.imwrite(path, crop)
    return path
```

- [ ] **Step 4: Update notify title in `notify.py`**

In `notify_catch`, replace the line `title = f"{action.label} {action.confidence:.0%}"` with:

```python
    if action.tier == "identified":
        title = f"{action.label} {action.confidence:.0%}"
    else:
        title = "Unidentified exotic"
```

- [ ] **Step 5: Update notify tests in `tests/test_notify.py`**

Every `CatchAction(...)` in `tests/test_notify.py` must now include a `tier=` argument. Add `tier="identified"` to each existing `CatchAction(...)` call (e.g. `CatchAction(track_id=7, label="Lamborghini", confidence=0.94, tier="identified")`). The assertions on `sent` titles stay valid because identified titles are unchanged.

- [ ] **Step 6: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_spotter.py tests/test_notify.py -q`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add spotter.py notify.py tests/test_spotter.py tests/test_notify.py
git commit -m "feat: save_catch logs tier, add save_all_car, notify title by tier"
```

---

### Task 4: Wire the tiers + all-cars net into `spotter.py`

**Files:**
- Modify: `spotter.py` (config + `run()`)

**Interfaces:**
- Consumes: `pick_best(..., IDENTIFY_FLOOR)`, `Verdict.is_identified/is_exotic`, `save_catch`, `save_all_car`, `CatchTracker(confirm_streak=CONFIRM_STREAK)`.

- [ ] **Step 1: Update the `spotter.py` config block**

Change the `MIN_CONFIDENCE` line and add `SAVE_ALL_CARS` (keep `CONFIRM_STREAK = 2`):

```python
IDENTIFY_FLOOR = 0.75  # confidence to NAME a brand; below this an exotic car is "unidentified"
CONFIRM_STREAK = 2     # consecutive exotic checks before a car counts (set 1 = most lenient)
SAVE_ALL_CARS = True   # also save every car to catches/all/ as a safety net
```

(Delete the old `MIN_CONFIDENCE = ...` line.)

- [ ] **Step 2: Update the supercars loop in `spotter.run()`**

Find the per-box body (from `verdict = pick_best(...)` to the `save_catch` call) and replace it with:

```python
                verdict = pick_best(classifier.score(crop), BRANDS, NEGATIVES,
                                    MARGIN, IDENTIFY_FLOOR)
                if verdict.is_identified:
                    color = (0, 215, 255)
                    label = f"{verdict.label} {verdict.confidence:.0%}"
                elif verdict.is_exotic:
                    color = (255, 200, 0)
                    label = "exotic?"
                else:
                    color = (120, 120, 120)
                    label = "car"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if SAVE_ALL_CARS and tid not in all_seen:
                    all_seen.add(tid)
                    save_all_car(OUT_DIR, crop, tid)

                action = tracker.update(tid, verdict)
                if action is not None:
                    save_catch(OUT_DIR, frame, action)
```

- [ ] **Step 3: Add the `all_seen` set in `run()`**

In `spotter.run()`, just after `tracker = CatchTracker(confirm_streak=CONFIRM_STREAK)`, add:

```python
    all_seen = set()
```

And inside the per-clip loop, just after `tracker.reset()`, add:

```python
        all_seen.clear()
```

- [ ] **Step 4: Verify**

Run: `.venv/bin/python -m pytest -q` → all pass.
Run the clip headless to confirm catches + tiers + the all/ folder fill:
```bash
rm -rf catches; .venv/bin/python -c "import spotter; spotter.SHOW=False; spotter.SOURCE='clips/test.mp4'; spotter.run()" 2>&1 | grep -E "caught" | head
ls catches/ catches/all/ 2>/dev/null | head
```
Expected: `[identified]`/`[unidentified]` catch lines; `catches/all/` contains `_car.jpg` files; `catches/` contains brand + `unidentified` jpgs. If nothing prints, note it for the reviewer (don't change thresholds without flagging).

- [ ] **Step 5: Commit**

```bash
git add spotter.py
git commit -m "feat: spotter draws/saves two tiers + all-cars safety net"
```

---

### Task 5: Wire the tiers + all-cars net + HUD split into `dashboard.py`

**Files:**
- Modify: `dashboard.py` (imports/config, `_process_frame`, `_make_hud`, `_make_gallery`, `_compose`, `run` state)

**Interfaces:**
- Consumes: everything from Tasks 1-4, plus `IDENTIFY_FLOOR`, `CONFIRM_STREAK`, `SAVE_ALL_CARS` imported from `spotter`.

- [ ] **Step 1: Update imports + config**

Change the `from spotter import (...)` line to also import the new names, and add `save_all_car`:

```python
from spotter import (BRANDS, NEGATIVES, MARGIN, IDENTIFY_FLOOR, CONFIRM_STREAK,
                     SAVE_ALL_CARS, MIN_BOX_AREA, CONF, OUT_DIR,
                     save_catch, save_all_car, list_sources)
```

(`MIN_CONFIDENCE` no longer exists — make sure it is not imported or referenced anywhere in `dashboard.py`.)

- [ ] **Step 2: Update the supercars branch in `_process_frame`**

Replace the block from `if verdict.is_supercar:` through the catch handling with:

```python
        if verdict.is_identified:
            _draw_box(annotated, x1, y1, x2, y2,
                      f"{verdict.label} {verdict.confidence:.0%}", GOLD)
        elif verdict.is_exotic:
            _draw_box(annotated, x1, y1, x2, y2, "exotic?", CYAN)
        else:
            _draw_box(annotated, x1, y1, x2, y2, "car", GREY)

        if SAVE_ALL_CARS and tid not in state["all_seen"]:
            state["all_seen"].add(tid)
            save_all_car(OUT_DIR, crop, tid)

        action = tracker.update(tid, verdict, fresh=fresh)
        if action is not None:
            path = save_catch(OUT_DIR, annotated, action)
            bucket = "ident_ids" if action.tier == "identified" else "unident_ids"
            state[bucket].add(tid)
            state["last"] = action.label
            state["thumbs"].append((crop.copy(), action.label, action.tier))
            state["thumbs"] = state["thumbs"][-12:]
            notify_catch(action, path)
```

Also update the `pick_best` call just above (in the same branch) to use `IDENTIFY_FLOOR`:

```python
            verdict = pick_best(classifier.score(crop), BRANDS, NEGATIVES,
                                MARGIN, IDENTIFY_FLOOR)
```

- [ ] **Step 3: Update `run()` state**

Replace the `state = {...}` initialiser in `run()` with:

```python
    state = {"ident_ids": set(), "unident_ids": set(), "thumbs": [], "last": None,
             "frame_no": 0, "verdict_cache": {}, "all_seen": set()}
```

And in the per-source loop, where `state["verdict_cache"].clear()` runs, also clear the per-clip sets:

```python
        state["verdict_cache"].clear()
        state["all_seen"].clear()
```

And update the `info` dict built each frame (replace the `"catches"` key):

```python
            info = {"mode": mode,
                    "identified": len(state["ident_ids"]),
                    "unidentified": len(state["unident_ids"]),
                    "fps": fps, "last": state["last"]}
```

- [ ] **Step 4: Update `_make_hud` to show the split**

Replace the `stats = (...)` line in `_make_hud` with:

```python
    stats = (f"MODE {info['mode'].upper()}     IDENTIFIED {info['identified']}"
             f"     UNKNOWN {info['unidentified']}     FPS {info['fps']:.0f}"
             f"     LAST {info['last'] or '-'}")
```

- [ ] **Step 5: Update `_make_gallery` to colour by tier**

The gallery now receives `(thumb, label, tier)` tuples. Replace the loop body in `_make_gallery`:

```python
    for thumb, label, tier in thumbs[::-1]:        # newest first, left to right
        if x + THUMB_W > width:
            break
        color = GOLD if tier == "identified" else CYAN
        cell = cv2.resize(thumb, (THUMB_W, th))
        bar[26:26 + th, x:x + THUMB_W] = cell
        cv2.rectangle(bar, (x, 26), (x + THUMB_W, 26 + th), color, 2)
        cv2.putText(bar, label[:16], (x + 2, GALLERY_H - 8), FONT, 0.45, color, 1)
        x += THUMB_W + 10
```

- [ ] **Step 6: Fix the dashboard tests for the new shapes**

In `tests/test_dashboard.py`, the `_compose` tests build an `info` dict and `thumbs`. Update them: `info` must use `"identified"`/`"unidentified"` keys instead of `"catches"`, and gallery thumbs are 3-tuples. Replace the two compose tests' inputs:

```python
def test_compose_has_full_dashboard_height_empty_gallery():
    raw = np.zeros((720, 1280, 3), np.uint8)
    annotated = np.zeros((720, 1280, 3), np.uint8)
    info = {"mode": "supercars", "identified": 0, "unidentified": 0, "fps": 12.0, "last": None}
    comp = dashboard._compose(raw, annotated, info, [])
    assert comp.shape[0] == EXPECTED_H and comp.shape[2] == 3

def test_compose_bars_match_middle_width_with_thumbs():
    raw = np.zeros((480, 640, 3), np.uint8)
    annotated = np.zeros((480, 640, 3), np.uint8)
    info = {"mode": "all", "identified": 3, "unidentified": 1, "fps": 20.0, "last": "Ferrari"}
    thumbs = [(np.zeros((100, 150, 3), np.uint8), "Ferrari", "identified")]
    comp = dashboard._compose(raw, annotated, info, thumbs)
    assert comp.shape[0] == EXPECTED_H and comp.shape[1] > 0
```

- [ ] **Step 7: Verify**

Run: `.venv/bin/python -m pytest -q` → all pass, 1 deselected.
Render a dashboard frame on the clip to confirm the HUD split + tier colors compose without error:
```bash
.venv/bin/python -c "import notify; notify.ENABLED=False; import dashboard, cv2; dashboard.SOURCE='clips/test.mp4'; saved=[]
def d(c,i):
    saved.append(1)
    if len(saved)>=120: cv2.imwrite('/tmp/dash_tier.png', c); return False
    return True
dashboard.run(display=d); print('wrote /tmp/dash_tier.png')"
```
Expected: writes `/tmp/dash_tier.png` with no error (HUD reads `IDENTIFIED n  UNKNOWN m`).

- [ ] **Step 8: Commit**

```bash
git add dashboard.py tests/test_dashboard.py
git commit -m "feat: dashboard two-tier display (gold/cyan), HUD split, all-cars net"
```

---

## Notes for the implementer

- Leniency dials live in `spotter.py`: `MARGIN` (catch-at-all bar), `IDENTIFY_FLOOR` (name-it bar), `CONFIRM_STREAK` (set 1 for most lenient), `SAVE_ALL_CARS`.
- `catches/all/` can fill fast on a busy street — it's a raw backup; safe to delete. It's already gitignored (`catches/`).
- Privacy hardening (plate/face blur, retention) was discussed and deferred — not in this plan.
