# Temporal Consistency + Higher Floor — Implementation Plan (Phase 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Kill the live false-alarm flood by requiring a car to be classified as the *same* supercar across several checks before it counts, plus raising the confidence floor.

**Architecture:** Add a per-track agreement streak to `CatchTracker`: a catch fires only after `confirm_streak` agreeing *fresh* classifications of the same brand; a different brand or a non-supercar resets the streak. Wire it into the spotter and dashboard (which passes `fresh=False` for cached verdicts so the throttle's repeats don't inflate the streak). Raise `MIN_CONFIDENCE` 0.70 → 0.85.

**Tech Stack:** Python, existing `catch.py` / `spotter.py` / `dashboard.py`, pytest.

## Global Constraints

- Python 3.10+ (`X | None`, `dict[int, float]`, dataclasses).
- Run everything via the venv: `.venv/bin/python -m pytest ...`.
- `CatchTracker(confirm_streak=1)` MUST preserve today's behavior — the existing `tests/test_catch.py` cases keep passing unchanged.
- `MIN_CONFIDENCE` lives only in `spotter.py`; `dashboard.py` imports it (single source).
- `fresh` semantics: `True` on an actual CLIP classification, `False` on a reused/cached verdict. The spotter never caches → always `True` (the default). The dashboard caches every `CLASSIFY_EVERY` frames → `True` only on the re-classify frame.
- `CONFIRM_STREAK = 3` default.

---

### Task 1: Per-track agreement streak in `CatchTracker`

**Files:**
- Modify: `catch.py`
- Test: `tests/test_catch.py`

**Interfaces:**
- Consumes: `brain.Verdict` (`label: str`, `confidence: float`, `is_supercar: bool`).
- Produces:
  - `CatchTracker(confirm_streak: int = 1)`.
  - `update(track_id: int, verdict: Verdict, fresh: bool = True) -> CatchAction | None` — advances the per-track streak only when `fresh`; a catch is emitted only once a track has `confirm_streak` agreeing supercar verdicts for the SAME label (then on each strictly-higher-confidence frame). A non-supercar or different-label fresh verdict resets the streak.
  - `reset()` clears streak + best state.
  - `CatchAction` unchanged.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_catch.py`:

```python
def test_streak_required_before_catching():
    t = CatchTracker(confirm_streak=3)
    assert t.update(1, sc(0.6)) is None          # streak 1
    assert t.update(1, sc(0.7)) is None           # streak 2
    assert t.update(1, sc(0.8)) == CatchAction(track_id=1, label="Ferrari", confidence=0.8)

def test_different_brand_resets_streak():
    t = CatchTracker(confirm_streak=2)
    other = Verdict(label="Lamborghini", confidence=0.9, is_supercar=True)
    assert t.update(1, sc(0.7)) is None           # Ferrari streak 1
    assert t.update(1, other) is None             # switch -> Lambo streak 1 (reset)
    assert t.update(1, other) == CatchAction(track_id=1, label="Lamborghini", confidence=0.9)

def test_non_supercar_resets_streak():
    t = CatchTracker(confirm_streak=2)
    assert t.update(1, sc(0.7)) is None           # streak 1
    assert t.update(1, normie(0.9)) is None       # reset
    assert t.update(1, sc(0.7)) is None           # streak 1 again, not caught

def test_cached_verdict_does_not_advance_streak():
    t = CatchTracker(confirm_streak=3)
    assert t.update(1, sc(0.6), fresh=True) is None   # streak 1
    assert t.update(1, sc(0.6), fresh=False) is None  # cached, no advance
    assert t.update(1, sc(0.6), fresh=False) is None  # cached, no advance
    assert t.update(1, sc(0.7), fresh=True) is None   # streak 2
    assert t.update(1, sc(0.8), fresh=True) is not None  # streak 3 -> catch
```

(`sc` and `normie` helpers already exist at the top of `tests/test_catch.py`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_catch.py -v`
Expected: the 4 new tests FAIL (`CatchTracker() got an unexpected keyword argument 'confirm_streak'` / `update() got an unexpected keyword argument 'fresh'`).

- [ ] **Step 3: Rewrite `CatchTracker` in `catch.py`**

Replace the `CatchTracker` class with:

```python
class CatchTracker:
    def __init__(self, confirm_streak: int = 1):
        self._best: dict[int, float] = {}
        self._streak: dict[int, tuple] = {}   # track_id -> (label, count)
        self.confirm_streak = confirm_streak

    def update(self, track_id: int, verdict: Verdict,
               fresh: bool = True) -> CatchAction | None:
        if fresh:
            if verdict.is_supercar:
                label, count = self._streak.get(track_id, (None, 0))
                count = count + 1 if label == verdict.label else 1
                self._streak[track_id] = (verdict.label, count)
            else:
                self._streak[track_id] = (None, 0)   # non-supercar breaks the streak

        label, count = self._streak.get(track_id, (None, 0))
        confirmed = (verdict.is_supercar
                     and label == verdict.label
                     and count >= self.confirm_streak)
        if not confirmed:
            return None

        prev = self._best.get(track_id)
        if prev is not None and verdict.confidence <= prev:
            return None
        self._best[track_id] = verdict.confidence
        return CatchAction(track_id=track_id, label=verdict.label,
                           confidence=verdict.confidence)

    def reset(self) -> None:
        self._best.clear()
        self._streak.clear()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_catch.py -v`
Expected: ALL pass — the 6 original tests (with default `confirm_streak=1`) AND the 4 new ones.

- [ ] **Step 5: Commit**

```bash
git add catch.py tests/test_catch.py
git commit -m "feat: per-track agreement streak in CatchTracker (temporal consistency)"
```

---

### Task 2: Wire the streak + higher floor into the spotter and dashboard

**Files:**
- Modify: `spotter.py` (config + tracker construction)
- Modify: `dashboard.py` (config + tracker construction + pass `fresh`)

**Interfaces:**
- Consumes: `CatchTracker(confirm_streak=...)` and `update(..., fresh=...)` from Task 1.
- Produces: no new public API — behavior change only.

- [ ] **Step 1: Raise the floor and add `CONFIRM_STREAK` in `spotter.py`**

In `spotter.py`'s config block, change the `MIN_CONFIDENCE` line and add `CONFIRM_STREAK`:

```python
MIN_CONFIDENCE = 0.85  # absolute brand confidence floor; kills weak/blurry false alarms
CONFIRM_STREAK = 3     # a car must be the SAME supercar this many checks before it counts
```

- [ ] **Step 2: Use `confirm_streak` in `spotter.run()`**

In `spotter.py`, find `tracker = CatchTracker()` inside `run()` and change it to:

```python
    tracker = CatchTracker(confirm_streak=CONFIRM_STREAK)
```

(The spotter does not cache verdicts, so its `tracker.update(tid, verdict)` call keeps the default `fresh=True` — no other change there.)

- [ ] **Step 3: Add `CONFIRM_STREAK` to the `dashboard.py` config block**

In `dashboard.py`'s config block (just below `CLASSIFY_EVERY`), add:

```python
CONFIRM_STREAK = 3   # a car must be the SAME supercar this many fresh checks before it counts
```

- [ ] **Step 4: Construct the dashboard tracker with the streak**

In `dashboard.py`'s `run()`, change `tracker = CatchTracker()` to:

```python
    tracker = CatchTracker(confirm_streak=CONFIRM_STREAK)
```

- [ ] **Step 5: Pass `fresh` from the classify-throttle in `dashboard.py`**

In `_process_frame`, the supercars branch currently reads:

```python
        cache = state["verdict_cache"]
        if _due_for_classify(cache, tid, state["frame_no"], CLASSIFY_EVERY):
            verdict = pick_best(classifier.score(crop), BRANDS, NEGATIVES,
                                MARGIN, MIN_CONFIDENCE)
            cache[tid] = (verdict, state["frame_no"])
        else:
            verdict = cache[tid][0]
```

Add a `fresh` flag and pass it to `update`:

```python
        cache = state["verdict_cache"]
        if _due_for_classify(cache, tid, state["frame_no"], CLASSIFY_EVERY):
            verdict = pick_best(classifier.score(crop), BRANDS, NEGATIVES,
                                MARGIN, MIN_CONFIDENCE)
            cache[tid] = (verdict, state["frame_no"])
            fresh = True
        else:
            verdict = cache[tid][0]
            fresh = False
```

Then change the catch line lower in the same branch from `action = tracker.update(tid, verdict)` to:

```python
        action = tracker.update(tid, verdict, fresh=fresh)
```

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass, 1 deselected (the CLIP integration test). No regressions.

- [ ] **Step 7: Verify on the clip that real catches survive the streak**

Run (notify disabled so the test doesn't ping the phone):
```bash
.venv/bin/python -c "import notify; notify.ENABLED=False; import dashboard; dashboard.SOURCE='clips/test.mp4'; dashboard.run(display=lambda c,i: i['frame'] < 200)" 2>&1 | grep -E "caught" | head
```
Expected: the Lamborghini is STILL caught (it's consistently classified across many frames), but far fewer one-frame flukes appear than before. If nothing is caught at all, the streak is too strict for the clip — note it for the reviewer (do not change `CONFIRM_STREAK` without flagging).

- [ ] **Step 8: Commit**

```bash
git add spotter.py dashboard.py
git commit -m "feat: require agreement streak + raise floor to 0.85 in spotter/dashboard"
```

---

## Notes for the implementer

- This is Phase 1 of the accuracy work. Phase 2 (the trained CLIP-feature classifier) is a separate plan, written after this ships.
- Tuning lives in the config blocks: `CONFIRM_STREAK` (stricter = fewer false alarms, may miss fast cars) and `MIN_CONFIDENCE`.
