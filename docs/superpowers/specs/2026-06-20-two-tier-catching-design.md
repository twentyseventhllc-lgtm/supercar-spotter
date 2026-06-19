# Two-Tier Catching + All-Cars Safety Net — Design

**Date:** 2026-06-20
**Status:** Approved (design), pending implementation plan

## Goal

Catch far more cars (it was too picky) by splitting catches into tiers instead of
"confident brand or nothing", and add a safety net that captures every car so the
ML can't silently miss one.

Three buckets:
1. **Identified** — a confidently-named brand (`Ferrari 95%`).
2. **Unidentified** — clearly exotic, but the brand isn't confident.
3. **All cars** — every car YOLO sees, saved regardless (safety net), toggleable.

## The decision logic (the core change)

Today a car is caught only if CLIP is confident about the brand AND it's the same
brand across the streak — that's why it's picky. New rule: **catch any car that
consistently looks exotic, then label it by confidence.**

Per CLIP check on a tracked car (`pick_best` returns a `Verdict`):
- **is_exotic** = best brand beats the best "ordinary car" negative by `MARGIN`
  (it looks like *some* supercar). This is the catch trigger — no high floor.
- **is_identified** = `is_exotic` AND best brand confidence ≥ `IDENTIFY_FLOOR`
  (sure enough to name it).

`Verdict` carries: `label` (top brand), `confidence`, `is_exotic`, `is_identified`.
(Replaces the old single `is_supercar`.)

## Catch flow per tracked car

```
every tracked car ─▶ save ONE photo to catches/all/ (if SAVE_ALL_CARS)   [no push]
        │
        ▼  is_exotic for CONFIRM_STREAK fresh checks in a row?
        │   (brand may flicker between exotics — that flicker IS the "unidentified" signal)
        ▼
   best check was is_identified ─▶ IDENTIFIED  → catches/<ts>_track<id>_<Brand>.jpg + push
   otherwise                     ─▶ UNIDENTIFIED → catches/<ts>_track<id>_unidentified.jpg + push
```

- The exotic **streak** replaces the old same-brand streak: a car must look exotic
  (beat ordinary) for `CONFIRM_STREAK` consecutive *fresh* checks. A non-exotic
  fresh check resets it. This kills single-frame flicker but is far less picky than
  requiring a confident, stable brand.
- **Tier** is decided when the streak confirms: identified if the best check so far
  was `is_identified`, else unidentified.

## Output — one folder, named files (cool catches); subfolder for the firehose

- Cool catches (identified + unidentified) → `catches/` with named files:
  - `<ts>_track<id>_<Brand>.jpg` (identified)
  - `<ts>_track<id>_unidentified.jpg` (unidentified)
- All-cars safety net → `catches/all/<ts>_track<id>_car.jpg` (one per car) — kept in
  its own subfolder so the cool catches aren't buried under normal cars.
- `catches/log.csv` gains a `tier` column (`identified` / `unidentified`); the all/
  folder is not logged to the main csv (it's a raw backup).

## Dashboard

- Box + gallery colors: **gold** = identified (brand name), **cyan** = unidentified
  (drawn as `exotic?`), grey `car` = everything else.
- HUD splits the count: `IDENTIFIED 3 · UNKNOWN 5`.

## Notifications

- Identified + unidentified push to the phone (the existing 60s cooldown applies).
  Title = brand for identified, `Unidentified exotic` for unidentified.
- All-cars saves are **silent** (no push — you don't want a ping per Honda).

## Config (leaning generous, since it was too picky)

- `MARGIN` — how exotic a car must look to catch at all (lower = more catches).
- `IDENTIFY_FLOOR` — confidence to *name* a brand (below = unidentified, not discarded).
  Replaces the role of `MIN_CONFIDENCE`.
- `CONFIRM_STREAK` — consecutive exotic checks (stays 2).
- `SAVE_ALL_CARS` — toggle the all/ safety net on/off.

## Components touched

- `brain.py` — `pick_best` returns `is_exotic` + `is_identified`; `Verdict` updated.
- `catch.py` — `CatchTracker` streaks on `is_exotic`; `CatchAction` gains `tier`;
  a separate one-shot-per-track path (or helper) for the all-cars net.
- `spotter.py` / `dashboard.py` — colors, `exotic?` label, HUD split, save-by-tier
  (incl. `catches/all/`), notify only for tiers.
- `notify.py` — title varies by tier.

## Testing

- `pick_best`: `is_exotic` / `is_identified` boundaries (pure, no model).
- `CatchTracker`: exotic-streak confirmation, tier selection (identified vs
  unidentified), non-exotic resets, all-cars one-per-track dedup (pure).
- `save_catch`/output: identified vs unidentified filename + `catches/all/` path +
  `tier` column in log.csv.
- Dashboard compose/HUD unaffected structurally; existing tests stay green.

## Out of scope (offered, user declined for now)

- License-plate / face blurring and auto-retention (privacy hardening) — noted,
  not built in this pass.
