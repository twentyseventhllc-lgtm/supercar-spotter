# Supercar Spotter — Design

**Date:** 2026-06-19
**Status:** Approved (design), pending implementation plan

## Goal

A fun project that watches a video feed of a street, finds every car, decides
whether each one is an exotic supercar (Ferrari, Lamborghini, Porsche, etc.) vs
an ordinary car, and saves a labeled photo of every supercar it catches.

"Nothing serious" — a catch-cam for fun, not a precision instrument.

## Approach (decided)

**Detector + Describer, two-stage, zero-shot (no model training):**

1. **YOLO** detects and tracks cars in the frame (uses the stock COCO `car` /
   `truck` classes — rock-solid at *finding* vehicles).
2. **CLIP** (zero-shot) looks at each cropped car and scores it against a
   text word-list of supercar brands plus "ordinary car" negatives. No dataset,
   no training — adding a brand is editing one line of text.

Rejected alternatives:
- **YOLO-World (single-stage open-vocab):** elegant but flakier on small/distant
  street cars and fiddlier to tune at brand granularity.
- **Custom-trained model:** best accuracy but requires collecting + hand-labeling
  thousands of images. Filed as possible Phase 2 if accuracy becomes a priority.

## Honest accuracy expectations

- "Exotic supercar vs normal car" → works well; supercars are visually distinctive.
- "Exact brand (Ferrari vs Lambo vs McLaren)" → good on clear shots, wrong
  sometimes. Acceptable for a fun catch-cam.

## Architecture

```
video feed ──▶ YOLO (find + track cars) ──▶ crop each car
                                                │
                                                ▼
                              CLIP (score crop vs brand word-list)
                                                │
                            ┌───────────────────┴───────────────┐
                       "ordinary car"                    "Ferrari" (0.82)
                         → ignore                    → save labeled snapshot
```

## Components (each small, isolated, testable)

### `brain.py` — the brand classifier
- **Does:** given a cropped car image (numpy/PIL), returns `(best_label, confidence)`
  by scoring the crop against a configurable list of text prompts via CLIP.
- **Used by:** `spotter.py`.
- **Depends on:** a CLIP implementation (open_clip / transformers — pick simplest
  install at build time), torch.
- **Testable in isolation:** feed it a few sample crops (a Ferrari, a Honda) and
  assert the labels. The "is this an exotic?" decision lives here.

### `spotter.py` — main loop + config
- **Does:** reads the video source, runs YOLO detect+track, crops tracked cars,
  calls `brain.py`, applies catch logic, draws a live preview, saves catches.
- **Config block at the top** (same friendly style as the existing `detect.py`):
  - `SOURCE` — path to a clip, a folder of clips, or `0` for the live webcam.
  - `BRANDS` — the supercar word-list.
  - `NEGATIVES` — the "ordinary car" prompts.
  - `MARGIN` — how much an exotic must beat "ordinary" to count (tune
    catch-more vs fewer-false-alarms).
  - `OUT_DIR` — where catches are saved.
- **Depends on:** ultralytics (YOLO), opencv, `brain.py`.

### `catches/` — output
- Timestamped `.jpg` per caught supercar, brand + confidence burned onto the image.
- `log.csv` — one row per catch (timestamp, brand, confidence, source frame).

## Key behavior: one car = one catch

A car takes ~30 frames to cross the view. We must not save 30 photos of the same
Ferrari. YOLO's **tracker** assigns each car a sticky ID; we classify a track a
few times as it passes and save **one best shot per track ID** (highest-confidence,
clearest frame). A track must not produce a second catch.

## What counts as a catch

Default editable lists:
- **BRANDS:** Ferrari, Lamborghini, Porsche, McLaren, Lotus, Aston Martin,
  Bugatti, Maserati, Audi R8, Corvette
- **NEGATIVES:** ordinary car, sedan, SUV, van, truck

A catch fires only when the best brand score beats the best negative score by at
least `MARGIN`.

## Testing workflow

1. Drop supercar street clips into `clips/`.
2. Run `spotter.py` with `SOURCE` pointed at the clips.
3. Watch the live window: boxes + brand labels on passing cars.
4. Confirm `catches/` fills only with exotics; tune `MARGIN` / lists.
5. Flip `SOURCE = 0` to go live on the webcam pointed out the window.

## Testing strategy

- **`brain.py`:** unit tests with a handful of sample car crops — assert exotic
  vs ordinary classification and that brand labels are reasonable.
- **Catch logic (dedup/cooldown):** unit-testable — feed a simulated sequence of
  per-track detections, assert exactly one catch per qualifying track.
- **Main loop:** manual integration test via the clips workflow above.

## Out of scope for v1 (potential Phase 2)

- Pinging the phone (a free Telegram bot that texts the caught photo).
- Training a custom model for pinpoint brand accuracy.

## Dependencies

- `ultralytics` (YOLO + tracker) — already implied by existing `detect.py`.
- `opencv-python` — capture, draw, save (pulled by ultralytics).
- A CLIP implementation (`open_clip_torch` or `transformers`) + `torch`.
- Captured in a `requirements.txt`.
