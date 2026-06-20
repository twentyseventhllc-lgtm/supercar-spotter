# Exact-Model Scanner — Design

**Date:** 2026-06-20
**Status:** Approved (design), pending implementation plan

## Goal

A batch script (`identify_models.py`, run on demand) that walks every caught car
photo across all three buckets, asks **Gemini** "what exact car is this?", and
records the answer — into `catches/models.csv` AND by renaming each photo to embed
the exact model. Separate from the live spotter (never slows catching).

## Scope

All three buckets: `catches/` (identified + unidentified `.jpg`s) and `catches/all/`
(the `_car.jpg` safety net). Note: `catches/all/` is large (~3,900 photos at time of
writing), so a full scan is many calls — see the `--limit` + dedup design below.

## Flow

```
catches/*.jpg  +  catches/all/*.jpg   ── find UN-scanned photos
        │
        ▼  for each (up to --limit)
   Gemini REST API  +  base64 photo  →  "Lamborghini Aventador SVJ"
        │
        ├─▶ append catches/models.csv   (orig_name, exact_model, time)
        └─▶ rename:  ..._Ferrari.jpg  →  ..._Ferrari~Ferrari_488_GTB.jpg
```

## Access — zero new setup, no new dependency

Calls the Gemini REST API directly with `urllib` (stdlib, exactly how `notify.py`
talks to ntfy) — `POST .../v1beta/models/<model>:generateContent` with the prompt +
the base64-inlined image. The API key is read at runtime from
`~/.gemini/settings.json` (`GEMINI_API_KEY`) or the `GEMINI_API_KEY` env var — so the
secret never lands in code or git. Default model `gemini-2.5-flash` (verified: it
returned `Lamborghini Aventador SVJ` on a real catch photo). No CLI, no SDK install.

## Dedup / resumable / batched

- A scanned photo gets a `~<model>` marker inserted into its filename. The scanner
  skips any file whose name already contains `~`. Original catch filenames never
  contain `~`, so this cleanly separates done vs pending.
- `--limit N` (default 200): process at most N new photos per run, so a 3,900-photo
  `all/` folder is chipped away in sessions rather than one multi-hour run.
- Ctrl-C safe: each photo's CSV row + rename happen before moving on, so progress
  always persists; the next run resumes at the first un-marked photo.

## The prompt (strict, single-line answer)

> "Identify the exact car in this image. Reply with ONLY the make and model (plus
> generation/year if clearly identifiable), e.g. `Porsche 911 (992) Turbo S`. If it
> is not clearly a car, or you cannot tell the model, reply exactly `unknown`."

The response's first non-empty line is taken as the model. `unknown` (case-insensitive)
→ recorded as `unknown` AND the file renamed to `...~unknown.jpg` — so it isn't
re-scanned next run (the same blurry photo gives the same answer). Every processed
photo gets a `~` marker, so the marker alone is the dedup signal.

## Components (small, isolated, testable)

`identify_models.py`:
- `parse_model(response) -> str` — first non-empty trimmed line; maps an `unknown`
  answer to the literal `"unknown"`. **Pure, unit-tested.**
- `safe_model_name(model) -> str` — sanitize a model string into a filename-safe token
  (spaces/`()` → `_`, strip other punctuation, collapse repeats). **Pure, unit-tested.**
- `marked_name(path, model) -> str` — build the `<stem>~<safe_model><ext>` target path.
  **Pure, unit-tested.**
- `pending_photos(catches_dir, limit) -> list[str]` — discover `.jpg` images under
  `catches/` and `catches/all/`, skipping any whose name contains `~`, capped at
  `limit`. **Tested with a tmp dir.**
- `_load_api_key() -> str` — read `GEMINI_API_KEY` from the env or `~/.gemini/settings.json`.
- `query_gemini(image_path, model, key) -> str` — base64 the image, POST to the Gemini
  REST API via `urllib`, return the model text from the response. Injectable so the
  orchestrator is tested with a fake; **already smoke-verified on a real catch photo.**
- `identify_all(limit, query=query_gemini)` — orchestrate: for each pending photo →
  query → parse → append CSV → rename with the `~model` marker (including `~unknown`).
  `query` is injectable for tests.
- `main()` / argparse: `--limit N`, `--model NAME`.

`tests/test_identify_models.py`.

## Testing

- `parse_model`, `safe_model_name`, `marked_name` — pure unit tests.
- `pending_photos` — tmp dir with `.jpg`/`~`-marked/non-jpg files; asserts filtering +
  limit.
- `identify_all` — tmp `catches/` with a couple fake jpgs + an injected fake `query`;
  asserts CSV rows written, files renamed with the `~model` marker, and an `unknown`
  answer renames to `...~unknown.jpg`.
- `query_gemini` — manual/smoke: run the real CLI on ONE existing catch photo, confirm
  it returns a plausible model string (verifiable headlessly — no GUI needed).

## Honest notes

- One REST call per image (~1–2s network each); `all/` (~3,900) is still a long sweep
  (~1–2 hours total) — that's what `--limit` + dedup + resume are for. Optional light
  concurrency could be added later, but sequential is fine for v1.
- Gemini is strong on exotics, decent on common cars, `unknown` when unsure. Not 100%.
- Cost is tiny — one small image + short prompt per call, on the user's existing key.
- Network failure on a photo is caught, logged, and that photo left un-marked so the
  next run retries it (never crashes the batch).
