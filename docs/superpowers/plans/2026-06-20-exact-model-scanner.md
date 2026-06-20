# Exact-Model Scanner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A batch script that asks Gemini the exact car model for every caught photo, writing the answer to `catches/models.csv` and renaming each photo to embed the model.

**Architecture:** A standalone `identify_models.py`. Pure helpers (parse, sanitize, discover) are isolated and unit-tested; `query_gemini` calls the Gemini REST API via stdlib `urllib` (no new dependency); `identify_all` orchestrates with an injectable query so it's fully testable. Dedup via a `~` filename marker; `--limit` + resume make the ~3,900-photo `all/` folder tractable.

**Tech Stack:** Python stdlib (`urllib`, `base64`, `json`, `csv`, `glob`, `re`), pytest. No new pip deps. Gemini REST (`gemini-2.5-flash`).

## Global Constraints

- Python 3.10+. Run via `.venv/bin/python ...` and `.venv/bin/python -m pytest ...`.
- NO new pip dependency — Gemini is called with `urllib` (stdlib), exactly like `notify.py` calls ntfy.
- API key is read at runtime from `GEMINI_API_KEY` env var, else `~/.gemini/settings.json` (`{"GEMINI_API_KEY": ...}`). The key MUST NOT appear in code or git.
- Default model `gemini-2.5-flash`. REST endpoint: `https://generativelanguage.googleapis.com/v1beta/models/<model>:generateContent?key=<key>`, POST JSON `{"contents":[{"parts":[{"text": PROMPT},{"inline_data":{"mime_type":"image/jpeg","data": <base64>}}]}]}`; model text at `candidates[0].content.parts[0].text`.
- Scan set: `catches/*.jpg` (identified + unidentified) and `catches/all/*.jpg`. NOT `catches/me/` (face enrollment). Skip any filename containing `~` (already scanned).
- Every scanned photo is renamed `<stem>~<safe_model><ext>` (including `~unknown`) and gets a `catches/models.csv` row `ts, orig_name, model`.
- On a per-photo error (network etc.): log it, leave the photo un-renamed/un-logged so the next run retries; never crash the batch.
- `--limit N` default 200.

---

### Task 1: `identify_models.py` — pure helpers (parse, sanitize, discover, key)

**Files:**
- Create: `identify_models.py`
- Create: `tests/test_identify_models.py`

**Interfaces:**
- Produces (all pure / no network):
  - `parse_model(response: str) -> str` — first non-empty trimmed line; an `unknown` line (case-insensitive) → `"unknown"`; empty response → `"unknown"`.
  - `safe_model_name(model: str) -> str` — filename-safe token: drop `()`, non-alphanumerics → `_`, strip leading/trailing `_`; empty → `"unknown"`.
  - `marked_name(path: str, model: str) -> str` — `<stem>~<safe_model><ext>`.
  - `pending_photos(catches_dir: str, limit: int) -> list[str]` — sorted `.jpg` under `catches_dir` and `catches_dir/all`, excluding names containing `~`, capped at `limit`.
  - `_load_api_key(settings_path: str = "~/.gemini/settings.json") -> str | None` — env `GEMINI_API_KEY` first, else the key from the settings JSON, else `None`.
  - Module constants `MODEL`, `CATCHES_DIR`, `DEFAULT_LIMIT`, `PROMPT`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_identify_models.py
import csv
import glob
import json
import os

import identify_models as im

def test_parse_model_takes_first_nonempty_line():
    assert im.parse_model("Ferrari 488 GTB") == "Ferrari 488 GTB"
    assert im.parse_model("\n\n  Porsche 911 (992) Turbo S \n") == "Porsche 911 (992) Turbo S"

def test_parse_model_unknown_and_empty():
    assert im.parse_model("unknown") == "unknown"
    assert im.parse_model("Unknown\n") == "unknown"
    assert im.parse_model("   ") == "unknown"

def test_safe_model_name():
    assert im.safe_model_name("Porsche 911 (992) Turbo S") == "Porsche_911_992_Turbo_S"
    assert im.safe_model_name("Lamborghini Aventador SVJ") == "Lamborghini_Aventador_SVJ"
    assert im.safe_model_name("") == "unknown"

def test_marked_name():
    assert im.marked_name("catches/x_Ferrari.jpg", "Ferrari 488 GTB") == \
        "catches/x_Ferrari~Ferrari_488_GTB.jpg"

def test_pending_photos_filters_and_limits(tmp_path):
    cat = tmp_path / "catches"
    (cat / "all").mkdir(parents=True)
    (cat / "a_Ferrari.jpg").write_bytes(b"x")
    (cat / "b_unidentified.jpg").write_bytes(b"x")
    (cat / "c_Ferrari~Ferrari_488.jpg").write_bytes(b"x")   # already scanned -> skip
    (cat / "all" / "d_car.jpg").write_bytes(b"x")
    (cat / "notes.txt").write_bytes(b"x")                    # not a jpg
    got = {os.path.basename(p) for p in im.pending_photos(str(cat), limit=100)}
    assert got == {"a_Ferrari.jpg", "b_unidentified.jpg", "d_car.jpg"}
    assert len(im.pending_photos(str(cat), limit=2)) == 2

def test_load_api_key_env_first(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    assert im._load_api_key() == "env-key"

def test_load_api_key_from_settings(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"GEMINI_API_KEY": "file-key"}))
    assert im._load_api_key(str(settings)) == "file-key"

def test_load_api_key_none(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert im._load_api_key(str(tmp_path / "nope.json")) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_identify_models.py -q`
Expected: FAIL (`No module named 'identify_models'`).

- [ ] **Step 3: Write `identify_models.py` (config + pure helpers)**

```python
# identify_models.py — ask Gemini the exact car model for every caught photo.
MODEL         = "gemini-2.5-flash"
CATCHES_DIR   = "catches"
DEFAULT_LIMIT = 200
PROMPT = ("Identify the exact car in this image. Reply with ONLY the make and model "
          "(plus generation/year if clearly identifiable), e.g. Porsche 911 (992) Turbo S. "
          "If it is not clearly a car or you cannot tell the model, reply exactly unknown.")

import argparse
import base64
import csv
import glob
import json
import os
import re
import time


def parse_model(response):
    for line in response.strip().splitlines():
        line = line.strip()
        if line:
            return "unknown" if line.lower() == "unknown" else line
    return "unknown"


def safe_model_name(model):
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", model.replace("(", "").replace(")", ""))
    return cleaned.strip("_") or "unknown"


def marked_name(path, model):
    stem, ext = os.path.splitext(path)
    return f"{stem}~{safe_model_name(model)}{ext}"


def pending_photos(catches_dir, limit):
    paths = sorted(glob.glob(os.path.join(catches_dir, "*.jpg")) +
                   glob.glob(os.path.join(catches_dir, "all", "*.jpg")))
    pending = [p for p in paths if "~" not in os.path.basename(p)]
    return pending[:limit]


def _load_api_key(settings_path="~/.gemini/settings.json"):
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    path = os.path.expanduser(settings_path)
    if os.path.exists(path):
        return json.load(open(path)).get("GEMINI_API_KEY")
    return None
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_identify_models.py -q`
Expected: all 8 PASS.

- [ ] **Step 5: Commit**

```bash
git add identify_models.py tests/test_identify_models.py
git commit -m "feat: exact-model scanner pure helpers (parse, sanitize, discover, key)"
```

---

### Task 2: Gemini REST query + `identify_all` orchestrator + CLI

**Files:**
- Modify: `identify_models.py` (append `query_gemini`, `identify_all`, `__main__`)
- Modify: `tests/test_identify_models.py` (add the orchestrator test)

**Interfaces:**
- Consumes: `parse_model`, `marked_name`, `pending_photos`, `_load_api_key`, constants.
- Produces:
  - `query_gemini(image_path, model, key) -> str` — base64 the image, POST to the Gemini REST API via `urllib`, return `candidates[0].content.parts[0].text`.
  - `identify_all(limit, query=None, catches_dir=CATCHES_DIR, csv_path=None)` — for each pending photo: `model = parse_model(query(path))`; append `[ts, basename, model]` to `csv_path` (default `<catches_dir>/models.csv`); `os.rename` to `marked_name`. Per-photo exceptions are caught, logged, and skip (no rename/no row). When `query is None`, builds the real one from `_load_api_key()` + `MODEL`.

- [ ] **Step 1: Write the failing orchestrator test**

Add to `tests/test_identify_models.py`:

```python
def test_identify_all_renames_and_logs(tmp_path):
    cat = tmp_path / "catches"
    (cat / "all").mkdir(parents=True)
    (cat / "a_Ferrari.jpg").write_bytes(b"x")
    (cat / "all" / "b_car.jpg").write_bytes(b"x")
    csv_path = cat / "models.csv"

    def fake_query(path):
        return "Ferrari 488 GTB" if "Ferrari" in path else "unknown"

    im.identify_all(limit=100, query=fake_query,
                    catches_dir=str(cat), csv_path=str(csv_path))

    names = {os.path.basename(p)
             for p in glob.glob(os.path.join(str(cat), "**", "*.jpg"), recursive=True)}
    assert any("~Ferrari_488_GTB" in n for n in names)   # identified -> renamed
    assert any("~unknown" in n for n in names)            # unknown -> renamed ~unknown
    rows = list(csv.reader(open(csv_path)))
    assert len(rows) == 2 and any("Ferrari 488 GTB" in r for r in rows)

def test_identify_all_skips_on_query_error(tmp_path):
    cat = tmp_path / "catches"
    cat.mkdir()
    (cat / "a.jpg").write_bytes(b"x")

    def boom(path):
        raise RuntimeError("network down")

    im.identify_all(limit=100, query=boom,
                    catches_dir=str(cat), csv_path=str(cat / "models.csv"))
    # untouched -> not renamed, retried next run
    assert os.path.exists(cat / "a.jpg")
    assert not os.path.exists(cat / "models.csv")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_identify_models.py -q`
Expected: the 2 new tests FAIL (`identify_all` undefined); the 8 from Task 1 still pass.

- [ ] **Step 3: Append the query + orchestrator + main to `identify_models.py`**

```python
def query_gemini(image_path, model, key):
    import urllib.request
    with open(image_path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    body = json.dumps({"contents": [{"parts": [
        {"text": PROMPT},
        {"inline_data": {"mime_type": "image/jpeg", "data": b64}}]}]}).encode()
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={key}")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    resp = json.load(urllib.request.urlopen(req, timeout=30))
    return resp["candidates"][0]["content"]["parts"][0]["text"]


def identify_all(limit, query=None, catches_dir=CATCHES_DIR, csv_path=None):
    csv_path = csv_path or os.path.join(catches_dir, "models.csv")
    if query is None:
        key = _load_api_key()
        if not key:
            print("No GEMINI_API_KEY (env or ~/.gemini/settings.json) — cannot scan.")
            return
        query = lambda path: query_gemini(path, MODEL, key)

    photos = pending_photos(catches_dir, limit)
    print(f"Scanning {len(photos)} photo(s)...")
    for i, path in enumerate(photos, 1):
        try:
            model = parse_model(query(path))
        except Exception as exc:                       # noqa: BLE001 - never crash the batch
            print(f"  [{i}/{len(photos)}] {os.path.basename(path)}: {exc} (retry next run)")
            continue
        ts = time.strftime("%Y%m%d-%H%M%S")
        with open(csv_path, "a", newline="") as fh:
            csv.writer(fh).writerow([ts, os.path.basename(path), model])
        new_path = marked_name(path, model)
        os.rename(path, new_path)
        print(f"  [{i}/{len(photos)}] {model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask Gemini the exact model of caught cars.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help="max new photos to scan this run")
    parser.add_argument("--model", default=MODEL, help="Gemini model name")
    args = parser.parse_args()
    MODEL = args.model
    identify_all(args.limit)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_identify_models.py -q`
Expected: all 10 PASS.

- [ ] **Step 5: Confirm the whole suite is still green**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass, 1 deselected (the existing CLIP integration test).

- [ ] **Step 6: Real end-to-end smoke (scans ONE real catch photo)**

Run: `.venv/bin/python identify_models.py --limit 1`
Expected: prints `Scanning 1 photo(s)...` then a real model line (e.g. `Lamborghini Aventador SVJ`); a `catches/models.csv` appears with one row; one photo in `catches/` is renamed with a `~<model>` suffix. (This makes ONE real Gemini call and renames ONE real file — both intended.) If it prints the no-key message, the key isn't readable — report it.

- [ ] **Step 7: Commit**

```bash
git add identify_models.py tests/test_identify_models.py
git commit -m "feat: Gemini REST query + identify_all orchestrator + CLI"
```

---

## Notes for the implementer

- To scan the whole backlog later: `python identify_models.py` repeatedly (each run does up to `--limit`, default 200, skipping already-marked photos). The `all/` folder (~3,900) takes several runs.
- The key is never written to code/git — only read from the env or `~/.gemini/settings.json` at runtime.
- `catches/` is gitignored, so renames + `models.csv` are local only.
