# identify_models.py — ask Gemini the exact car model for every caught photo.
MODEL         = "gemini-2.5-flash"
CATCHES_DIR   = "catches"
DEFAULT_LIMIT = 200
RPM_DELAY     = 6      # seconds between Gemini calls — stay under the free tier's per-minute cap
MAX_RETRIES   = 4      # on HTTP 429, wait (Retry-After or 30s) and retry this many times
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
import urllib.error
import urllib.request


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


def pending_photos(catches_dir, limit, include_all=True):
    paths = glob.glob(os.path.join(catches_dir, "*.jpg"))           # the cool catches
    if include_all:
        paths += glob.glob(os.path.join(catches_dir, "all", "*.jpg"))  # the firehose
    pending = [p for p in sorted(paths) if "~" not in os.path.basename(p)]
    return pending[:limit]


def _load_api_key(settings_path="~/.gemini/settings.json"):
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    path = os.path.expanduser(settings_path)
    if os.path.exists(path):
        return json.load(open(path)).get("GEMINI_API_KEY")
    return None


def query_gemini(image_path, model, key):
    time.sleep(RPM_DELAY)                          # throttle to respect the per-minute cap
    with open(image_path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    body = json.dumps({"contents": [{"parts": [
        {"text": PROMPT},
        {"inline_data": {"mime_type": "image/jpeg", "data": b64}}]}]}).encode()
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={key}")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = json.load(urllib.request.urlopen(req, timeout=30))
            return resp["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as exc:
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            if exc.code == 429 and attempt < MAX_RETRIES:
                wait = int(retry_after) if (retry_after or "").isdigit() else 30
                print(f"    rate-limited, waiting {wait}s (attempt {attempt + 1})...")
                time.sleep(wait)
                continue
            raise


def identify_all(limit, query=None, catches_dir=CATCHES_DIR, csv_path=None, include_all=True):
    csv_path = csv_path or os.path.join(catches_dir, "models.csv")
    if query is None:
        key = _load_api_key()
        if not key:
            print("No GEMINI_API_KEY (env or ~/.gemini/settings.json) — cannot scan.")
            return
        query = lambda path: query_gemini(path, MODEL, key)

    photos = pending_photos(catches_dir, limit, include_all)
    print(f"Scanning {len(photos)} photo(s)...")
    for i, path in enumerate(photos, 1):
        try:
            model = parse_model(query(path))
        except urllib.error.HTTPError as exc:
            if exc.code == 429:                        # quota exhausted even after retries
                print(f"  [{i}/{len(photos)}] rate limit reached — stopping. "
                      "Run again later (progress is saved).")
                break
            print(f"  [{i}/{len(photos)}] {os.path.basename(path)}: {exc} (retry next run)")
            continue
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
    parser.add_argument("--delay", type=float, default=RPM_DELAY,
                        help="seconds between calls (lower it if you enable billing/paid tier)")
    parser.add_argument("--skip-all", action="store_true",
                        help="only scan the cool catches/, not the big catches/all/ firehose")
    args = parser.parse_args()
    MODEL = args.model
    RPM_DELAY = args.delay
    identify_all(args.limit, include_all=not args.skip_all)
