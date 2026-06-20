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
