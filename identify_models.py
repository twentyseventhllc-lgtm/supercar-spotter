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
