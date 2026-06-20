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
