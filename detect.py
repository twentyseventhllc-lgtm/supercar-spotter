from ultralytics import YOLO

# ─── pick a mode ──────────────────────────────────────────────
# "detect"      → boxes around people only
# "detect_all"  → boxes around everything (cars, dogs, etc.)
# "track"       → boxes + a sticky ID on each person as they move
# "pose"        → skeleton / joints instead of boxes
# "segment"     → exact outline (mask) of each person
MODE = "detect"
# ──────────────────────────────────────────────────────────────

CONFIG = {
    "detect":     ("yolo11n.pt",     "predict", [0]),
    "detect_all": ("yolo11n.pt",     "predict", None),
    "track":      ("yolo11n.pt",     "track",   [0]),
    "pose":       ("yolo11n-pose.pt","predict", None),
    "segment":    ("yolo11n-seg.pt", "predict", [0]),
}

weights, method, classes = CONFIG[MODE]
model = YOLO(weights)   # auto-downloads the first time

kwargs = {"source": 0, "show": True}   # source=0 is your webcam
if classes is not None:
    kwargs["classes"] = classes

# run it (press q in the window to quit)
getattr(model, method)(**kwargs)
