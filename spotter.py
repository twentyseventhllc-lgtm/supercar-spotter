# spotter.py — Supercar Spotter
# ─── config ───────────────────────────────────────────────────
SOURCE     = "clips/test.mp4"   # a video file, a folder of clips, or 0 for webcam
BRANDS     = ["Ferrari", "Lamborghini", "Porsche", "McLaren", "Lotus",
              "Aston Martin", "Bugatti", "Maserati", "Audi R8", "Corvette"]
NEGATIVES  = ["ordinary car", "sedan", "SUV", "van", "truck"]
MARGIN     = 0.15      # how much a brand must beat the best normie score to count
MIN_CONFIDENCE = 0.85  # absolute brand confidence floor; kills weak/blurry false alarms
CONFIRM_STREAK = 3     # a car must be the SAME supercar this many checks before it counts
MIN_BOX_AREA = 4000    # px²; skip tiny far-away cars
CONF       = 0.30      # YOLO detection confidence floor
CAR_CLASSES = [2]      # COCO 'car'
OUT_DIR    = "catches"
SHOW       = True      # live preview window
# ──────────────────────────────────────────────────────────────

import csv
import glob
import os
import time

import cv2
from ultralytics import YOLO

from brain import BrandClassifier, pick_best
from catch import CatchTracker

LABELS = BRANDS + NEGATIVES
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def list_sources(source):
    if isinstance(source, int):
        return [source]
    if os.path.isdir(source):
        files = sorted(
            f for f in glob.glob(os.path.join(source, "*"))
            if f.lower().endswith(VIDEO_EXTS)
        )
        return files or []
    return [source]


def save_catch(out_dir, frame, action):
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    safe_label = action.label.replace(" ", "_")
    path = os.path.join(out_dir, f"{ts}_track{action.track_id}_{safe_label}.jpg")
    cv2.imwrite(path, frame)
    with open(os.path.join(out_dir, "log.csv"), "a", newline="") as fh:
        csv.writer(fh).writerow([ts, action.track_id, action.label,
                                 f"{action.confidence:.3f}", path])
    print(f"📸 caught {action.label} ({action.confidence:.0%}) → {path}")
    return path


def run():
    classifier = BrandClassifier(labels=LABELS)
    tracker = CatchTracker(confirm_streak=CONFIRM_STREAK)

    for src in list_sources(SOURCE):
        tracker.reset()  # fresh track IDs per clip
        model = YOLO("yolo11n.pt")
        stream = model.track(source=src, classes=CAR_CLASSES, conf=CONF,
                             persist=True, stream=True, verbose=False)

        for result in stream:
            frame = result.orig_img.copy()
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                _show(frame)
                continue

            for box, tid in zip(boxes, boxes.id.int().tolist()):
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
                    continue
                crop = result.orig_img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                verdict = pick_best(classifier.score(crop), BRANDS, NEGATIVES,
                                    MARGIN, MIN_CONFIDENCE)
                color = (0, 215, 255) if verdict.is_supercar else (120, 120, 120)
                label = (f"{verdict.label} {verdict.confidence:.0%}"
                         if verdict.is_supercar else "car")
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                action = tracker.update(tid, verdict)
                if action is not None:
                    save_catch(OUT_DIR, frame, action)

            if not _show(frame):
                return


def _show(frame):
    if not SHOW:
        return True
    cv2.imshow("Supercar Spotter", frame)
    return cv2.waitKey(1) & 0xFF != ord("q")  # False => quit


if __name__ == "__main__":
    run()
    cv2.destroyAllWindows()
