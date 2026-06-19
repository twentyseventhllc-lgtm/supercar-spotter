# dashboard.py — Supercar Spotter, deluxe live view.
#
# One window, four panels:
#   • top HUD bar      — title + live stats (mode, catch count, FPS, last brand)
#   • left pane        — the raw camera feed (clean)
#   • right pane       — the detection feed (boxes + brand labels)
#   • bottom gallery   — a strip of supercars caught this session
#
# Switch modes LIVE (no restart) with number keys:
#   [1] supercars  → cars are brand-scored by CLIP; exotics get caught + saved
#   [2] all        → every object YOLO sees, labeled (car, person, dog, ...)
#   [3] people     → just people
#   [4] cars       → every car boxed (no brand scoring)
#   [q] quit
#
# Reuses the tested core: brain (CLIP + decision), catch (dedup), spotter (save + config).
import time

import cv2
import numpy as np
from ultralytics import YOLO

from brain import BrandClassifier, pick_best
from catch import CatchTracker
from notify import notify_catch
from spotter import (BRANDS, NEGATIVES, MARGIN, MIN_CONFIDENCE,
                     MIN_BOX_AREA, CONF, OUT_DIR, save_catch, list_sources)

# ─── config ───────────────────────────────────────────────────
SOURCE  = 0          # webcam index, a clip path, or a folder of clips
PANE_H  = 380        # on-screen height of each feed pane (px)
CLASSIFY_EVERY = 6   # supercars mode: re-run CLIP on a tracked car every N frames
                     # (reuse the last verdict between) — higher = faster, laggier labels
CONFIRM_STREAK = 3   # a car must be the SAME supercar this many fresh checks before it counts
# ──────────────────────────────────────────────────────────────

CAR_CLASS = 2
PERSON_CLASS = 0
MODES = ["supercars", "all", "people", "cars"]
MODE_KEYS = {ord("1"): "supercars", ord("2"): "all",
             ord("3"): "people", ord("4"): "cars"}

FONT = cv2.FONT_HERSHEY_SIMPLEX
GOLD = (0, 215, 255)
GREY = (150, 150, 150)
WHITE = (240, 240, 240)
GREEN = (90, 220, 120)
CYAN = (230, 200, 60)

HUD_H = 72
GALLERY_H = 120
DIV_W = 4
THUMB_W = 150
LABELS = BRANDS + NEGATIVES


def _fit(frame, h):
    height, width = frame.shape[:2]
    return cv2.resize(frame, (max(1, int(width * h / height)), h))


def _draw_box(img, x1, y1, x2, y2, label, color):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(img, label, (x1, max(16, y1 - 7)), FONT, 0.6, color, 2)


def _make_hud(width, info):
    bar = np.full((HUD_H, width, 3), 18, np.uint8)
    cv2.putText(bar, "SUPERCAR SPOTTER", (14, 30), FONT, 0.85, GOLD, 2)
    stats = (f"MODE {info['mode'].upper()}     CATCHES {info['catches']}"
             f"     FPS {info['fps']:.0f}     LAST {info['last'] or '-'}")
    cv2.putText(bar, stats, (260, 30), FONT, 0.62, WHITE, 2)
    cv2.putText(bar, "[1] supercars   [2] all   [3] people   [4] cars   [q] quit",
                (14, 58), FONT, 0.55, GREY, 1)
    return bar


def _make_gallery(width, thumbs):
    bar = np.full((GALLERY_H, width, 3), 24, np.uint8)
    cv2.putText(bar, "CAUGHT", (14, 18), FONT, 0.5, GOLD, 1)
    th = GALLERY_H - 40
    x = 14
    for thumb, brand in thumbs[::-1]:          # newest first, left to right
        if x + THUMB_W > width:
            break
        cell = cv2.resize(thumb, (THUMB_W, th))
        bar[26:26 + th, x:x + THUMB_W] = cell
        cv2.rectangle(bar, (x, 26), (x + THUMB_W, 26 + th), GOLD, 2)
        cv2.putText(bar, brand[:16], (x + 2, GALLERY_H - 8), FONT, 0.45, GOLD, 1)
        x += THUMB_W + 10
    if not thumbs:
        cv2.putText(bar, "no supercars caught yet", (14, 70), FONT, 0.6, GREY, 1)
    return bar


def _compose(raw, annotated, info, thumbs):
    raw_r, ann_r = _fit(raw, PANE_H), _fit(annotated, PANE_H)
    cv2.putText(raw_r, "RAW", (10, 26), FONT, 0.7, WHITE, 2)
    cv2.putText(ann_r, "DETECTIONS", (10, 26), FONT, 0.7, GOLD, 2)
    div = np.full((PANE_H, DIV_W, 3), 60, np.uint8)
    middle = np.hstack([raw_r, div, ann_r])
    width = middle.shape[1]
    return np.vstack([_make_hud(width, info), middle, _make_gallery(width, thumbs)])


def _iter_dets(result):
    """Yield (x1, y1, x2, y2, cls_id, track_id) for each detection in a frame."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return
    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    clss = boxes.cls.int().tolist()
    ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(clss)
    for (x1, y1, x2, y2), cls_id, tid in zip(xyxy, clss, ids):
        yield x1, y1, x2, y2, cls_id, tid


def _due_for_classify(cache, tid, frame_no, every):
    """True if this track should be (re-)scored by CLIP now. Between re-scores we
    reuse the cached verdict, so CLIP runs ~1/every as often (the FPS win)."""
    last = cache.get(tid)
    return last is None or (frame_no - last[1]) >= every


def _process_frame(result, mode, classifier, tracker, state):
    """Draw detections for the current mode onto a copy of the frame; run catch
    logic in supercars mode. Returns the annotated frame."""
    raw = result.orig_img
    annotated = raw.copy()
    names = result.names
    state["frame_no"] += 1

    for x1, y1, x2, y2, cls_id, tid in _iter_dets(result):
        if mode == "people":
            if cls_id == PERSON_CLASS:
                _draw_box(annotated, x1, y1, x2, y2, "person", GREEN)
            continue
        if mode == "cars":
            if cls_id == CAR_CLASS:
                _draw_box(annotated, x1, y1, x2, y2, "car", CYAN)
            continue
        if mode == "all":
            _draw_box(annotated, x1, y1, x2, y2, names.get(cls_id, str(cls_id)), CYAN)
            continue

        # mode == "supercars": only cars, brand-scored by CLIP
        if cls_id != CAR_CLASS:
            continue
        if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA or tid is None:
            _draw_box(annotated, x1, y1, x2, y2, "car", GREY)
            continue
        crop = raw[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Only re-run CLIP every CLASSIFY_EVERY frames per car; reuse otherwise.
        cache = state["verdict_cache"]
        if _due_for_classify(cache, tid, state["frame_no"], CLASSIFY_EVERY):
            verdict = pick_best(classifier.score(crop), BRANDS, NEGATIVES,
                                MARGIN, MIN_CONFIDENCE)
            cache[tid] = (verdict, state["frame_no"])
            fresh = True
        else:
            verdict = cache[tid][0]
            fresh = False

        if verdict.is_supercar:
            _draw_box(annotated, x1, y1, x2, y2,
                      f"{verdict.label} {verdict.confidence:.0%}", GOLD)
        else:
            _draw_box(annotated, x1, y1, x2, y2, "car", GREY)

        action = tracker.update(tid, verdict, fresh=fresh)
        if action is not None:
            path = save_catch(OUT_DIR, annotated, action)
            state["caught_ids"].add(tid)
            state["last"] = action.label
            state["thumbs"].append((crop.copy(), action.label))
            state["thumbs"] = state["thumbs"][-12:]
            notify_catch(action, path)   # one phone push per car (self-throttled)

    return annotated


def run(source=None, display=None, max_frames=None):
    # Read the module-level SOURCE at call time (not as a default arg, which
    # would freeze it at import time and ignore `dashboard.SOURCE = ...`).
    if source is None:
        source = SOURCE
    classifier = BrandClassifier(labels=LABELS)
    tracker = CatchTracker(confirm_streak=CONFIRM_STREAK)
    mode = "supercars"
    state = {"caught_ids": set(), "thumbs": [], "last": None,
             "frame_no": 0, "verdict_cache": {}}
    fps, prev = 0.0, time.time()
    seen = 0

    for src in list_sources(source):
        tracker.reset()
        state["verdict_cache"].clear()   # fresh track ids per clip
        model = YOLO("yolo11n.pt")
        # No `classes=` filter: detect everything so modes can switch live.
        try:
            stream = model.track(source=src, conf=CONF, persist=True,
                                 stream=True, verbose=False)
            for result in stream:
                annotated = _process_frame(result, mode, classifier, tracker, state)

                now = time.time()
                dt = now - prev
                prev = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)
                info = {"mode": mode, "catches": len(state["caught_ids"]),
                        "fps": fps, "last": state["last"]}
                composite = _compose(result.orig_img, annotated, info, state["thumbs"])

                seen += 1
                if display is not None:
                    if display(composite, {**info, "frame": seen}) is False:
                        return
                else:
                    cv2.imshow("Supercar Spotter — Live", composite)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        cv2.destroyAllWindows()
                        return
                    if key in MODE_KEYS:
                        mode = MODE_KEYS[key]
                if max_frames and seen >= max_frames:
                    return
        except (ConnectionError, cv2.error) as e:
            print(f"\n⚠️  Lost the camera/source {src!r}.\n   {e}\n"
                  "   • Valid indexes on this Mac: run  pick_camera.py  to list them.\n"
                  "   • iPhone (Continuity Camera) must be MOUNTED, LOCKED and STILL —\n"
                  "     if it connects then drops after ~2s, OpenCV can't hold it.\n"
                  "   • Most reliable: record a clip on your phone and point SOURCE at the file.")
            continue

    if display is None:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
