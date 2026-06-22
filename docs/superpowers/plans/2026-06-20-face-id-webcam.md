# Face-ID Webcam Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A live webcam app that draws a body hitbox + face box around each person and tags them yellow `WW · admin` if it's the owner, blue `WW · unknown` otherwise.

**Architecture:** YOLO detects persons (body hitbox). `facenet-pytorch` (MTCNN) detects faces and FaceNet embeds them into 512-d vectors; a vector close (euclidean) to an enrolled owner vector ⇒ admin. Pure logic (`is_admin`, geometry, save/load) is isolated and unit-tested; the model wrapper and the live loop are thin.

**Tech Stack:** Python, ultralytics YOLO, facenet-pytorch (MTCNN + InceptionResnetV1), OpenCV, numpy, pytest. Runs on the existing `.venv` (torch/MPS).

## Global Constraints

- Python 3.10+. Run everything via `.venv/bin/python -m pytest ...` and `.venv/bin/python ...`.
- `face_id.py` must keep torch / facenet_pytorch imports INSIDE the `FaceID` class (lazy), so the pure functions (`is_admin`, `face_center_in_box`, `save_reference`, `load_reference`) import with no heavy deps and stay fast to test.
- Model-dependent tests are marked `integration` and excluded by the default run (existing `pytest.ini` already has `addopts = -m "not integration"`).
- Tag text is the literal string `"WW"`. Colors (BGR): admin = yellow `(0, 255, 255)`, unknown = blue `(255, 0, 0)`, no-face = grey `(150, 150, 150)`.
- Owner reference stored at `me/embeddings.npy` (gitignore `me/`). `THRESHOLD = 0.9` default (euclidean on L2-normalized embeddings).

---

### Task 1: Setup — add facenet-pytorch, gitignore, import smoke test

**Files:**
- Modify: `requirements.txt`, `.gitignore`
- Create: `tests/test_faceid_smoke.py`

**Interfaces:**
- Produces: an env where `facenet_pytorch` imports.

- [ ] **Step 1: Add the dependency to `requirements.txt`**

Append this line:
```
facenet-pytorch>=2.5.3
```

- [ ] **Step 2: Ignore the owner reference folder in `.gitignore`**

Append:
```
# Face-ID owner reference
me/
```

- [ ] **Step 3: Install it (without disturbing the existing torch/MPS build)**

Run: `.venv/bin/pip install facenet-pytorch`
Then check torch is intact: `.venv/bin/python -c "import torch; print(torch.__version__)"`
Expected: facenet-pytorch installs and torch still imports.
IF the pip output shows it trying to UNINSTALL/REPLACE torch or torchvision with a different version, abort that and instead run:
`.venv/bin/pip install --no-deps facenet-pytorch` then `.venv/bin/pip install requests` (facenet needs it). This reuses our existing torch/torchvision.

- [ ] **Step 4: Write the smoke test `tests/test_faceid_smoke.py`**

```python
def test_facenet_imports():
    import facenet_pytorch  # noqa: F401
```

- [ ] **Step 5: Run it**

Run: `.venv/bin/python -m pytest tests/test_faceid_smoke.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .gitignore tests/test_faceid_smoke.py
git commit -m "chore: add facenet-pytorch for face-id webcam"
```

---

### Task 2: `face_id.py` — pure logic (match, geometry, save/load)

**Files:**
- Create: `face_id.py`
- Create: `tests/test_face_id.py`

**Interfaces:**
- Produces (all importable with NO torch):
  - `is_admin(embedding, reference, threshold) -> bool` — True iff the min euclidean distance from `embedding` (1-d np array) to any row of `reference` (2-d np array) is `< threshold`. `None`/empty reference → False.
  - `face_center_in_box(face_box, person_box) -> bool` — True iff the centre of `face_box` (x1,y1,x2,y2) lies inside `person_box`.
  - `save_reference(embeddings, path) -> None` — `np.save` a 2-d array, creating parent dirs.
  - `load_reference(path) -> np.ndarray | None` — load it, or `None` if the file is missing.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_face_id.py
import numpy as np
from face_id import is_admin, face_center_in_box, save_reference, load_reference

def test_is_admin_true_when_close():
    ref = np.array([[1.0, 0.0, 0.0]])
    near = np.array([0.95, 0.05, 0.0])
    assert is_admin(near, ref, threshold=0.9) is True

def test_is_admin_false_when_far():
    ref = np.array([[1.0, 0.0, 0.0]])
    far = np.array([-1.0, 0.0, 0.0])
    assert is_admin(far, ref, threshold=0.9) is False

def test_is_admin_false_when_no_reference():
    assert is_admin(np.array([1.0, 0.0]), None, threshold=0.9) is False
    assert is_admin(np.array([1.0, 0.0]), np.empty((0, 2)), threshold=0.9) is False

def test_is_admin_uses_closest_reference():
    ref = np.array([[5.0, 5.0], [1.0, 0.0]])   # second row is close
    emb = np.array([1.05, 0.0])
    assert is_admin(emb, ref, threshold=0.5) is True

def test_face_center_inside_person_box():
    assert face_center_in_box((40, 40, 60, 60), (0, 0, 100, 200)) is True

def test_face_center_outside_person_box():
    assert face_center_in_box((400, 40, 460, 100), (0, 0, 100, 200)) is False

def test_save_load_round_trip(tmp_path):
    embs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    path = str(tmp_path / "me" / "embeddings.npy")
    save_reference(embs, path)
    loaded = load_reference(path)
    assert np.allclose(loaded, embs)

def test_load_missing_returns_none(tmp_path):
    assert load_reference(str(tmp_path / "nope.npy")) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_face_id.py -q`
Expected: FAIL (`No module named 'face_id'`).

- [ ] **Step 3: Write `face_id.py` (pure parts only)**

```python
# face_id.py — face recognition for the Face-ID webcam.
# Pure helpers (no torch) live at module top so they import fast for tests.
# The heavy model wrapper (FaceID) lazy-imports torch/facenet_pytorch.
import os

import numpy as np


def is_admin(embedding, reference, threshold):
    if reference is None or len(reference) == 0:
        return False
    dists = np.linalg.norm(np.asarray(reference) - np.asarray(embedding), axis=1)
    return bool(dists.min() < threshold)


def face_center_in_box(face_box, person_box):
    fx1, fy1, fx2, fy2 = face_box
    px1, py1, px2, py2 = person_box
    cx, cy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
    return bool(px1 <= cx <= px2 and py1 <= cy <= py2)


def save_reference(embeddings, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, np.asarray(embeddings))


def load_reference(path):
    if not os.path.exists(path):
        return None
    return np.load(path)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_face_id.py -q`
Expected: all 8 PASS.

- [ ] **Step 5: Commit**

```bash
git add face_id.py tests/test_face_id.py
git commit -m "feat: face_id pure logic (is_admin, geometry, save/load)"
```

---

### Task 3: `FaceID` model wrapper — detect + embed faces

**Files:**
- Modify: `face_id.py` (append `FaceID`)
- Create: `tests/test_faceid_model.py`

**Interfaces:**
- Consumes: numpy, torch, facenet_pytorch, cv2, PIL (all lazy, inside the class).
- Produces:
  - `FaceID()` — loads MTCNN (keep_all) + InceptionResnetV1(pretrained="vggface2") on cuda/mps/cpu.
  - `.embed_faces(frame_bgr) -> list[(box, embedding)]` — each `box` is an int `(x1,y1,x2,y2)` numpy array, each `embedding` a 512-d L2-normalized float32 numpy array. Empty list when no face is found.

- [ ] **Step 1: Write the failing (integration) test**

```python
# tests/test_faceid_model.py
import numpy as np
import pytest

pytestmark = pytest.mark.integration  # downloads MTCNN + FaceNet weights

def test_embed_faces_returns_list_and_no_face_is_empty():
    from face_id import FaceID
    fid = FaceID()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)   # no face present
    out = fid.embed_faces(blank)
    assert isinstance(out, list)
    assert out == []                                   # nothing to detect
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_faceid_model.py -m integration -q`
Expected: FAIL (`cannot import name 'FaceID'`).

- [ ] **Step 3: Append `FaceID` to `face_id.py`**

```python
class FaceID:
    """Detect faces (MTCNN) and embed them (FaceNet). Lazy-imports heavy deps."""

    def __init__(self):
        import torch
        from facenet_pytorch import MTCNN, InceptionResnetV1

        self._torch = torch
        if torch.cuda.is_available():
            self.device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        print(f"(FaceID running on {self.device})")

    def embed_faces(self, frame_bgr):
        import cv2
        from PIL import Image

        torch = self._torch
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        boxes, _ = self.mtcnn.detect(pil)
        if boxes is None:
            return []
        faces = self.mtcnn(pil)            # aligned face tensors, same order as boxes
        if faces is None:
            return []
        if faces.ndim == 3:                # single face -> add batch dim
            faces = faces.unsqueeze(0)

        with torch.no_grad():
            embs = self.resnet(faces.to(self.device))
            embs = embs / embs.norm(dim=-1, keepdim=True)   # L2-normalize
        embs = embs.cpu().numpy().astype("float32")

        out = []
        for box, emb in zip(boxes, embs):
            out.append((box.astype(int), emb))
        return out
```

- [ ] **Step 4: Run the integration test to verify pass**

Run: `.venv/bin/python -m pytest tests/test_faceid_model.py -m integration -q`
Expected: PASS (first run downloads weights — slow, normal).

- [ ] **Step 5: Confirm the default fast suite still skips it**

Run: `.venv/bin/python -m pytest -q`
Expected: pure tests run; the integration test is deselected.

- [ ] **Step 6: Commit**

```bash
git add face_id.py tests/test_faceid_model.py
git commit -m "feat: FaceID model wrapper (MTCNN detect + FaceNet embed)"
```

---

### Task 4: `facecam.py` — live app + enroll mode + drawing

**Files:**
- Create: `facecam.py`
- Create: `tests/test_facecam.py`

**Interfaces:**
- Consumes: `ultralytics.YOLO`, `cv2`, `face_id` (`FaceID`, `is_admin`, `face_center_in_box`, `save_reference`, `load_reference`).
- Produces: a runnable script (`python facecam.py` = live; `python facecam.py --enroll` = capture owner shots). Plus a pure `role_and_color(is_admin_bool) -> (str, tuple)` helper.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_facecam.py
def test_role_and_color_admin():
    import facecam
    role, color = facecam.role_and_color(True)
    assert role == "admin" and color == facecam.YELLOW

def test_role_and_color_unknown():
    import facecam
    role, color = facecam.role_and_color(False)
    assert role == "unknown" and color == facecam.BLUE

def test_tag_is_ww():
    import facecam
    assert facecam.TAG == "WW"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_facecam.py -q`
Expected: FAIL (`No module named 'facecam'`).

- [ ] **Step 3: Write `facecam.py`**

```python
# facecam.py — Face-ID webcam: yellow "WW admin" for the owner, blue "WW unknown" for others.
# ─── config ───────────────────────────────────────────────────
SOURCE       = 0          # webcam index
THRESHOLD    = 0.9        # max face distance to count as the owner (lower = stricter)
TAG          = "WW"       # text drawn on top of every box
ENROLL_SHOTS = 5          # how many reference shots --enroll captures
REF_PATH     = "me/embeddings.npy"
PERSON_CONF  = 0.40       # YOLO person-detection confidence floor
# ──────────────────────────────────────────────────────────────

import argparse

import cv2
import numpy as np

from face_id import (FaceID, is_admin, face_center_in_box,
                     save_reference, load_reference)

YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)
GREY = (150, 150, 150)
FONT = cv2.FONT_HERSHEY_SIMPLEX
PERSON_CLASS = 0


def role_and_color(admin):
    return ("admin", YELLOW) if admin else ("unknown", BLUE)


def _draw(frame, box, color, label):
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, TAG, (x1, max(28, y1 - 26)), FONT, 0.7, color, 2)
    cv2.putText(frame, label, (x1, max(12, y1 - 6)), FONT, 0.6, color, 2)


def run():
    reference = load_reference(REF_PATH)
    if reference is None:
        print("No enrolled face yet — run:  python facecam.py --enroll")
        return
    from ultralytics import YOLO
    faceid = FaceID()
    yolo = YOLO("yolo11n.pt")
    cap = cv2.VideoCapture(SOURCE)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = yolo(frame, classes=[PERSON_CLASS], conf=PERSON_CONF, verbose=False)[0]
        person_boxes = (result.boxes.xyxy.cpu().numpy().astype(int)
                        if result.boxes is not None else [])

        faces = [(box, is_admin(emb, reference, THRESHOLD))
                 for box, emb in faceid.embed_faces(frame)]

        for fbox, admin in faces:
            role, color = role_and_color(admin)
            _draw(frame, fbox, color, role)        # face box

        for pbox in person_boxes:                  # body hitbox, coloured by its face
            match = next((a for fb, a in faces if face_center_in_box(fb, pbox)), None)
            if match is None:
                _draw(frame, pbox, GREY, "?")
            else:
                role, color = role_and_color(match)
                _draw(frame, pbox, color, role)

        cv2.imshow("Face ID", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def enroll():
    faceid = FaceID()
    cap = cv2.VideoCapture(SOURCE)
    shots = []
    print(f"Enroll: aim your face, press SPACE to capture ({ENROLL_SHOTS} shots), q to stop.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = faceid.embed_faces(frame)
        for box, _ in faces:
            _draw(frame, box, YELLOW, f"{len(shots)}/{ENROLL_SHOTS}")
        cv2.imshow("Enroll - SPACE to capture, q to finish", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and faces:
            shots.append(faces[0][1])              # first face's embedding
            print(f"captured {len(shots)}/{ENROLL_SHOTS}")
            if len(shots) >= ENROLL_SHOTS:
                break
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    if shots:
        save_reference(np.array(shots), REF_PATH)
        print(f"✅ saved {len(shots)} reference shots to {REF_PATH}")
    else:
        print("No shots captured — nothing saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enroll", action="store_true", help="capture owner reference shots")
    args = parser.parse_args()
    enroll() if args.enroll else run()
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_facecam.py -q`
Expected: 3 PASS. (Importing `facecam` does NOT open a camera or load models — those happen only inside `run()`/`enroll()`.)

- [ ] **Step 5: Confirm the whole suite is green**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass, 1 deselected (the FaceID integration test).

- [ ] **Step 6: Commit**

```bash
git add facecam.py tests/test_facecam.py
git commit -m "feat: facecam live app + enroll mode + drawing"
```

---

## Notes for the implementer

- The live run is a manual human step (needs a real webcam + display, which a headless agent can't do). The implementer writes + commits the code and the pure tests; the user runs `python facecam.py --enroll` then `python facecam.py`.
- Dials in `facecam.py`: `THRESHOLD` (lower = stricter "is it me"), `ENROLL_SHOTS`, `TAG`.
- Branch `face-id-cam` is separate from the supercar work.
