# Face-ID Webcam — Design

**Date:** 2026-06-20
**Status:** Approved (design), pending implementation plan

## Goal

A live webcam app that detects each person, decides whether it's the owner ("me"),
and labels them: **you → yellow** boxes tagged `WW · admin`; **anyone else → blue**
boxes tagged `WW · unknown`.

A fun, separate app from the Supercar Spotter (new branch `face-id-cam`), reusing
the same repo/venv (YOLO + torch + MPS).

## What it draws

For every detected person, two boxes (the "hitbox"):
- a **body hitbox** (whole person, from YOLO)
- a **face box** inside it (from the face detector)

Both coloured by identity, with `WW` on top and `admin` / `unknown` underneath.
- **You** → yellow.
- **Anyone else** → blue.

## How it decides "is it me"

1. **Detect** — YOLO finds each person → body hitbox. A face detector (MTCNN) finds
   the face within the person crop → face box.
2. **Recognize** — a face-embedding model (FaceNet, pretrained on VGGFace2) turns the
   face into a 512-d "fingerprint" vector. Compare (distance) to the enrolled
   fingerprint(s) of the owner. Below a threshold → **admin (you)**; else **unknown**.
3. **Draw** — yellow (admin) / blue (unknown) body + face boxes, `WW` tag + the role.

Multiple people in frame are judged independently (you yellow, the rest blue).

## Enrollment (one-time)

`python facecam.py --enroll` opens the webcam; press **SPACE** to snap ~5 reference
shots of your face; it computes + saves their fingerprints to `me/embeddings.npy`.
Re-runnable to add more shots (more = more reliable). The live app loads these on
start; if none exist it tells you to enroll first.

## Tech choice

**`facenet-pytorch`** (MTCNN face detection + InceptionResnetV1 / FaceNet
embeddings) — pip-installable, runs on the existing torch/MPS stack, no dlib
compilation. YOLO (already present) provides the body hitbox.

Rejected: `face_recognition` (dlib) — painful compile on macOS; `insightface` —
heavier, separate ONNX runtime.

## Components (small, isolated, testable)

### `face_id.py` — `FaceID`
- **Does:** detect+embed faces, enroll/save/load the owner's reference fingerprints,
  and judge a face embedding as admin-or-not.
- **Interface:**
  - `embed_faces(frame_bgr) -> list[(box, embedding)]` — MTCNN detect + FaceNet embed.
  - `enroll(embeddings, path)` / `load(path) -> reference` — save/load owner refs.
  - `is_admin(embedding, reference, threshold) -> bool` — pure distance check
    (min distance to any reference embedding < threshold). **This is the unit-tested
    core** (no model needed — feed vectors).
- **Depends on:** torch, facenet-pytorch (lazy-imported), numpy.

### `facecam.py` — main loop + enroll mode + drawing
- **Does:** webcam capture; YOLO person detection; per person, face detect+recognize;
  draw body + face boxes (yellow/blue) with the `WW` / role labels; live window.
  `--enroll` runs the capture-reference flow instead.
- **Config block at top** (same friendly style as `dashboard.py`): `SOURCE` (camera),
  `THRESHOLD` (match strictness), `TAG = "WW"`, colours, `ENROLL_SHOTS`.
- **Depends on:** ultralytics (YOLO), cv2, `face_id.py`.

### `me/` — saved owner fingerprints (`embeddings.npy`); gitignored.

## Testing

- `is_admin` distance/threshold logic — pure unit tests with synthetic vectors
  (a near vector → admin; a far vector → unknown; empty reference → unknown).
- Enroll save/load round-trip — unit test (write then load embeddings).
- Face detect/embed + the live loop — manual (needs a real webcam/face); a small
  smoke test that `facecam.py` imports without loading models.

## Honest notes / out of scope

- Webcam face-rec is good but not flawless — lighting/angle/threshold matter; more
  enrollment shots improve it.
- No anti-spoofing (a photo of you could pass) — out of scope for a fun build.
- Privacy: this stores only the owner's own face fingerprints locally; no upload.
