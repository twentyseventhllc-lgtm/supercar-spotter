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
