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
