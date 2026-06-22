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


class FaceID:
    """Detect faces (MTCNN) and embed them (FaceNet). Lazy-imports heavy deps."""

    def __init__(self):
        import torch
        from facenet_pytorch import MTCNN, InceptionResnetV1

        self._torch = torch
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            # MPS (Apple Silicon) is skipped: MTCNN uses adaptive_avg_pool2d with
            # non-divisible sizes that MPS does not yet support (pytorch #96056).
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
