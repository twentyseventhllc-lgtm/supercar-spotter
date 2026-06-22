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
