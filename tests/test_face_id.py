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
