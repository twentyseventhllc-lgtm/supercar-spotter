import notify
from catch import CatchAction


def test_build_request_targets_topic_with_image_body_and_headers():
    req = notify._build_request("mytopic", "Ferrari 90%", "caught one!",
                                b"\xff\xd8jpegbytes", "shot.jpg",
                                server="https://ntfy.sh")
    assert req.full_url == "https://ntfy.sh/mytopic"
    assert req.method == "PUT"
    assert req.data == b"\xff\xd8jpegbytes"
    assert req.get_header("Title") == "Ferrari 90%"
    assert req.get_header("Filename") == "shot.jpg"
    assert req.get_header("Message") == "caught one!"


def test_build_request_strips_non_ascii_from_headers():
    # An emoji in the title must not survive into the (latin-1) header.
    req = notify._build_request("t", "Ferrari 90% 🏎️", "caught 🎉", b"x", "a.jpg")
    title = req.get_header("Title")
    assert title == "Ferrari 90% "
    title.encode("latin-1")  # would raise if a non-ascii char slipped through


def test_notify_catch_pushes_once_per_track(monkeypatch):
    notify._notified.clear()
    monkeypatch.setattr(notify, "_last_push_ts", None)
    monkeypatch.setattr(notify, "ENABLED", True)
    monkeypatch.setattr(notify, "NTFY_TOPIC", "t")
    monkeypatch.setattr(notify, "NOTIFY_COOLDOWN", 0)   # isolate the per-track dedup
    sent = []
    monkeypatch.setattr(notify, "ntfy_photo",
                        lambda topic, title, msg, path: sent.append((title, path)))

    action = CatchAction(track_id=7, label="Lamborghini", confidence=0.94, tier="identified")
    assert notify.notify_catch(action, "catches/a.jpg") is True
    # same track again (e.g. a higher-confidence frame) must NOT push again
    better = CatchAction(track_id=7, label="Lamborghini", confidence=0.99, tier="identified")
    assert notify.notify_catch(better, "catches/b.jpg") is False
    # a different car does push
    other = CatchAction(track_id=8, label="Porsche", confidence=0.80, tier="identified")
    assert notify.notify_catch(other, "catches/c.jpg") is True

    assert sent == [("Lamborghini 94%", "catches/a.jpg"),
                    ("Porsche 80%", "catches/c.jpg")]


def test_notify_catch_rate_limited_by_cooldown(monkeypatch):
    # The flood-protection: at most one push per NOTIFY_COOLDOWN seconds. A car
    # blocked by the cooldown is NOT marked done, so it can still push once the
    # window clears (we don't lose it forever).
    notify._notified.clear()
    monkeypatch.setattr(notify, "_last_push_ts", None)
    monkeypatch.setattr(notify, "ENABLED", True)
    monkeypatch.setattr(notify, "NTFY_TOPIC", "t")
    monkeypatch.setattr(notify, "NOTIFY_COOLDOWN", 60)
    sent = []
    monkeypatch.setattr(notify, "ntfy_photo",
                        lambda topic, title, msg, path: sent.append(title))

    first = CatchAction(track_id=7, label="Lamborghini", confidence=0.94, tier="identified")
    later = CatchAction(track_id=8, label="Porsche", confidence=0.80, tier="identified")
    assert notify.notify_catch(first, "a.jpg", now=0) is True     # first push
    assert notify.notify_catch(later, "b.jpg", now=10) is False   # within cooldown -> dropped
    assert notify.notify_catch(later, "b.jpg", now=70) is True    # cooldown cleared -> sends

    assert sent == ["Lamborghini 94%", "Porsche 80%"]


def test_shrink_for_push_reduces_dimensions_and_size(tmp_path):
    # Full-res catch frames trigger ntfy's 413; the push image must be downscaled.
    import cv2
    import numpy as np
    big = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    path = tmp_path / "big.jpg"
    cv2.imwrite(str(path), big)
    raw = path.read_bytes()

    out = notify._shrink_for_push(str(path), max_side=800, quality=70)
    decoded = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
    assert max(decoded.shape[:2]) <= 800       # downscaled
    assert len(out) < len(raw)                 # smaller payload


def test_notify_catch_noop_when_disabled(monkeypatch):
    notify._notified.clear()
    monkeypatch.setattr(notify, "ENABLED", False)
    calls = []
    monkeypatch.setattr(notify, "ntfy_photo",
                        lambda *a, **k: calls.append(a))
    action = CatchAction(track_id=1, label="Ferrari", confidence=0.9, tier="identified")
    assert notify.notify_catch(action, "catches/x.jpg") is False
    assert calls == []


def test_notify_catch_noop_when_no_image(monkeypatch):
    notify._notified.clear()
    monkeypatch.setattr(notify, "ENABLED", True)
    monkeypatch.setattr(notify, "NTFY_TOPIC", "t")
    calls = []
    monkeypatch.setattr(notify, "ntfy_photo", lambda *a, **k: calls.append(a))
    action = CatchAction(track_id=1, label="Ferrari", confidence=0.9, tier="identified")
    assert notify.notify_catch(action, None) is False
    assert calls == []
