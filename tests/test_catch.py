from brain import Verdict
from catch import CatchTracker, CatchAction

def sc(conf):       # a supercar verdict
    return Verdict(label="Ferrari", confidence=conf, is_supercar=True)

def normie(conf):   # a non-supercar verdict
    return Verdict(label="Ferrari", confidence=conf, is_supercar=False)

def test_short_track_catches_at_streak_2_not_3():
    # Regression: on an unstable feed YOLO fragments a car into short-lived track
    # ids. A track that gets only 2 fresh agreeing supercar verdicts before it is
    # re-id'd NEVER catches at confirm_streak=3 (the "detects nothing" bug) but
    # does at confirm_streak=2 — which is why the live default was relaxed to 2.
    strict = CatchTracker(confirm_streak=3)
    assert strict.update(1, sc(0.9)) is None
    assert strict.update(1, sc(0.9)) is None      # only 2 checks -> never caught at 3

    relaxed = CatchTracker(confirm_streak=2)
    assert relaxed.update(1, sc(0.9)) is None
    assert relaxed.update(1, sc(0.9)) is not None  # caught on the 2nd check


def test_normie_never_catches():
    t = CatchTracker()
    assert t.update(1, normie(0.9)) is None

def test_first_supercar_frame_catches():
    t = CatchTracker()
    assert t.update(1, sc(0.6)) == CatchAction(track_id=1, label="Ferrari", confidence=0.6)

def test_higher_confidence_same_track_recatches():
    t = CatchTracker()
    t.update(1, sc(0.6))
    assert t.update(1, sc(0.8)) == CatchAction(track_id=1, label="Ferrari", confidence=0.8)

def test_lower_confidence_same_track_is_ignored():
    t = CatchTracker()
    t.update(1, sc(0.8))
    assert t.update(1, sc(0.5)) is None

def test_two_tracks_are_independent():
    t = CatchTracker()
    assert t.update(1, sc(0.6)) is not None
    assert t.update(2, sc(0.6)) is not None

def test_reset_clears_state():
    t = CatchTracker()
    t.update(1, sc(0.6))
    t.reset()
    assert t.update(1, sc(0.6)) is not None  # track 1 is fresh again


def test_streak_required_before_catching():
    t = CatchTracker(confirm_streak=3)
    assert t.update(1, sc(0.6)) is None          # streak 1
    assert t.update(1, sc(0.7)) is None           # streak 2
    assert t.update(1, sc(0.8)) == CatchAction(track_id=1, label="Ferrari", confidence=0.8)


def test_different_brand_resets_streak():
    t = CatchTracker(confirm_streak=2)
    other = Verdict(label="Lamborghini", confidence=0.9, is_supercar=True)
    assert t.update(1, sc(0.7)) is None           # Ferrari streak 1
    assert t.update(1, other) is None             # switch -> Lambo streak 1 (reset)
    assert t.update(1, other) == CatchAction(track_id=1, label="Lamborghini", confidence=0.9)


def test_non_supercar_resets_streak():
    t = CatchTracker(confirm_streak=2)
    assert t.update(1, sc(0.7)) is None           # streak 1
    assert t.update(1, normie(0.9)) is None       # reset
    assert t.update(1, sc(0.7)) is None           # streak 1 again, not caught


def test_cached_verdict_does_not_advance_streak():
    t = CatchTracker(confirm_streak=3)
    assert t.update(1, sc(0.6), fresh=True) is None   # streak 1
    assert t.update(1, sc(0.6), fresh=False) is None  # cached, no advance
    assert t.update(1, sc(0.6), fresh=False) is None  # cached, no advance
    assert t.update(1, sc(0.7), fresh=True) is None   # streak 2
    assert t.update(1, sc(0.8), fresh=True) is not None  # streak 3 -> catch
