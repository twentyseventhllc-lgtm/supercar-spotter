from brain import Verdict
from catch import CatchTracker, CatchAction

def sc(conf):       # a supercar verdict
    return Verdict(label="Ferrari", confidence=conf, is_supercar=True)

def normie(conf):   # a non-supercar verdict
    return Verdict(label="Ferrari", confidence=conf, is_supercar=False)

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
