from brain import Verdict
from catch import CatchTracker, CatchAction

def ex(conf, ident=False, label="Ferrari"):    # an exotic verdict
    return Verdict(label=label, confidence=conf, is_exotic=True, is_identified=ident)

def ordinary(conf=0.1):                          # a non-exotic verdict
    return Verdict(label="Ferrari", confidence=conf, is_exotic=False, is_identified=False)

def test_exotic_streak_required():
    t = CatchTracker(confirm_streak=2)
    assert t.update(1, ex(0.5)) is None
    assert t.update(1, ex(0.5)) == CatchAction(track_id=1, label="unidentified",
                                               confidence=0.5, tier="unidentified")

def test_identified_when_confident():
    t = CatchTracker(confirm_streak=2)
    assert t.update(1, ex(0.9, ident=True)) is None
    a = t.update(1, ex(0.9, ident=True))
    assert a.tier == "identified" and a.label == "Ferrari" and a.confidence == 0.9

def test_unidentified_when_exotic_but_not_named():
    t = CatchTracker(confirm_streak=1)
    a = t.update(1, ex(0.5, ident=False))
    assert a.tier == "unidentified" and a.label == "unidentified"

def test_non_exotic_resets_streak():
    t = CatchTracker(confirm_streak=2)
    assert t.update(1, ex(0.6)) is None            # streak 1
    assert t.update(1, ordinary()) is None          # reset
    assert t.update(1, ex(0.6)) is None             # streak 1 again, not caught

def test_unidentified_upgrades_to_identified():
    t = CatchTracker(confirm_streak=1)
    first = t.update(1, ex(0.5, ident=False))
    assert first.tier == "unidentified"
    upgraded = t.update(1, ex(0.8, ident=True))
    assert upgraded is not None and upgraded.tier == "identified" and upgraded.label == "Ferrari"

def test_cached_verdict_does_not_advance_streak():
    t = CatchTracker(confirm_streak=2)
    assert t.update(1, ex(0.6), fresh=True) is None
    assert t.update(1, ex(0.6), fresh=False) is None   # cached, no advance
    assert t.update(1, ex(0.6), fresh=True) is not None # 2nd fresh -> catch

def test_two_tracks_independent():
    t = CatchTracker(confirm_streak=1)
    assert t.update(1, ex(0.5)) is not None
    assert t.update(2, ex(0.5)) is not None

def test_reset_clears_state():
    t = CatchTracker(confirm_streak=1)
    t.update(1, ex(0.5))
    t.reset()
    assert t.update(1, ex(0.5)) is not None
