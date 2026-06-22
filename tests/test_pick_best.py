from brain import pick_best, Verdict

BRANDS = ["Ferrari", "Lamborghini"]
NEGS = ["ordinary car", "sedan"]

def test_confident_brand_is_identified():
    scores = {"Ferrari": 0.8, "ordinary car": 0.1}
    v = pick_best(scores, BRANDS, NEGS, margin=0.15, identify_floor=0.70)
    assert v == Verdict(label="Ferrari", confidence=0.8, is_exotic=True, is_identified=True)

def test_exotic_but_below_floor_is_unidentified():
    # Looks clearly more exotic than ordinary, but not confident enough to name.
    scores = {"Ferrari": 0.55, "ordinary car": 0.1}
    v = pick_best(scores, BRANDS, NEGS, margin=0.15, identify_floor=0.70)
    assert v.is_exotic is True and v.is_identified is False and v.label == "Ferrari"

def test_ordinary_is_neither():
    scores = {"Ferrari": 0.2, "ordinary car": 0.6}
    v = pick_best(scores, BRANDS, NEGS, margin=0.15, identify_floor=0.70)
    assert v.is_exotic is False and v.is_identified is False

def test_within_margin_not_exotic():
    scores = {"Lamborghini": 0.5, "ordinary car": 0.45}
    v = pick_best(scores, BRANDS, NEGS, margin=0.15, identify_floor=0.70)
    assert v.is_exotic is False

def test_no_brand_in_scores():
    v = pick_best({"ordinary car": 0.9}, BRANDS, NEGS, margin=0.15, identify_floor=0.70)
    assert v == Verdict(label="", confidence=0.0, is_exotic=False, is_identified=False)

def test_default_floor_zero_identifies_any_exotic():
    scores = {"Ferrari": 0.3, "ordinary car": 0.05}
    v = pick_best(scores, BRANDS, NEGS, margin=0.15)
    assert v.is_exotic is True and v.is_identified is True
