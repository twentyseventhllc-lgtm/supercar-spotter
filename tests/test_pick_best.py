from brain import pick_best, Verdict

BRANDS = ["Ferrari", "Lamborghini"]
NEGS = ["ordinary car", "sedan"]

def test_clear_supercar_is_caught():
    scores = {"Ferrari": 0.7, "Lamborghini": 0.05, "ordinary car": 0.1, "sedan": 0.05}
    v = pick_best(scores, BRANDS, NEGS, margin=0.15)
    assert v == Verdict(label="Ferrari", confidence=0.7, is_supercar=True)

def test_normie_car_not_caught():
    scores = {"Ferrari": 0.2, "ordinary car": 0.6}
    v = pick_best(scores, BRANDS, NEGS, margin=0.15)
    assert v.label == "Ferrari" and v.confidence == 0.2 and v.is_supercar is False

def test_within_margin_not_caught():
    scores = {"Lamborghini": 0.5, "ordinary car": 0.45}
    v = pick_best(scores, BRANDS, NEGS, margin=0.15)
    assert v.label == "Lamborghini" and v.is_supercar is False

def test_no_brand_in_scores():
    scores = {"ordinary car": 0.9}
    v = pick_best(scores, BRANDS, NEGS, margin=0.15)
    assert v == Verdict(label="", confidence=0.0, is_supercar=False)

def test_no_negatives_present_uses_zero_floor():
    scores = {"Ferrari": 0.3}
    v = pick_best(scores, BRANDS, NEGS, margin=0.15)
    assert v.is_supercar is True
