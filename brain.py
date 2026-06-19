# NOTE: keep torch / open_clip imports OUT of module top level (see Task 4).
from dataclasses import dataclass


@dataclass
class Verdict:
    label: str
    confidence: float
    is_supercar: bool


def pick_best(scores, brands, negatives, margin):
    brand_scores = {label: scores[label] for label in brands if label in scores}
    if not brand_scores:
        return Verdict(label="", confidence=0.0, is_supercar=False)

    best_label = max(brand_scores, key=brand_scores.get)
    best_brand = brand_scores[best_label]

    neg_scores = [scores[label] for label in negatives if label in scores]
    best_neg = max(neg_scores) if neg_scores else 0.0

    is_supercar = (best_brand - best_neg) >= margin
    return Verdict(label=best_label, confidence=best_brand, is_supercar=is_supercar)
