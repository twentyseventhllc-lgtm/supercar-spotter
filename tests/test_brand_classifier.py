import numpy as np
import pytest

pytestmark = pytest.mark.integration  # downloads CLIP weights; run with -m integration

def test_score_returns_normalized_distribution_over_labels():
    from brain import BrandClassifier
    labels = ["a sports car", "a tree", "a sandwich"]
    clf = BrandClassifier(labels=labels, template="{}")
    crop = np.zeros((120, 200, 3), dtype=np.uint8)  # any valid BGR image
    scores = clf.score(crop)
    assert set(scores.keys()) == set(labels)
    assert all(0.0 <= v <= 1.0 for v in scores.values())
    assert abs(sum(scores.values()) - 1.0) < 1e-3
