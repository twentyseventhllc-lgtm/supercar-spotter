import numpy as np

import dashboard

EXPECTED_H = dashboard.HUD_H + dashboard.PANE_H + dashboard.GALLERY_H


def test_compose_has_full_dashboard_height_empty_gallery():
    raw = np.zeros((720, 1280, 3), np.uint8)
    annotated = np.zeros((720, 1280, 3), np.uint8)
    info = {"mode": "supercars", "catches": 0, "fps": 12.0, "last": None}
    comp = dashboard._compose(raw, annotated, info, [])
    assert comp.shape[0] == EXPECTED_H
    assert comp.shape[2] == 3


def test_compose_bars_match_middle_width_with_thumbs():
    # vstack requires the HUD, middle row, and gallery to share one width — this
    # would raise if a bar were built at a different width than the feed row.
    raw = np.zeros((480, 640, 3), np.uint8)
    annotated = np.zeros((480, 640, 3), np.uint8)
    info = {"mode": "all", "catches": 3, "fps": 20.0, "last": "Ferrari"}
    thumbs = [(np.zeros((100, 150, 3), np.uint8), "Ferrari")]
    comp = dashboard._compose(raw, annotated, info, thumbs)
    assert comp.shape[0] == EXPECTED_H
    assert comp.shape[1] > 0


def test_modes_cover_all_number_keys():
    assert set(dashboard.MODE_KEYS.values()) == set(dashboard.MODES)
