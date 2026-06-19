"""
Unit tests for spotter.py's two pure helper functions:
  - list_sources(source)
  - save_catch(out_dir, frame, action)

These tests do NOT import YOLO, CLIP, or open any GUI windows.
`import spotter` is safe because models only load inside run().
"""
import csv
import glob
import os

import numpy as np
import pytest

import spotter
from catch import CatchAction


# ─── list_sources ────────────────────────────────────────────

def test_list_sources_webcam_int():
    """Integer 0 (webcam) passes through as-is."""
    assert spotter.list_sources(0) == [0]


def test_list_sources_nondir_path():
    """Non-directory string paths pass through unchanged."""
    assert spotter.list_sources("some/nonexistent.mp4") == ["some/nonexistent.mp4"]


def test_list_sources_directory_with_videos(tmp_path):
    """Directory: returns sorted video files, excludes non-video files."""
    (tmp_path / "a.mp4").touch()
    (tmp_path / "b.mov").touch()
    (tmp_path / "notes.txt").touch()

    result = spotter.list_sources(str(tmp_path))
    basenames = [os.path.basename(p) for p in result]
    assert basenames == ["a.mp4", "b.mov"]


def test_list_sources_empty_directory(tmp_path):
    """Empty directory returns empty list."""
    result = spotter.list_sources(str(tmp_path))
    assert result == []


# ─── save_catch ──────────────────────────────────────────────

def test_save_catch_names_file_and_logs_tier(tmp_path):
    import numpy as np
    from catch import CatchAction
    import spotter
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    action = CatchAction(track_id=7, label="Ferrari", confidence=0.83, tier="identified")
    spotter.save_catch(str(tmp_path), frame, action)
    jpgs = list(tmp_path.glob("*.jpg"))
    assert len(jpgs) == 1 and "track7" in jpgs[0].name and "Ferrari" in jpgs[0].name
    row = (tmp_path / "log.csv").read_text().strip().split(",")
    assert "identified" in row and "Ferrari" in row and "7" in row

def test_save_all_car_writes_to_all_subfolder(tmp_path):
    import numpy as np
    import spotter
    crop = np.zeros((20, 30, 3), dtype=np.uint8)
    path = spotter.save_all_car(str(tmp_path), crop, track_id=12)
    assert (tmp_path / "all").is_dir()
    assert path.endswith(".jpg") and "track12" in path and "_car" in path
    assert len(list((tmp_path / "all").glob("*.jpg"))) == 1
