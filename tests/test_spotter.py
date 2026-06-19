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

def test_save_catch_creates_jpg_with_correct_name(tmp_path):
    """save_catch writes exactly one .jpg whose name contains track7 and Ferrari."""
    action = CatchAction(track_id=7, label="Ferrari", confidence=0.83)
    frame = np.zeros((20, 30, 3), dtype=np.uint8)

    spotter.save_catch(str(tmp_path), frame, action)

    jpgs = glob.glob(str(tmp_path / "*.jpg"))
    assert len(jpgs) == 1, f"Expected 1 jpg, found: {jpgs}"
    name = os.path.basename(jpgs[0])
    assert "track7" in name, f"Expected 'track7' in filename: {name}"
    assert "Ferrari" in name, f"Expected 'Ferrari' in filename: {name}"


def test_save_catch_writes_log_csv(tmp_path):
    """save_catch appends a row to log.csv containing label, track_id, and path."""
    action = CatchAction(track_id=7, label="Ferrari", confidence=0.83)
    frame = np.zeros((20, 30, 3), dtype=np.uint8)

    spotter.save_catch(str(tmp_path), frame, action)

    log_path = tmp_path / "log.csv"
    assert log_path.exists(), "log.csv was not created"

    with open(log_path, newline="") as fh:
        rows = list(csv.reader(fh))

    assert len(rows) == 1, f"Expected 1 CSV row, got {len(rows)}"
    row = rows[0]
    row_str = ",".join(row)
    assert "Ferrari" in row_str, f"'Ferrari' missing from CSV row: {row}"
    assert "7" in row_str, f"track_id '7' missing from CSV row: {row}"
    # The path column (last) should reference the jpg that was written
    assert row[-1].endswith(".jpg"), f"Last column should be a .jpg path: {row}"
