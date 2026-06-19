from dataclasses import dataclass
from brain import Verdict


@dataclass
class CatchAction:
    track_id: int
    label: str
    confidence: float


class CatchTracker:
    def __init__(self):
        self._best: dict[int, float] = {}

    def update(self, track_id: int, verdict: Verdict) -> CatchAction | None:
        if not verdict.is_supercar:
            return None
        prev = self._best.get(track_id)
        if prev is not None and verdict.confidence <= prev:
            return None
        self._best[track_id] = verdict.confidence
        return CatchAction(track_id=track_id, label=verdict.label, confidence=verdict.confidence)

    def reset(self) -> None:
        self._best.clear()
