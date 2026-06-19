from dataclasses import dataclass
from brain import Verdict


@dataclass
class CatchAction:
    track_id: int
    label: str
    confidence: float


class CatchTracker:
    def __init__(self, confirm_streak: int = 1):
        self._best: dict[int, float] = {}
        self._streak: dict[int, tuple] = {}   # track_id -> (label, count)
        self.confirm_streak = confirm_streak

    def update(self, track_id: int, verdict: Verdict,
               fresh: bool = True) -> CatchAction | None:
        if fresh:
            if verdict.is_supercar:
                label, count = self._streak.get(track_id, (None, 0))
                count = count + 1 if label == verdict.label else 1
                self._streak[track_id] = (verdict.label, count)
            else:
                self._streak[track_id] = (None, 0)   # non-supercar breaks the streak

        label, count = self._streak.get(track_id, (None, 0))
        confirmed = (verdict.is_supercar
                     and label == verdict.label
                     and count >= self.confirm_streak)
        if not confirmed:
            return None

        prev = self._best.get(track_id)
        if prev is not None and verdict.confidence <= prev:
            return None
        self._best[track_id] = verdict.confidence
        return CatchAction(track_id=track_id, label=verdict.label,
                           confidence=verdict.confidence)

    def reset(self) -> None:
        self._best.clear()
        self._streak.clear()
