from dataclasses import dataclass
from brain import Verdict


@dataclass
class CatchAction:
    track_id: int
    label: str
    confidence: float
    tier: str   # "identified" or "unidentified"


class CatchTracker:
    def __init__(self, confirm_streak: int = 1):
        self.confirm_streak = confirm_streak
        self._tracks: dict[int, dict] = {}

    def update(self, track_id: int, verdict: Verdict,
               fresh: bool = True) -> CatchAction | None:
        st = self._tracks.get(track_id)
        if st is None:
            st = {"count": 0, "best_conf": 0.0, "brand": None, "brand_conf": 0.0,
                  "emitted_conf": None, "emitted_tier": None}
            self._tracks[track_id] = st

        if fresh:
            if verdict.is_exotic:
                st["count"] += 1
                st["best_conf"] = max(st["best_conf"], verdict.confidence)
                if verdict.is_identified and verdict.confidence > st["brand_conf"]:
                    st["brand"] = verdict.label
                    st["brand_conf"] = verdict.confidence
            else:
                st["count"] = 0

        if st["count"] < self.confirm_streak:
            return None

        if st["brand"] is not None:
            tier, label, conf = "identified", st["brand"], st["brand_conf"]
        else:
            tier, label, conf = "unidentified", "unidentified", st["best_conf"]

        upgraded = st["emitted_tier"] == "unidentified" and tier == "identified"
        if st["emitted_conf"] is not None and conf <= st["emitted_conf"] and not upgraded:
            return None
        st["emitted_conf"] = conf
        st["emitted_tier"] = tier
        return CatchAction(track_id=track_id, label=label, confidence=conf, tier=tier)

    def reset(self) -> None:
        self._tracks.clear()
