"""Count tracking decisions and optionally save replayable corner observations."""

import json
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REASON_LABELS = {
    "no_markers": "NO USABLE MARKERS",
    "invalid_corners": "INVALID CORNERS",
    "pose_rejected": "POSE REJECTED",
    "inconsistent_faces": "FACES DISAGREE",
    "tracking": "TRACKING",
}


class TrackingDiagnostics:
    def __init__(self, path: Path | None, metadata: dict):
        self.counts: Counter[str] = Counter()
        self.recent: deque[str] = deque(maxlen=60)
        self.stream = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Append sessions so restarting a diagnostic run preserves evidence.
            self.stream = path.open("a", encoding="utf-8", buffering=1)
            self._write(
                {
                    "type": "session",
                    "schema_version": 1,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    **metadata,
                }
            )

    def _write(self, record: dict) -> None:
        if self.stream is not None:
            self.stream.write(json.dumps(record, allow_nan=False) + "\n")

    def record(self, frame_index, timestamp, tracker, detections, pose) -> None:
        reason = tracker.last_reason
        self.counts[reason] += 1
        self.recent.append(reason)
        if self.stream is None:
            return
        self._write(
            {
                "type": "frame",
                "frame": frame_index,
                "elapsed_seconds": timestamp,
                "reason": reason,
                "decoded_ids": list(tracker.detected_ids),
                "duplicate_ids": list(tracker.duplicate_ids),
                "rejected_candidate_count": tracker.rejected_candidate_count,
                "usable_ids": sorted(detections),
                "corners_px": {str(i): np.asarray(points).tolist() for i, points in detections.items()},
                "shortest_edge_px": {
                    str(i): float(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1).min())
                    for i, points in detections.items()
                },
                "used_ids": list(pose.marker_ids) if pose else [],
                "reprojection_error_px": pose.reprojection_error_px if pose else None,
            }
        )

    def recent_summary(self) -> str:
        counts = Counter(self.recent)
        return (
            f"Last {len(self.recent)} frames: OK {counts['tracking']} | "
            f"NO ID {counts['no_markers']} | POSE {counts['pose_rejected']} | "
            f"CONFLICT {counts['inconsistent_faces']} | CORNERS {counts['invalid_corners']}"
        )

    def summary(self) -> str:
        total = self.counts.total()
        lost = total - self.counts["tracking"]
        percentage = 100 * lost / total if total else 0.0
        reasons = ", ".join(f"{reason}={count}" for reason, count in sorted(self.counts.items()))
        return f"Diagnostics: LOST {lost}/{total} ({percentage:.1f}%); {reasons}"

    def close(self) -> None:
        if self.stream is not None:
            self._write({"type": "summary", "total_frames": self.counts.total(), "counts": dict(self.counts)})
            self.stream.close()
            self.stream = None
