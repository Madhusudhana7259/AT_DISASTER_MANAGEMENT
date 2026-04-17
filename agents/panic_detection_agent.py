import numpy as np
from collections import deque


class PanicDetectionAgent:
    def __init__(
        self,
        flow_threshold=1.2,
        speed_threshold=6.0,
        slow_baseline_threshold=2.5,
        run_speed_threshold=6.0,
        spike_delta=3.0,
        spike_ratio=2.0,
        baseline_window=20,
        spike_persist_frames=4,
        stale_track_frames=60,
    ):
        self.flow_threshold = flow_threshold
        self.speed_threshold = speed_threshold
        self.slow_baseline_threshold = slow_baseline_threshold
        self.run_speed_threshold = run_speed_threshold
        self.spike_delta = spike_delta
        self.spike_ratio = spike_ratio
        self.baseline_window = baseline_window
        self.spike_persist_frames = spike_persist_frames
        self.stale_track_frames = stale_track_frames

        self.speed_history = {}
        self.spike_streak = {}
        self.track_last_seen = {}

    def _cleanup_stale_tracks(self, frame_id):
        stale = [
            tid
            for tid, last_seen in self.track_last_seen.items()
            if (frame_id - last_seen) > self.stale_track_frames
        ]
        for tid in stale:
            self.track_last_seen.pop(tid, None)
            self.speed_history.pop(tid, None)
            self.spike_streak.pop(tid, None)

    def process(self, scene_state):
        frame_id = int(scene_state.frame_id)
        people = [obj for obj in scene_state.objects if obj.get("class") == "person"]
        speeds = [obj.get("speed", 0.0) for obj in people]

        sudden_run_ids = []
        for person in people:
            track_id = person["track_id"]
            speed = float(person.get("speed", 0.0))

            if track_id not in self.speed_history:
                self.speed_history[track_id] = deque(maxlen=self.baseline_window)
            if track_id not in self.spike_streak:
                self.spike_streak[track_id] = 0

            history = self.speed_history[track_id]
            prior_speeds = list(history)
            baseline = float(np.mean(prior_speeds)) if prior_speeds else 0.0

            # Trigger sudden-run only when a previously slow person accelerates strongly.
            spike = (
                baseline <= self.slow_baseline_threshold
                and speed >= self.run_speed_threshold
                and (speed - baseline) >= self.spike_delta
                and (speed >= max(self.run_speed_threshold, baseline * self.spike_ratio))
            )

            if spike:
                self.spike_streak[track_id] += 1
            else:
                self.spike_streak[track_id] = max(0, self.spike_streak[track_id] - 1)

            if self.spike_streak[track_id] >= self.spike_persist_frames:
                sudden_run_ids.append(track_id)

            history.append(speed)
            self.track_last_seen[track_id] = frame_id

        self._cleanup_stale_tracks(frame_id)

        mean_speed = float(np.mean(speeds)) if speeds else 0.0
        flow_mean = float(scene_state.global_motion.get("mean_magnitude", 0.0))

        flow_signal = min(flow_mean / self.flow_threshold, 1.0) if self.flow_threshold > 0 else 0.0
        speed_signal = min(mean_speed / self.speed_threshold, 1.0) if self.speed_threshold > 0 else 0.0

        sudden_run_score = 1.0 if sudden_run_ids else 0.0
        score = 0.45 * flow_signal + 0.25 * speed_signal + 0.30 * sudden_run_score
        panic = score >= 0.65

        return {
            "panic": panic,
            "score": float(score),
            "mean_speed": mean_speed,
            "flow_mean": flow_mean,
            "sudden_run": len(sudden_run_ids) > 0,
            "sudden_run_count": len(sudden_run_ids),
            "sudden_run_ids": sudden_run_ids,
        }
