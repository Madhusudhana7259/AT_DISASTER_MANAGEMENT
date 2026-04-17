import math


class SuspiciousObjectAgent:
    def __init__(
        self,
        owner_attach_distance_px=220,
        owner_keep_distance_px=240,
        unattended_distance_px=240,
        unattended_frames_threshold=12,
        person_missing_frames_threshold=25,
        reassign_frames_threshold=6,
        stale_bag_frames=180,
        id_switch_match_distance_px=80,
        id_switch_match_frames=20,
        static_motion_threshold_px=8,
        static_frames_threshold=20,
        no_person_frames_threshold=20,
        carried_distance_px=20,
    ):
        self.owner_attach_distance_px = owner_attach_distance_px
        self.owner_keep_distance_px = owner_keep_distance_px
        self.unattended_distance_px = unattended_distance_px
        self.unattended_frames_threshold = unattended_frames_threshold
        self.person_missing_frames_threshold = person_missing_frames_threshold
        self.reassign_frames_threshold = reassign_frames_threshold
        self.stale_bag_frames = stale_bag_frames
        self.id_switch_match_distance_px = id_switch_match_distance_px
        self.id_switch_match_frames = id_switch_match_frames
        self.static_motion_threshold_px = static_motion_threshold_px
        self.static_frames_threshold = static_frames_threshold
        self.no_person_frames_threshold = no_person_frames_threshold
        self.carried_distance_px = carried_distance_px
        self.suspicious_classes = {"backpack", "handbag", "suitcase", "bag"}
        self.person_last_seen = {}
        self.bag_states = {}

    @staticmethod
    def _center(bbox):
        x1, y1, x2, y2 = bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    @staticmethod
    def _distance(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    @staticmethod
    def _safe_float(value):
        if value is None:
            return None
        value = float(value)
        if not math.isfinite(value):
            return None
        return value

    @staticmethod
    def _point_to_bbox_distance(point, bbox):
        px, py = point
        x1, y1, x2, y2 = bbox

        dx = max(x1 - px, 0, px - x2)
        dy = max(y1 - py, 0, py - y2)
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def _iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def _build_bag_candidates(self, scene_state):
        tracked_bags = [
            obj for obj in scene_state.objects if obj.get("class") in self.suspicious_classes
        ]
        candidates = [dict(obj) for obj in tracked_bags]

        detection_bags = [
            det for det in getattr(scene_state, "detections", [])
            if det.get("class") in self.suspicious_classes
        ]
        frame_id = int(scene_state.frame_id)

        for idx, det in enumerate(detection_bags):
            dbbox = det["bbox"]
            overlaps_tracked = any(
                self._iou(dbbox, tb["bbox"]) >= 0.3 for tb in tracked_bags
            )
            if overlaps_tracked:
                continue

            candidates.append(
                {
                    "track_id": f"det_{frame_id}_{idx}",
                    "class": det["class"],
                    "bbox": dbbox,
                }
            )

        return candidates

    def _nearest_person(self, bag_center, person_centers_by_id, exclude_id=None):
        candidates = [
            (pid, self._distance(bag_center, pcenter))
            for pid, pcenter in person_centers_by_id.items()
            if pid != exclude_id
        ]
        if not candidates:
            return None, float("inf")
        return min(candidates, key=lambda x: x[1])

    def _nearest_person_to_bag(self, bag_bbox, person_bboxes_by_id, exclude_id=None):
        bag_center = self._center(bag_bbox)
        candidates = []
        for pid, pbbox in person_bboxes_by_id.items():
            if pid == exclude_id:
                continue
            # More robust than center-to-center for person-bag relation.
            dist = self._point_to_bbox_distance(bag_center, pbbox)
            candidates.append((pid, dist))

        if not candidates:
            return None, float("inf")
        return min(candidates, key=lambda x: x[1])

    def _get_bag_state(self, bag_id, frame_id):
        if bag_id not in self.bag_states:
            self.bag_states[bag_id] = {
                "owner_id": None,
                "unattended_frames": 0,
                "last_seen_frame": frame_id,
                "candidate_owner_id": None,
                "candidate_owner_frames": 0,
                "last_center": None,
                "class": None,
                "static_frames": 0,
                "no_person_frames": 0,
            }
        return self.bag_states[bag_id]

    def _match_id_switched_bag(self, bag_id, bag_class, bag_center, frame_id):
        if bag_id in self.bag_states:
            return bag_id

        best_id = None
        best_dist = float("inf")

        for sid, state in self.bag_states.items():
            if state.get("class") != bag_class:
                continue

            if (frame_id - state.get("last_seen_frame", -10**9)) > self.id_switch_match_frames:
                continue

            prev_center = state.get("last_center")
            if prev_center is None:
                continue

            dist = self._distance(prev_center, bag_center)
            if dist <= self.id_switch_match_distance_px and dist < best_dist:
                best_dist = dist
                best_id = sid

        if best_id is None:
            return bag_id

        # Transfer state to new track ID when DeepSORT flips bag ID.
        self.bag_states[bag_id] = self.bag_states.pop(best_id)
        return bag_id

    def _cleanup_stale_bags(self, frame_id):
        stale_ids = [
            bag_id
            for bag_id, state in self.bag_states.items()
            if (frame_id - state["last_seen_frame"]) > self.stale_bag_frames
        ]
        for bag_id in stale_ids:
            del self.bag_states[bag_id]

    def process(self, scene_state):
        frame_id = int(scene_state.frame_id)
        persons = [obj for obj in scene_state.objects if obj.get("class") == "person"]
        bags = self._build_bag_candidates(scene_state)

        person_centers_by_id = {
            obj["track_id"]: self._center(obj["bbox"]) for obj in persons
        }
        person_bboxes_by_id = {
            obj["track_id"]: obj["bbox"] for obj in persons
        }
        for pid in person_centers_by_id:
            self.person_last_seen[pid] = frame_id

        suspicious_count = 0
        unattended_bag_ids = []
        unattended_bboxes = []
        nearest_distances = []
        debug_bags = []

        for bag in bags:
            raw_bag_id = bag["track_id"]
            bag_class = bag["class"]
            bcenter = self._center(bag["bbox"])
            bag_id = self._match_id_switched_bag(
                bag_id=raw_bag_id,
                bag_class=bag_class,
                bag_center=bcenter,
                frame_id=frame_id,
            )
            state = self._get_bag_state(bag_id, frame_id)
            state["last_seen_frame"] = frame_id
            prev_center = state["last_center"]
            if prev_center is None:
                state["static_frames"] = 0
            else:
                shift = self._distance(prev_center, bcenter)
                if shift <= self.static_motion_threshold_px:
                    state["static_frames"] += 1
                else:
                    state["static_frames"] = 0
            state["last_center"] = bcenter
            state["class"] = bag_class

            owner_id = state["owner_id"]
            nearest_any_pid, nearest_any_dist = self._nearest_person_to_bag(
                bag_bbox=bag["bbox"],
                person_bboxes_by_id=person_bboxes_by_id,
            )
            if nearest_any_pid is None or nearest_any_dist >= self.owner_attach_distance_px:
                state["no_person_frames"] += 1
            else:
                state["no_person_frames"] = 0

            # No owner yet: attach initial owner only if someone is close enough.
            if owner_id is None:
                nearest_pid, nearest_dist = nearest_any_pid, nearest_any_dist
                nearest_distances.append(nearest_dist if nearest_pid is not None else float("inf"))
                if nearest_pid is not None and nearest_dist <= self.owner_attach_distance_px:
                    state["owner_id"] = nearest_pid
                    state["unattended_frames"] = 0
                    state["candidate_owner_id"] = None
                    state["candidate_owner_frames"] = 0
                # Fallback unattended logic for cases where owner never got a stable ID.
                if (
                    state["static_frames"] >= self.static_frames_threshold
                    and state["no_person_frames"] >= self.no_person_frames_threshold
                ):
                    suspicious_count += 1
                    unattended_bag_ids.append(bag_id)
                    if isinstance(bag_id, str) and bag_id.startswith("det_"):
                        unattended_bboxes.append(
                            {"track_id": bag_id, "class": bag_class, "bbox": bag["bbox"]}
                        )
                debug_bags.append(
                    {
                        "bag_id": bag_id,
                        "class": bag_class,
                        "owner_id": state.get("owner_id"),
                        "nearest_person_id": nearest_pid,
                        "nearest_person_dist": self._safe_float(nearest_dist),
                        "owner_dist": None,
                        "owner_recently_seen": False,
                        "static_frames": int(state.get("static_frames", 0)),
                        "no_person_frames": int(state.get("no_person_frames", 0)),
                        "unattended_frames": int(state.get("unattended_frames", 0)),
                        "owner_based_unattended": False,
                        "fallback_unattended": bool(
                            state["static_frames"] >= self.static_frames_threshold
                            and state["no_person_frames"] >= self.no_person_frames_threshold
                        ),
                    }
                )
                continue

            # Owner exists. Evaluate if bag is still attended by owner.
            owner_bbox = person_bboxes_by_id.get(owner_id)
            if owner_bbox is not None:
                owner_dist = self._point_to_bbox_distance(bcenter, owner_bbox)
                nearest_distances.append(owner_dist)
            else:
                owner_dist = float("inf")
                nearest_distances.append(owner_dist)

            owner_last_seen = self.person_last_seen.get(owner_id, -10**9)
            owner_recently_seen = (frame_id - owner_last_seen) <= self.person_missing_frames_threshold

            # Strong guard: if bag is currently close to any person, treat as attended.
            # This prevents false unattended alerts for carried/nearby bags when owner_id drifts.
            if nearest_any_pid is not None and nearest_any_dist <= self.owner_keep_distance_px:
                state["unattended_frames"] = 0
                if nearest_any_dist <= self.carried_distance_px:
                    state["owner_id"] = nearest_any_pid
                    state["candidate_owner_id"] = None
                    state["candidate_owner_frames"] = 0
                continue

            if owner_bbox is not None and owner_dist <= self.owner_keep_distance_px:
                # Owner remains with bag.
                state["unattended_frames"] = 0
                state["candidate_owner_id"] = None
                state["candidate_owner_frames"] = 0
                continue

            # Optional re-assignment if another person stays near bag for multiple frames.
            nearest_pid, nearest_dist = self._nearest_person_to_bag(
                bag_bbox=bag["bbox"],
                person_bboxes_by_id=person_bboxes_by_id,
                exclude_id=owner_id,
            )
            if nearest_pid is not None and nearest_dist <= self.owner_attach_distance_px:
                if state["candidate_owner_id"] == nearest_pid:
                    state["candidate_owner_frames"] += 1
                else:
                    state["candidate_owner_id"] = nearest_pid
                    state["candidate_owner_frames"] = 1

                if state["candidate_owner_frames"] >= self.reassign_frames_threshold:
                    state["owner_id"] = nearest_pid
                    state["unattended_frames"] = 0
                    state["candidate_owner_id"] = None
                    state["candidate_owner_frames"] = 0
                    continue
            else:
                state["candidate_owner_id"] = None
                state["candidate_owner_frames"] = 0

            # Count unattended only when owner is meaningfully away or missing.
            if owner_dist >= self.unattended_distance_px or not owner_recently_seen:
                state["unattended_frames"] += 1
            else:
                # Small separation/occlusion: avoid rapid false alerts.
                state["unattended_frames"] = max(0, state["unattended_frames"] - 1)

            owner_based_unattended = state["unattended_frames"] >= self.unattended_frames_threshold
            fallback_unattended = (
                state["static_frames"] >= self.static_frames_threshold
                and state["no_person_frames"] >= self.no_person_frames_threshold
            )
            if owner_based_unattended or fallback_unattended:
                suspicious_count += 1
                unattended_bag_ids.append(bag_id)
                if isinstance(bag_id, str) and bag_id.startswith("det_"):
                    unattended_bboxes.append(
                        {"track_id": bag_id, "class": bag_class, "bbox": bag["bbox"]}
                    )

            debug_bags.append(
                {
                    "bag_id": bag_id,
                    "class": bag_class,
                    "owner_id": state.get("owner_id"),
                    "nearest_person_id": nearest_any_pid,
                    "nearest_person_dist": self._safe_float(nearest_any_dist),
                    "owner_dist": self._safe_float(owner_dist),
                    "owner_recently_seen": bool(owner_recently_seen),
                    "static_frames": int(state.get("static_frames", 0)),
                    "no_person_frames": int(state.get("no_person_frames", 0)),
                    "unattended_frames": int(state.get("unattended_frames", 0)),
                    "owner_based_unattended": bool(owner_based_unattended),
                    "fallback_unattended": bool(fallback_unattended),
                }
            )

        score = 0.0 if not bags else suspicious_count / len(bags)
        self._cleanup_stale_bags(frame_id)

        return {
            "suspicious": suspicious_count > 0,
            "score": float(score),
            "suspicious_count": suspicious_count,
            "bag_count": len(bags),
            "unattended_bag_ids": unattended_bag_ids,
            "unattended_bboxes": unattended_bboxes,
            "nearest_distances": [self._safe_float(v) for v in nearest_distances],
            "debug": {
                "thresholds": {
                    "owner_attach_distance_px": self.owner_attach_distance_px,
                    "owner_keep_distance_px": self.owner_keep_distance_px,
                    "unattended_distance_px": self.unattended_distance_px,
                    "unattended_frames_threshold": self.unattended_frames_threshold,
                    "static_motion_threshold_px": self.static_motion_threshold_px,
                    "static_frames_threshold": self.static_frames_threshold,
                    "no_person_frames_threshold": self.no_person_frames_threshold,
                },
                "summary": {
                    "tracked_bag_candidates": len(
                        [obj for obj in scene_state.objects if obj.get("class") in self.suspicious_classes]
                    ),
                    "raw_detection_bag_candidates": len(
                        [
                            det
                            for det in getattr(scene_state, "detections", [])
                            if det.get("class") in self.suspicious_classes
                        ]
                    ),
                    "total_bag_candidates_used": len(bags),
                },
                "bag_debug": debug_bags,
            },
        }
