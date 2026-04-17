class CrowdDensityAgent:
    def __init__(self, low_threshold=8, high_threshold=16):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def process(self, scene_state):
        people = [obj for obj in scene_state.objects if obj.get("class") == "person"]
        crowd_count = len(people)

        if crowd_count >= self.high_threshold:
            level = "high"
            score = 1.0
        elif crowd_count >= self.low_threshold:
            level = "medium"
            score = 0.6
        else:
            level = "low"
            score = 0.2

        return {
            "crowd_count": crowd_count,
            "density_level": level,
            "score": score,
        }
