class RiskEvaluationAgent:
    def __init__(
        self,
        abnormal_weight=0.4,
        crowd_weight=0.2,
        panic_weight=0.25,
        suspicious_weight=0.15,
    ):
        self.abnormal_weight = abnormal_weight
        self.crowd_weight = crowd_weight
        self.panic_weight = panic_weight
        self.suspicious_weight = suspicious_weight

    def process(self, abnormal_result, crowd_result, panic_result, suspicious_result):
        a = float(abnormal_result.get("score", 0.0))
        c = float(crowd_result.get("score", 0.0))
        p = float(panic_result.get("score", 0.0))
        s = float(suspicious_result.get("score", 0.0))

        score = (
            self.abnormal_weight * a
            + self.crowd_weight * c
            + self.panic_weight * p
            + self.suspicious_weight * s
        )

        if score >= 0.75:
            level = "critical"
        elif score >= 0.5:
            level = "high"
        elif score >= 0.3:
            level = "medium"
        else:
            level = "low"

        return {
            "risk_score": float(score),
            "risk_level": level,
            "components": {
                "abnormal": a,
                "crowd": c,
                "panic": p,
                "suspicious_object": s,
            },
        }
