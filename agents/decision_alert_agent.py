class DecisionAlertAgent:
    def process(self, risk_result, abnormal_result, panic_result, suspicious_result):
        alerts = []

        # Gate abnormal-pattern alert to sudden-run behavior to reduce false positives.
        if abnormal_result.get("abnormal", False) and panic_result.get("sudden_run", False):
            alerts.append("Abnormal activity pattern detected")

        if panic_result.get("panic", False):
            alerts.append("Potential panic behavior detected")

        if suspicious_result.get("suspicious", False):
            alerts.append("Unattended or suspicious object detected")

        if risk_result.get("risk_level") in {"high", "critical"}:
            alerts.append("Escalate to security control room")

        return {
            "alert": len(alerts) > 0,
            "alerts": alerts,
            "report": {
                "risk_level": risk_result.get("risk_level"),
                "risk_score": risk_result.get("risk_score"),
                "abnormal_score": abnormal_result.get("score"),
                "panic_score": panic_result.get("score"),
                "sudden_run": panic_result.get("sudden_run", False),
                "sudden_run_count": panic_result.get("sudden_run_count", 0),
                "suspicious_score": suspicious_result.get("score"),
            },
        }
