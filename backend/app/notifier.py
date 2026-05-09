import os
import re
import smtplib
import ssl
import threading
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict


class NotificationError(RuntimeError):
    pass


class NotificationConfigError(NotificationError):
    pass


_ENV_LOADED = False
_ENV_LOCK = threading.Lock()


def _load_env_file_once():
    global _ENV_LOADED
    with _ENV_LOCK:
        if _ENV_LOADED:
            return

        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.exists():
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault(key, value)

        _ENV_LOADED = True


class AlertReportNotifier:
    def _build_report_payload(self, status: Dict[str, Any]) -> Dict[str, str]:
        updated_at = float(status.get("updated_at", 0.0) or 0.0)
        timestamp = datetime.fromtimestamp(updated_at).strftime("%Y-%m-%d %H:%M:%S")
        frame_id = int(status.get("frame_id", 0))
        disaster_type = status.get("disaster_type", "Normal")
        risk = status.get("risk", {})
        risk_level = str(risk.get("risk_level", "low")).upper()
        risk_score = float(risk.get("risk_score", 0.0))
        alerts = status.get("alerts", []) or []
        agents = status.get("agents", {}) or {}
        abnormal = agents.get("abnormal_activity", {}) or {}
        crowd = agents.get("crowd_density", {}) or {}
        panic = agents.get("panic", {}) or {}
        suspicious = agents.get("suspicious_object", {}) or {}

        alert_lines = ["- No active alerts"] if not alerts else [f"- {a}" for a in alerts]
        subject = f"[Crowd Safety] {disaster_type} | Risk {risk_level}"
        email_body = "\n".join(
            [
                "AI Crowd Safety Alert Report",
                "",
                f"Timestamp: {timestamp}",
                f"Frame ID: {frame_id}",
                f"Disaster Type: {disaster_type}",
                f"Risk: {risk_level} ({risk_score:.2f})",
                "",
                "Alert Messages:",
                *alert_lines,
                "",
                "Agent Summary:",
                f"- Abnormal Activity: detected={bool(abnormal.get('detected', False))}, score={float(abnormal.get('score', 0.0)):.2f}",
                f"- Crowd Density: level={crowd.get('density_level', 'low')}, people={int(crowd.get('crowd_count', 0))}",
                f"- Panic: detected={bool(panic.get('detected', False))}, sudden_run={bool(panic.get('sudden_run', False))}, runners={int(panic.get('sudden_run_count', 0))}",
                f"- Suspicious Object: detected={bool(suspicious.get('detected', False))}, unattended={int(suspicious.get('unattended_count', 0))}, bags={int(suspicious.get('bag_count', 0))}",
            ]
        )
        return {
            "subject": subject,
            "email_body": email_body,
        }

    def _read_email_config(self):
        _load_env_file_once()

        config = {
            "smtp_host": os.getenv("ALERT_SMTP_HOST", "smtp.gmail.com").strip(),
            "smtp_port": int(os.getenv("ALERT_SMTP_PORT", "587")),
            "smtp_username": os.getenv("ALERT_SMTP_USERNAME", "your_email@gmail.com").strip(),
            "smtp_password": os.getenv("ALERT_SMTP_PASSWORD", "app_password_here").strip(),
            "sender": os.getenv("ALERT_EMAIL_FROM", "").strip(),
            "recipient_to": os.getenv("ALERT_EMAIL_TO", "recipient@example.com").strip(),
            "use_tls": os.getenv("ALERT_SMTP_USE_TLS", "true").strip().lower() == "true",
            "use_ssl": os.getenv("ALERT_SMTP_USE_SSL", "false").strip().lower() == "true",
        }
        if not config["sender"]:
            config["sender"] = config["smtp_username"]

        if config["smtp_username"] == "your_email@gmail.com" or config["smtp_password"] == "app_password_here":
            raise NotificationConfigError(
                "Update ALERT_SMTP_USERNAME and ALERT_SMTP_PASSWORD in .env before sending email"
            )
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", config["recipient_to"]):
            raise NotificationConfigError(
                "Update ALERT_EMAIL_TO in .env with a valid recipient email"
            )
        return config

    def _send_email(self, subject: str, body: str):
        config = self._read_email_config()

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = config["sender"]
        msg["To"] = config["recipient_to"]
        msg.set_content(body)

        if config["use_ssl"]:
            with smtplib.SMTP_SSL(config["smtp_host"], config["smtp_port"], timeout=20) as server:
                server.login(config["smtp_username"], config["smtp_password"])
                server.send_message(msg)
            return

        with smtplib.SMTP(config["smtp_host"], config["smtp_port"], timeout=20) as server:
            if config["use_tls"]:
                server.starttls(context=ssl.create_default_context())
            server.login(config["smtp_username"], config["smtp_password"])
            server.send_message(msg)

    def send_email_report(self, status: Dict[str, Any]) -> Dict[str, str]:
        config = self._read_email_config()
        payload = self._build_report_payload(status=status)
        self._send_email(
            subject=payload["subject"],
            body=payload["email_body"],
        )
        return {
            "recipient": config["recipient_to"],
            "subject": payload["subject"],
        }
