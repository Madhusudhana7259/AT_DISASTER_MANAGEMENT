import { useEffect, useMemo, useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

function StatusChip({ label, tone = "ok" }) {
  return <span className={`chip chip-${tone}`}>{label}</span>;
}

function MetricCard({ title, value, subtitle, tone = "ok" }) {
  return (
    <article className={`metric-card metric-${tone}`}>
      <h3>{title}</h3>
      <div className="metric-value">{value}</div>
      {subtitle ? <p>{subtitle}</p> : null}
    </article>
  );
}

function formatTime(unixSeconds) {
  if (!unixSeconds) return "--";
  return new Date(unixSeconds * 1000).toLocaleTimeString();
}

export default function App() {
  const [status, setStatus] = useState(null);
  const [clock, setClock] = useState(new Date());
  const [eventLog, setEventLog] = useState([]);
  const [sendLoading, setSendLoading] = useState(false);
  const [sendError, setSendError] = useState("");
  const [sendSuccess, setSendSuccess] = useState("");
  const [lastReportAt, setLastReportAt] = useState(null);

  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/status`);
        const data = await res.json();
        setStatus(data);
      } catch (err) {
        console.error("status fetch error", err);
      }
    };

    load();
    const id = setInterval(load, 900);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const id = setInterval(() => setClock(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (!status) return;
    const alerts = status.alerts ?? [];
    const message = alerts[0] || `${status.disaster_type || "Normal"} scene update`;
    const level = alerts.length > 0 ? "danger" : "ok";
    const stamp = formatTime(status.updated_at);
    const id = `${status.frame_id}-${stamp}-${message}`;

    setEventLog((prev) => {
      if (prev[0]?.id === id) return prev;
      return [{ id, stamp, message, level }, ...prev].slice(0, 10);
    });
  }, [status]);

  useEffect(() => {
    let cancelled = false;
    let busy = false;

    const sendReport = async () => {
      if (busy) return;
      busy = true;
      setSendLoading(true);
      setSendError("");
      try {
        const res = await fetch(`${API_BASE}/api/report/send`, {
          method: "POST",
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.detail || "Could not send report.");
        }
        if (!cancelled) {
          setSendSuccess(data.message || "Automatic report sent successfully.");
          setLastReportAt(new Date());
        }
      } catch (err) {
        if (!cancelled) {
          setSendError(err.message || "Failed to send automatic report.");
        }
      } finally {
        busy = false;
        if (!cancelled) {
          setSendLoading(false);
        }
      }
    };

    const intervalId = setInterval(sendReport, 60_000);
    return () => {
      cancelled = true;
      clearInterval(intervalId);
    };
  }, []);

  const agents = status?.agents ?? {};
  const abnormal = agents.abnormal_activity ?? {};
  const crowd = agents.crowd_density ?? {};
  const panic = agents.panic ?? {};
  const suspicious = agents.suspicious_object ?? {};
  const risk = status?.risk ?? {};
  const alerts = status?.alerts ?? [];

  const riskTone = useMemo(() => {
    const level = risk.risk_level;
    if (level === "critical" || level === "high") return "danger";
    if (level === "medium") return "warn";
    return "ok";
  }, [risk.risk_level]);

  return (
    <main className="dashboard">
      <header className="panel topbar">
        <div>
          <h1>AI Crowd Safety Control Room</h1>
          <p>Live surveillance, multi-agent risk intelligence, and auto incident reporting</p>
        </div>
        <div className="top-meta">
          <div className="top-meta-item">
            <span>Local Time</span>
            <strong>{clock.toLocaleTimeString()}</strong>
          </div>
          <div className="top-meta-item">
            <span>Frame</span>
            <strong>#{status?.frame_id ?? 0}</strong>
          </div>
          <StatusChip
            label={`Risk ${String(risk.risk_level || "low").toUpperCase()}`}
            tone={riskTone}
          />
        </div>
      </header>

      <section className="workspace">
        <aside className="panel left-col">
          <div className="panel-head">
            <h2>Live Alerts</h2>
            <span className="mono">{alerts.length} active</span>
          </div>
          {alerts.length === 0 ? (
            <p className="empty-alert">No active alerts. Scene is currently stable.</p>
          ) : (
            <ul className="alert-list">
              {alerts.map((a) => (
                <li key={a}>{a}</li>
              ))}
            </ul>
          )}

          <div className="timeline">
            <h3>Event Timeline</h3>
            {eventLog.length === 0 ? (
              <p className="empty-alert">Waiting for live events...</p>
            ) : (
              <ul className="timeline-list">
                {eventLog.map((item) => (
                  <li key={item.id} className={`timeline-item ${item.level}`}>
                    <span>{item.stamp}</span>
                    <p>{item.message}</p>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </aside>

        <section className="panel center-col">
          <div className="panel-head">
            <h2>Surveillance Feed</h2>
            <span className="mono">Updated {formatTime(status?.updated_at)}</span>
          </div>

          <div className="video-stage">
            <img
              src={`${API_BASE}/api/stream`}
              alt="Live surveillance feed"
              className="video-feed"
            />
            <div className="video-overlay">
              <StatusChip
                label={status?.warmup ? "Model Warming Up" : "Model Active"}
                tone={status?.warmup ? "warn" : "ok"}
              />
              <StatusChip
                label={status?.disaster_type || "Normal"}
                tone={alerts.length > 0 ? "danger" : "ok"}
              />
              <StatusChip
                label={`Abnormal ${abnormal.detected ? "YES" : "NO"}`}
                tone={abnormal.detected ? "danger" : "ok"}
              />
              <StatusChip
                label={`Suspicious ${suspicious.unattended_count ?? 0}`}
                tone={suspicious.detected ? "danger" : "ok"}
              />
            </div>
          </div>

          <div className="legend-row">
            <span className="legend-item legend-abnormal">Abnormal activity</span>
            <span className="legend-item legend-panic">Sudden run</span>
            <span className="legend-item legend-suspicious">Suspicious object</span>
          </div>
        </section>

        <aside className="panel right-col">
          <div className="panel-head">
            <h2>Agent Telemetry</h2>
            <span className="mono">Realtime</span>
          </div>

          <div className="metrics-grid">
            <MetricCard
              title="Abnormal Activity"
              value={abnormal.detected ? "Detected" : "Normal"}
              subtitle={`Score: ${(abnormal.score ?? 0).toFixed(2)}`}
              tone={abnormal.detected ? "danger" : "ok"}
            />
            <MetricCard
              title="Crowd Density"
              value={String(crowd.density_level || "low").toUpperCase()}
              subtitle={`People: ${crowd.crowd_count ?? 0}`}
              tone={crowd.density_level === "high" ? "warn" : "ok"}
            />
            <MetricCard
              title="Panic Detection"
              value={panic.detected ? "Triggered" : "Stable"}
              subtitle={`Sudden runners: ${panic.sudden_run_count ?? 0}`}
              tone={panic.detected ? "danger" : "ok"}
            />
            <MetricCard
              title="Suspicious Object"
              value={suspicious.detected ? "Detected" : "Clear"}
              subtitle={`Unattended: ${suspicious.unattended_count ?? 0} / Bags: ${suspicious.bag_count ?? 0}`}
              tone={suspicious.detected ? "danger" : "ok"}
            />
            <MetricCard
              title="Risk Score"
              value={(risk.risk_score ?? 0).toFixed(2)}
              subtitle={`Level: ${String(risk.risk_level || "low").toUpperCase()}`}
              tone={riskTone}
            />
            <MetricCard
              title="Disaster Type"
              value={status?.disaster_type || "Normal"}
              subtitle={`Alerts: ${alerts.length}`}
              tone={alerts.length > 0 ? "danger" : "ok"}
            />
          </div>

          <div className="report-panel">
            <h3>Auto Alert Report</h3>
            <p>Report dispatch runs every 60 seconds to the fixed recipient from `.env`.</p>
            {sendLoading ? <p className="report-msg">Sending scheduled report...</p> : null}
            {sendError ? <p className="report-msg report-error">{sendError}</p> : null}
            {sendSuccess ? <p className="report-msg report-success">{sendSuccess}</p> : null}
            <p className="report-note">
              Last sent at {lastReportAt ? lastReportAt.toLocaleTimeString() : "--"}
            </p>
          </div>
        </aside>
      </section>

      <footer className="panel footer-strip">
        <span>System: {status?.warmup ? "Model warmup in progress" : "Realtime monitoring active"}</span>
        <span>Camera: Cam-01</span>
        <span>Next auto report: every 60 seconds</span>
      </footer>
    </main>
  );
}
