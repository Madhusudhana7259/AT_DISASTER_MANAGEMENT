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

export default function App() {
  const [status, setStatus] = useState(null);

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
    <main className="app-shell">
      <header className="hero">
        <div>
          <h1>AI Crowd Safety Control Room</h1>
          <p>Live surveillance stream with multi-agent risk intelligence</p>
        </div>
        <StatusChip
          label={`Risk ${String(risk.risk_level || "low").toUpperCase()}`}
          tone={riskTone}
        />
      </header>

      <section className="content-grid">
        <section className="video-panel card">
          <div className="panel-head">
            <h2>Live Video</h2>
            <span className="mono">Frame #{status?.frame_id ?? 0}</span>
          </div>
          <div className="video-wrap">
            <img
              src={`${API_BASE}/api/stream`}
              alt="Live surveillance feed"
              className="video-feed"
            />
          </div>
          <div className="video-foot">
            <StatusChip
              label={status?.warmup ? "Model Warming Up" : "Model Active"}
              tone={status?.warmup ? "warn" : "ok"}
            />
            <StatusChip
              label={status?.disaster_type || "Normal"}
              tone={alerts.length > 0 ? "danger" : "ok"}
            />
          </div>
        </section>

        <section className="metrics-panel card">
          <div className="panel-head">
            <h2>Agent Levels</h2>
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
        </section>
      </section>

      <section className="alerts card">
        <div className="panel-head">
          <h2>Alert Messages</h2>
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
      </section>
    </main>
  );
}
