import { useState, useEffect, useRef } from "react";

/* ═══════════════════════════════════════════════════════════════
   AGENT DEFINITIONS
   ═══════════════════════════════════════════════════════════════ */

const AGENTS = [
  {
    id: "data",
    label: "DataAgent",
    file: "data_agent.py",
    color: "#0ea5e9",
    colorDark: "#0369a1",
    gradient: ["#0ea5e9", "#0284c7"],
    tagLabel: "DETERMINISTIC",
    tagColor: "#0ea5e9",
    description: "Loads raw CSVs, applies perimeter filters (year, country, airport, zone), and engineers 54 route-level features via FeatureBuilder — the same module the classical pipeline uses, so feature parity between architectures is guaranteed by construction.",
    reads: ["perimeter"],
    writes: ["df_raw", "df_allarmi", "df_viaggiatori", "data_meta", "df_features", "feature_meta"],
    stats: { input: "2 CSV files", output: "567 routes x 54 features", rows: "~8,400 records" },
  },
  {
    id: "baseline",
    label: "BaselineAgent",
    file: "baseline_agent.py",
    color: "#f59e0b",
    colorDark: "#b45309",
    gradient: ["#f59e0b", "#d97706"],
    tagLabel: "DETERMINISTIC",
    tagColor: "#f59e0b",
    description: "Computes cross-sectional baselines using robust z-scores (median + MAD). Flags routes deviating from the population norm on 13 security-relevant features.",
    reads: ["df_features"],
    writes: ["df_baseline", "baseline_meta"],
    stats: { input: "54 features", output: "13 z-scores + flags", method: "robust MAD z-score" },
  },
  {
    id: "outlier",
    label: "OutlierAgent",
    file: "outlier_agent.py",
    color: "#ef4444",
    colorDark: "#b91c1c",
    gradient: ["#ef4444", "#dc2626"],
    tagLabel: "DETERMINISTIC",
    tagColor: "#ef4444",
    description: "Runs a 4-model weighted ensemble (IsolationForest, LOF, Z-score, Autoencoder) and assigns data-driven risk labels: ALTA, MEDIA, NORMALE.",
    reads: ["df_baseline"],
    writes: ["df_anomalies", "anomaly_meta"],
    stats: { models: "4 ensemble", thresholdAlta: "p97 = 0.358", thresholdMedia: "p90 = 0.290" },
  },
  {
    id: "risk",
    label: "RiskProfilingAgent",
    file: "risk_profiling_agent.py",
    color: "#ec4899",
    colorDark: "#be185d",
    gradient: ["#ec4899", "#db2777"],
    tagLabel: "DETERMINISTIC",
    tagColor: "#ec4899",
    description: "Applies the five business rules of the classical post-processing layer (high INTERPOL rate, high rejection, low closure, multi-source, high alarm rate), produces a confidence score (60% ML + 40% rules) and the final classification CRITICO / ALTO / MEDIO / BASSO.",
    reads: ["df_anomalies"],
    writes: ["df_risk", "risk_meta"],
    stats: { rules: "5 business rules", confidence: "0.6·ML + 0.4·rules", levels: "CRITICO/ALTO/MEDIO/BASSO" },
  },
  {
    id: "report",
    label: "ReportAgent",
    file: "report_agent.py",
    color: "#a855f7",
    colorDark: "#7e22ce",
    gradient: ["#a855f7", "#9333ea"],
    tagLabel: "LLM-POWERED",
    tagColor: "#a855f7",
    description: "Uses Claude to generate narrative explanations for each anomalous route, citing top z-score drivers and risk drivers from the RiskProfilingAgent. Supports dry-run mode for cost-free testing. Optional — only runs when explicitly enabled.",
    reads: ["df_risk", "risk_meta", "perimeter"],
    writes: ["report (JSON)", "report_path"],
    stats: { model: "Claude Sonnet", modes: "LLM / dry-run", output: "JSON report" },
  },
];

/* ═══════════════════════════════════════════════════════════════
   PER-AGENT SVG ICONS
   ═══════════════════════════════════════════════════════════════ */

function IconData({ size = 48 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 48 48" fill="none">
      <ellipse cx="24" cy="12" rx="16" ry="6" fill="#0ea5e9" opacity="0.25" />
      <ellipse cx="24" cy="12" rx="16" ry="6" stroke="#0ea5e9" strokeWidth="2" fill="none" />
      <path d="M8 12v8c0 3.3 7.2 6 16 6s16-2.7 16-6v-8" stroke="#0ea5e9" strokeWidth="2" fill="none" />
      <ellipse cx="24" cy="20" rx="16" ry="6" stroke="#0ea5e9" strokeWidth="1" fill="none" opacity="0.4" />
      <path d="M8 20v8c0 3.3 7.2 6 16 6s16-2.7 16-6v-8" stroke="#0ea5e9" strokeWidth="2" fill="none" />
      <ellipse cx="24" cy="28" rx="16" ry="6" stroke="#0ea5e9" strokeWidth="1" fill="none" opacity="0.3" />
      {/* filter funnel */}
      <path d="M30 34l6-4v-4l-4 3" stroke="#38bdf8" strokeWidth="1.5" opacity="0.7" />
      <circle cx="35" cy="26" r="2" fill="#38bdf8" opacity="0.6" />
    </svg>
  );
}

function IconBaseline({ size = 48 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 48 48" fill="none">
      {/* bell curve */}
      <path
        d="M4 38 Q10 38 14 34 Q18 28 20 18 Q22 8 24 6 Q26 8 28 18 Q30 28 34 34 Q38 38 44 38"
        stroke="#f59e0b"
        strokeWidth="2.5"
        fill="none"
      />
      {/* fill under curve */}
      <path
        d="M4 38 Q10 38 14 34 Q18 28 20 18 Q22 8 24 6 Q26 8 28 18 Q30 28 34 34 Q38 38 44 38 L44 42 L4 42 Z"
        fill="#f59e0b"
        opacity="0.1"
      />
      {/* mean line */}
      <line x1="24" y1="4" x2="24" y2="42" stroke="#f59e0b" strokeWidth="1" strokeDasharray="3 2" opacity="0.5" />
      {/* sigma markers */}
      <line x1="14" y1="36" x2="14" y2="40" stroke="#fbbf24" strokeWidth="1.5" opacity="0.7" />
      <line x1="34" y1="36" x2="34" y2="40" stroke="#fbbf24" strokeWidth="1.5" opacity="0.7" />
      <text x="13" y="45" fontSize="6" fill="#fbbf24" opacity="0.7" textAnchor="middle">-2.5σ</text>
      <text x="35" y="45" fontSize="6" fill="#fbbf24" opacity="0.7" textAnchor="middle">+2.5σ</text>
    </svg>
  );
}

function IconOutlier({ size = 48 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 48 48" fill="none">
      {/* normal cluster */}
      {[[18,26],[22,22],[20,30],[26,24],[24,28],[16,24],[22,32],[28,28],[20,20],[26,32],[14,28],[30,24]].map(([cx, cy], i) => (
        <circle key={i} cx={cx} cy={cy} r="2" fill="#64748b" opacity="0.4" />
      ))}
      {/* outliers - bright red, away from cluster */}
      <circle cx="40" cy="10" r="3" fill="#ef4444" opacity="0.9" />
      <circle cx="8" cy="40" r="3" fill="#ef4444" opacity="0.9" />
      <circle cx="42" cy="38" r="2.5" fill="#ef4444" opacity="0.7" />
      {/* detection boundary */}
      <ellipse cx="22" cy="26" rx="14" ry="10" stroke="#ef4444" strokeWidth="1.5" strokeDasharray="4 3" fill="none" opacity="0.4" />
      {/* alert icon on outlier */}
      <path d="M38 7 L40 3 L42 7" stroke="#fca5a5" strokeWidth="1.5" fill="none" />
      <circle cx="40" cy="9" r="0.5" fill="#fca5a5" />
    </svg>
  );
}

function IconReport({ size = 48 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 48 48" fill="none">
      {/* document */}
      <rect x="10" y="4" width="28" height="36" rx="3" fill="#a855f7" opacity="0.1" stroke="#a855f7" strokeWidth="1.5" />
      {/* text lines */}
      <line x1="16" y1="12" x2="32" y2="12" stroke="#a855f7" strokeWidth="2" opacity="0.6" />
      <line x1="16" y1="18" x2="30" y2="18" stroke="#a855f7" strokeWidth="1.5" opacity="0.4" />
      <line x1="16" y1="22" x2="28" y2="22" stroke="#a855f7" strokeWidth="1.5" opacity="0.4" />
      <line x1="16" y1="26" x2="32" y2="26" stroke="#a855f7" strokeWidth="1.5" opacity="0.4" />
      <line x1="16" y1="30" x2="24" y2="30" stroke="#a855f7" strokeWidth="1.5" opacity="0.4" />
      {/* AI sparkle */}
      <path d="M36 6 L38 2 L40 6 L44 8 L40 10 L38 14 L36 10 L32 8 Z" fill="#c084fc" opacity="0.7" />
      <path d="M34 32 L35 30 L36 32 L38 33 L36 34 L35 36 L34 34 L32 33 Z" fill="#c084fc" opacity="0.5" />
    </svg>
  );
}

function IconRisk({ size = 48 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 48 48" fill="none">
      {/* shield */}
      <path d="M24 4 L40 10 V24 C40 32 32 40 24 44 C16 40 8 32 8 24 V10 Z"
            stroke="#ec4899" strokeWidth="2" fill="#ec4899" fillOpacity="0.1" strokeLinejoin="round" />
      {/* checklist rules inside */}
      <line x1="14" y1="16" x2="34" y2="16" stroke="#ec4899" strokeWidth="1.5" opacity="0.7" />
      <line x1="14" y1="22" x2="32" y2="22" stroke="#ec4899" strokeWidth="1.5" opacity="0.5" />
      <line x1="14" y1="28" x2="34" y2="28" stroke="#ec4899" strokeWidth="1.5" opacity="0.5" />
      <line x1="14" y1="34" x2="30" y2="34" stroke="#ec4899" strokeWidth="1.5" opacity="0.4" />
      {/* check marks */}
      <path d="M11 16 L12.5 17.5 L15 14.5" stroke="#f9a8d4" strokeWidth="1.5" fill="none" />
      <path d="M11 22 L12.5 23.5 L15 20.5" stroke="#f9a8d4" strokeWidth="1.5" fill="none" />
      <path d="M11 28 L12.5 29.5 L15 26.5" stroke="#f9a8d4" strokeWidth="1.5" fill="none" />
    </svg>
  );
}

const ICON_MAP = {
  data: IconData,
  baseline: IconBaseline,
  outlier: IconOutlier,
  risk: IconRisk,
  report: IconReport,
};

/* ═══════════════════════════════════════════════════════════════
   ANIMATED DATA PARTICLES
   ═══════════════════════════════════════════════════════════════ */

function DataParticles({ activeEdge, agents }) {
  const [particles, setParticles] = useState([]);
  const frameRef = useRef(0);

  useEffect(() => {
    if (activeEdge === null) { setParticles([]); return; }
    const fromIdx = agents.findIndex(a => a.id === AGENTS[activeEdge].id);
    const toIdx = fromIdx + 1;
    if (toIdx >= agents.length) { setParticles([]); return; }

    const interval = setInterval(() => {
      frameRef.current += 1;
      setParticles(prev => {
        const next = prev
          .map(p => ({ ...p, progress: p.progress + p.speed }))
          .filter(p => p.progress < 1);
        if (frameRef.current % 3 === 0) {
          next.push({
            id: frameRef.current,
            progress: 0,
            speed: 0.015 + Math.random() * 0.01,
            offsetX: (Math.random() - 0.5) * 20,
          });
        }
        return next;
      });
    }, 30);
    return () => clearInterval(interval);
  }, [activeEdge]);

  if (activeEdge === null) return null;
  const fromIdx = activeEdge;
  const toIdx = fromIdx + 1;
  if (toIdx >= AGENTS.length) return null;

  const color = AGENTS[toIdx].color;

  return (
    <>
      {particles.map(p => {
        const y = 0 + p.progress * 1;
        return (
          <div
            key={p.id}
            style={{
              position: "absolute",
              left: `calc(50% + ${p.offsetX}px)`,
              top: `${(fromIdx * 160 + 140) + p.progress * 160}px`,
              width: 6,
              height: 6,
              borderRadius: "50%",
              background: color,
              boxShadow: `0 0 8px ${color}`,
              opacity: 1 - p.progress * 0.5,
              transform: "translate(-50%, -50%)",
              pointerEvents: "none",
              transition: "none",
            }}
          />
        );
      })}
    </>
  );
}

/* ═══════════════════════════════════════════════════════════════
   AGENT CARD
   ═══════════════════════════════════════════════════════════════ */

function AgentCard({ agent, index, isActive, isSelected, onClick, totalAgents }) {
  const Icon = ICON_MAP[agent.id];
  const isLast = index === totalAgents - 1;

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", position: "relative" }}>
      {/* Card */}
      <div
        onClick={() => onClick(agent.id)}
        style={{
          width: 320,
          background: isSelected
            ? `linear-gradient(135deg, ${agent.color}15, ${agent.color}08)`
            : "rgba(15, 23, 42, 0.8)",
          border: `1.5px solid ${isSelected ? agent.color : isActive ? `${agent.color}60` : "#1e293b"}`,
          borderRadius: 16,
          padding: "20px 24px",
          cursor: "pointer",
          transition: "all 0.3s ease",
          position: "relative",
          overflow: "hidden",
          backdropFilter: "blur(8px)",
          boxShadow: isSelected
            ? `0 0 30px ${agent.color}20, 0 4px 20px rgba(0,0,0,0.3)`
            : isActive ? `0 0 15px ${agent.color}10` : "0 2px 10px rgba(0,0,0,0.2)",
        }}
      >
        {/* Accent bar at top */}
        <div style={{
          position: "absolute", top: 0, left: 0, right: 0, height: 3,
          background: `linear-gradient(90deg, ${agent.gradient[0]}, ${agent.gradient[1]})`,
          opacity: isSelected || isActive ? 1 : 0.3,
          transition: "opacity 0.3s",
        }} />

        <div style={{ display: "flex", alignItems: "flex-start", gap: 16 }}>
          {/* Icon */}
          <div style={{
            flexShrink: 0,
            width: 56, height: 56,
            borderRadius: 14,
            background: `${agent.color}10`,
            border: `1px solid ${agent.color}30`,
            display: "flex", alignItems: "center", justifyContent: "center",
            transition: "all 0.3s",
            transform: isActive ? "scale(1.05)" : "scale(1)",
          }}>
            <Icon size={40} />
          </div>

          {/* Text */}
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
              <span style={{
                fontSize: 11, fontWeight: 700, color: "#475569",
                fontFamily: "monospace", letterSpacing: 1,
              }}>
                AGENT {index + 1}
              </span>
              <span style={{
                fontSize: 9, fontWeight: 700, padding: "2px 8px", borderRadius: 999,
                background: agent.id === "report" ? `${agent.tagColor}25` : `${agent.tagColor}15`,
                color: agent.tagColor, letterSpacing: 0.5,
              }}>
                {agent.tagLabel}
              </span>
            </div>
            <div style={{ fontSize: 17, fontWeight: 800, color: "#f1f5f9", letterSpacing: -0.3 }}>
              {agent.label}
            </div>
            <div style={{ fontSize: 11, color: "#64748b", fontFamily: "monospace", marginTop: 2 }}>
              {agent.file}
            </div>
          </div>
        </div>

        {/* Expanded details */}
        {isSelected && (
          <div style={{ marginTop: 16, paddingTop: 14, borderTop: `1px solid ${agent.color}20` }}>
            <p style={{ fontSize: 13, color: "#94a3b8", lineHeight: 1.6, margin: "0 0 14px" }}>
              {agent.description}
            </p>

            <div style={{ display: "flex", gap: 20 }}>
              <div>
                <div style={{ fontSize: 10, fontWeight: 700, color: "#475569", textTransform: "uppercase", letterSpacing: 1, marginBottom: 6 }}>
                  Reads
                </div>
                {agent.reads.map(r => (
                  <div key={r} style={{ fontSize: 11, color: "#38bdf8", fontFamily: "monospace", marginBottom: 3, display: "flex", alignItems: "center", gap: 4 }}>
                    <span style={{ color: "#334155" }}>&#8592;</span> {r}
                  </div>
                ))}
              </div>
              <div>
                <div style={{ fontSize: 10, fontWeight: 700, color: "#475569", textTransform: "uppercase", letterSpacing: 1, marginBottom: 6 }}>
                  Writes
                </div>
                {agent.writes.map(w => (
                  <div key={w} style={{ fontSize: 11, color: "#4ade80", fontFamily: "monospace", marginBottom: 3, display: "flex", alignItems: "center", gap: 4 }}>
                    <span style={{ color: "#334155" }}>&#8594;</span> {w}
                  </div>
                ))}
              </div>
            </div>

            {/* Stats */}
            <div style={{
              marginTop: 12, display: "flex", gap: 8, flexWrap: "wrap",
            }}>
              {Object.entries(agent.stats).map(([k, v]) => (
                <span key={k} style={{
                  fontSize: 10, padding: "3px 10px", borderRadius: 6,
                  background: "rgba(30, 41, 59, 0.8)", color: "#94a3b8",
                  border: "1px solid #1e293b",
                }}>
                  <span style={{ color: "#64748b" }}>{k}: </span>
                  <span style={{ color: "#e2e8f0" }}>{v}</span>
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Connector arrow */}
      {!isLast && (
        <div style={{
          display: "flex", flexDirection: "column", alignItems: "center",
          height: 40, justifyContent: "center", position: "relative",
        }}>
          <div style={{
            width: 2, height: 24,
            background: isActive
              ? `linear-gradient(to bottom, ${agent.color}80, ${AGENTS[index + 1]?.color || agent.color}80)`
              : "#1e293b",
            transition: "background 0.5s",
          }} />
          <svg width="12" height="8" viewBox="0 0 12 8" style={{ marginTop: -1 }}>
            <path
              d="M1 1 L6 6 L11 1"
              stroke={isActive ? AGENTS[index + 1]?.color || "#475569" : "#334155"}
              strokeWidth="2"
              fill="none"
              strokeLinecap="round"
              strokeLinejoin="round"
              style={{ transition: "stroke 0.5s" }}
            />
          </svg>
          {/* Optional dashed label for last edge */}
          {index === totalAgents - 2 && (
            <span style={{
              position: "absolute", right: -72, top: "50%", transform: "translateY(-50%)",
              fontSize: 9, color: "#475569", fontStyle: "italic", letterSpacing: 0.5,
            }}>
              optional
            </span>
          )}
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   FLOW ANIMATION CONTROLS
   ═══════════════════════════════════════════════════════════════ */

function FlowControls({ isPlaying, onToggle, activeStep, totalSteps, onReset }) {
  return (
    <div style={{
      display: "flex", alignItems: "center", justifyContent: "center",
      gap: 12, marginBottom: 20,
    }}>
      <button
        onClick={onToggle}
        style={{
          padding: "8px 20px", borderRadius: 10, border: "1px solid #334155",
          background: isPlaying ? "#312e81" : "rgba(15, 23, 42, 0.8)",
          color: isPlaying ? "#c7d2fe" : "#94a3b8",
          cursor: "pointer", fontSize: 13, fontWeight: 600,
          display: "flex", alignItems: "center", gap: 6,
          transition: "all 0.2s",
        }}
      >
        {isPlaying ? (
          <>
            <svg width="14" height="14" viewBox="0 0 14 14"><rect x="2" y="2" width="4" height="10" rx="1" fill="currentColor" /><rect x="8" y="2" width="4" height="10" rx="1" fill="currentColor" /></svg>
            Pause
          </>
        ) : (
          <>
            <svg width="14" height="14" viewBox="0 0 14 14"><path d="M3 1.5 L12 7 L3 12.5Z" fill="currentColor" /></svg>
            Simulate Flow
          </>
        )}
      </button>
      <button
        onClick={onReset}
        style={{
          padding: "8px 14px", borderRadius: 10, border: "1px solid #334155",
          background: "transparent", color: "#64748b", cursor: "pointer",
          fontSize: 13, fontWeight: 600, transition: "all 0.2s",
        }}
      >
        Reset
      </button>
      {/* Progress dots */}
      <div style={{ display: "flex", gap: 5, marginLeft: 8 }}>
        {AGENTS.map((a, i) => (
          <div
            key={a.id}
            style={{
              width: 8, height: 8, borderRadius: "50%",
              background: i <= activeStep ? a.color : "#1e293b",
              border: `1.5px solid ${i <= activeStep ? a.color : "#334155"}`,
              transition: "all 0.4s",
              boxShadow: i === activeStep ? `0 0 8px ${a.color}` : "none",
            }}
          />
        ))}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   STATE FLOW SIDEBAR
   ═══════════════════════════════════════════════════════════════ */

function StateSidebar({ activeStep }) {
  const fields = [
    { key: "perimeter", agent: -1, type: "dict", label: "User input" },
    { key: "df_raw", agent: 0, type: "DataFrame" },
    { key: "df_allarmi", agent: 0, type: "DataFrame" },
    { key: "df_viaggiatori", agent: 0, type: "DataFrame" },
    { key: "data_meta", agent: 0, type: "dict" },
    { key: "df_features", agent: 0, type: "DataFrame" },
    { key: "feature_meta", agent: 0, type: "dict" },
    { key: "df_baseline", agent: 1, type: "DataFrame" },
    { key: "baseline_meta", agent: 1, type: "dict" },
    { key: "df_anomalies", agent: 2, type: "DataFrame" },
    { key: "anomaly_meta", agent: 2, type: "dict" },
    { key: "df_risk", agent: 3, type: "DataFrame" },
    { key: "risk_meta", agent: 3, type: "dict" },
    { key: "report", agent: 4, type: "dict" },
    { key: "report_path", agent: 4, type: "str" },
  ];

  return (
    <div style={{
      background: "rgba(15, 23, 42, 0.6)", border: "1px solid #1e293b",
      borderRadius: 14, padding: "16px 18px", width: 220,
      backdropFilter: "blur(8px)",
    }}>
      <div style={{ fontSize: 12, fontWeight: 700, color: "#64748b", textTransform: "uppercase", letterSpacing: 1, marginBottom: 12 }}>
        AgentState
      </div>
      {fields.map(f => {
        const filled = f.agent <= activeStep;
        const justFilled = f.agent === activeStep;
        const agentColor = f.agent >= 0 ? AGENTS[f.agent]?.color : "#6366f1";
        return (
          <div
            key={f.key}
            style={{
              display: "flex", alignItems: "center", gap: 8,
              padding: "4px 8px", marginBottom: 2, borderRadius: 6,
              background: justFilled ? `${agentColor}10` : "transparent",
              transition: "all 0.4s",
            }}
          >
            <div style={{
              width: 6, height: 6, borderRadius: "50%",
              background: filled ? agentColor : "#1e293b",
              border: `1px solid ${filled ? agentColor : "#334155"}`,
              boxShadow: justFilled ? `0 0 6px ${agentColor}` : "none",
              transition: "all 0.4s",
              flexShrink: 0,
            }} />
            <span style={{
              fontSize: 11, fontFamily: "monospace",
              color: filled ? "#e2e8f0" : "#334155",
              transition: "color 0.4s",
            }}>
              {f.key}
            </span>
            <span style={{
              fontSize: 9, color: filled ? "#64748b" : "#1e293b",
              marginLeft: "auto", transition: "color 0.4s",
            }}>
              {f.type}
            </span>
          </div>
        );
      })}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   MAIN COMPONENT
   ═══════════════════════════════════════════════════════════════ */

export default function AgentGraph() {
  const [selected, setSelected] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [activeStep, setActiveStep] = useState(-1);
  const timerRef = useRef(null);

  const handleToggle = () => {
    if (isPlaying) {
      clearInterval(timerRef.current);
      setIsPlaying(false);
    } else {
      setIsPlaying(true);
      setActiveStep(-1);
      let step = -1;
      timerRef.current = setInterval(() => {
        step += 1;
        if (step >= AGENTS.length) {
          clearInterval(timerRef.current);
          setIsPlaying(false);
          return;
        }
        setActiveStep(step);
        setSelected(AGENTS[step].id);
      }, 1400);
    }
  };

  const handleReset = () => {
    clearInterval(timerRef.current);
    setIsPlaying(false);
    setActiveStep(-1);
    setSelected(null);
  };

  useEffect(() => () => clearInterval(timerRef.current), []);

  const handleClick = (id) => {
    if (isPlaying) return;
    setSelected(selected === id ? null : id);
    const idx = AGENTS.findIndex(a => a.id === id);
    setActiveStep(selected === id ? -1 : idx);
  };

  return (
    <div style={{
      fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
      background: "linear-gradient(180deg, #020617 0%, #0f172a 100%)",
      color: "#e2e8f0",
      padding: "32px 24px",
      borderRadius: 20,
      minHeight: 600,
      position: "relative",
      overflow: "hidden",
    }}>
      {/* Background grid pattern */}
      <div style={{
        position: "absolute", inset: 0, opacity: 0.03,
        backgroundImage: "radial-gradient(#475569 1px, transparent 1px)",
        backgroundSize: "24px 24px",
        pointerEvents: "none",
      }} />

      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: 8, position: "relative" }}>
        <div style={{
          display: "inline-flex", alignItems: "center", gap: 8,
          padding: "4px 14px", borderRadius: 999, marginBottom: 10,
          background: "rgba(99, 102, 241, 0.1)", border: "1px solid rgba(99, 102, 241, 0.2)",
        }}>
          <svg width="14" height="14" viewBox="0 0 14 14">
            <circle cx="7" cy="7" r="5.5" stroke="#818cf8" strokeWidth="1.5" fill="none" />
            <circle cx="7" cy="4" r="1.5" fill="#818cf8" />
            <circle cx="4" cy="9" r="1.5" fill="#818cf8" />
            <circle cx="10" cy="9" r="1.5" fill="#818cf8" />
            <line x1="7" y1="5.5" x2="4.5" y2="7.5" stroke="#818cf8" strokeWidth="0.8" />
            <line x1="7" y1="5.5" x2="9.5" y2="7.5" stroke="#818cf8" strokeWidth="0.8" />
          </svg>
          <span style={{ fontSize: 11, fontWeight: 600, color: "#a5b4fc", letterSpacing: 0.5 }}>
            LangGraph Supervisor Pattern
          </span>
        </div>
        <h1 style={{
          margin: 0, fontSize: 26, fontWeight: 800,
          background: "linear-gradient(135deg, #f8fafc, #94a3b8)",
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
          letterSpacing: -0.5,
        }}>
          Multi-Agent Pipeline Architecture
        </h1>
        <p style={{ margin: "6px 0 0", fontSize: 13, color: "#475569" }}>
          Airport Anomaly Detection — Reply x LUISS 2026
        </p>
      </div>

      <FlowControls
        isPlaying={isPlaying}
        onToggle={handleToggle}
        activeStep={activeStep}
        totalSteps={AGENTS.length}
        onReset={handleReset}
      />

      {/* Main layout: sidebar + graph */}
      <div style={{ display: "flex", gap: 24, justifyContent: "center", position: "relative" }}>
        {/* Graph column */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", position: "relative" }}>
          <DataParticles activeEdge={isPlaying && activeStep >= 0 && activeStep < AGENTS.length - 1 ? activeStep : null} agents={AGENTS} />
          {AGENTS.map((agent, i) => (
            <AgentCard
              key={agent.id}
              agent={agent}
              index={i}
              isActive={i <= activeStep}
              isSelected={selected === agent.id}
              onClick={handleClick}
              totalAgents={AGENTS.length}
            />
          ))}
        </div>

        {/* State sidebar */}
        <StateSidebar activeStep={activeStep} />
      </div>

      {/* Footer legend */}
      <div style={{
        display: "flex", justifyContent: "center", gap: 20, marginTop: 24,
        fontSize: 11, color: "#475569",
      }}>
        <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <span style={{ width: 8, height: 8, borderRadius: 2, background: "#0ea5e920", border: "1px solid #0ea5e9", display: "inline-block" }} />
          Deterministic (no LLM)
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <span style={{ width: 8, height: 8, borderRadius: 2, background: "#a855f720", border: "1px solid #a855f7", display: "inline-block" }} />
          LLM-powered (optional)
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <svg width="20" height="2"><line x1="0" y1="1" x2="20" y2="1" stroke="#475569" strokeWidth="1.5" strokeDasharray="3 2" /></svg>
          Optional connection
        </span>
      </div>
    </div>
  );
}
