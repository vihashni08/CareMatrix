/**
 * Command Center View — System-level high-level multi-patient overview.
 * 
 * Provides:
 * 1. Global KPI Metrics (System Status, Active Alerts, Monitored Beds, SSE Status)
 * 2. Multi-Bed Patient Census with Live Vital Telemetry & Risk Badges
 * 3. Real-Time Active Alert Feed with Action Lifecycles
 * 4. 5-Agent Health Matrix (Monitoring, Risk, Data Analysis, Clinical Reasoning, Care Coordination)
 * 5. Multi-Agent Pipeline Architecture Diagram (Derived State)
 * 6. Adaptive Execution & Surveillance Metrics
 */

window.CommandCenterView = function() {
  const state = window.useCareMatrixState();
  const dispatch = window.useCareMatrixDispatch();

  const { system, patients, activeAlerts } = state;
  const patientList = Object.values(patients);

  const handleSelectBed = (patientId) => {
    dispatch({ type: 'SET_SELECTED_PATIENT_ID', payload: patientId });
    dispatch({ type: 'SET_ACTIVE_TAB', payload: 'patient_surveillance' });
  };

  const handleSelectAgent = (agentName) => {
    dispatch({ type: 'SET_SELECTED_AGENT', payload: agentName });
    dispatch({ type: 'SET_ACTIVE_TAB', payload: 'agent_inspection' });
  };

  const handleAcknowledgeAlert = async (alertId) => {
    try {
      await window.CareMatrixApi.acknowledgeAlert(alertId, 'Attending Physician');
    } catch (e) {
      console.error(e);
    }
  };

  const handleResolveAlert = async (alertId) => {
    try {
      await window.CareMatrixApi.resolveAlert(alertId, 'Bedside evaluation complete; patient stable under monitoring.');
    } catch (e) {
      console.error(e);
    }
  };

  // 5 Registered Agents definition for health matrix
  const agentDefs = [
    { key: 'MonitoringAgent', label: 'MONITORING AGENT', role: 'Vitals Surveillance & Anomaly Gating', type: 'Agentic Tool Use' },
    { key: 'RiskAgent', label: 'RISK AGENT', role: 'Physiological Deterioration Inference', type: 'Trained ML Model' },
    { key: 'DataAnalysisAgent', label: 'DATA ANALYSIS AGENT', role: 'Rate-of-Change, Slopes & Drift', type: 'Agentic Tool Use' },
    { key: 'ClinicalReasoningAgent', label: 'CLINICAL REASONING AGENT', role: 'RAG Grounding & Evidence Synthesis', type: 'Agentic Tool Use' },
    { key: 'CareCoordinationAgent', label: 'CARE COORDINATION AGENT', role: 'Bedside Action Plan & Pathways', type: 'Deterministic CDS' },
  ];

  const agentHealthData = system.agents || system.supervisor?.agents || {};
  const metrics = system.runtimeMetrics || {};

  return (
    <main className="flex-1 p-6 space-y-6 max-w-7xl mx-auto w-full">
      {/* ===================================================================== */}
      {/* 1. TOP SYSTEM KPI HERO STRIP                                          */}
      {/* ===================================================================== */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {/* System Health */}
        <div className="bg-medCard border border-medBorder rounded-2xl p-4 shadow-lg flex items-center justify-between">
          <div>
            <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">System Health</div>
            <div className="text-xl font-black text-white mt-1 flex items-center gap-2">
              <span className={`h-2.5 w-2.5 rounded-full ${system.allHealthy ? 'bg-emerald-400' : 'bg-red-400 animate-ping'}`}></span>
              <span>{system.allHealthy ? 'HEALTHY' : 'DEGRADED'}</span>
            </div>
            <div className="text-[10px] text-slate-500 mt-1">
              {system.totalAgents || 5} Registered Agents Online
            </div>
          </div>
          <div className="h-10 w-10 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-center text-lg">
            🛡️
          </div>
        </div>

        {/* Active Alerts */}
        <div className={`border rounded-2xl p-4 shadow-lg flex items-center justify-between ${
          activeAlerts.length > 0
            ? 'bg-red-950/40 border-red-700/80'
            : 'bg-medCard border-medBorder'
        }`}>
          <div>
            <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">Active Alerts</div>
            <div className={`text-2xl font-black mt-1 ${activeAlerts.length > 0 ? 'text-red-300' : 'text-slate-100'}`}>
              {activeAlerts.length}
            </div>
            <div className="text-[10px] text-slate-500 mt-1">
              Tier 2 Escalation Guardrails
            </div>
          </div>
          <div className="h-10 w-10 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-center text-lg">
            🚨
          </div>
        </div>

        {/* Monitored Beds */}
        <div className="bg-medCard border border-medBorder rounded-2xl p-4 shadow-lg flex items-center justify-between">
          <div>
            <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">Monitored Beds</div>
            <div className="text-2xl font-black text-white mt-1">
              {patientList.length}
            </div>
            <div className="text-[10px] text-slate-500 mt-1">
              Continuous Telemetry Streams
            </div>
          </div>
          <div className="h-10 w-10 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-center text-lg">
            🏥
          </div>
        </div>

        {/* SSE Event Transport */}
        <div className="bg-medCard border border-medBorder rounded-2xl p-4 shadow-lg flex items-center justify-between">
          <div>
            <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">SSE Event Bus</div>
            <div className={`text-xl font-black mt-1 font-mono ${
              system.sseStatus === 'CONNECTED' ? 'text-emerald-400' : 'text-amber-400'
            }`}>
              {system.sseStatus}
            </div>
            <div className="text-[10px] text-slate-500 mt-1">
              /api/events/stream
            </div>
          </div>
          <div className="h-10 w-10 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-center text-lg">
            ⚡
          </div>
        </div>
      </section>

      {/* ===================================================================== */}
      {/* 2. PATIENT CENSUS GRID (ALL BEDS)                                     */}
      {/* ===================================================================== */}
      <section className="bg-medCard border border-medBorder rounded-2xl p-5 shadow-xl space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-medBorder pb-3 gap-2">
          <div>
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              Inpatient Telemetry Census
              <span className="text-xs font-normal text-slate-400">
                (Click any bed for in-depth patient surveillance)
              </span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Multi-patient real-time hemodynamic surveillance with continuous anomaly detection.
            </p>
          </div>
          <span className="text-xs font-mono text-slate-400">
            {patientList.length} Monitored Beds Active
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {patientList.map(p => {
            const hasAlert = p.has_active_alert || p.status === 'ALERT';
            const isSurveillance = p.status === 'SURVEILLANCE';
            const safeNumeric = (value) => {
              const n = Number(value);
              return Number.isFinite(n) ? n : null;
            };
            const hrValue = safeNumeric(p.latest_vitals?.hr ?? p.latest_vitals?.HR);
            const mapValue = safeNumeric(p.latest_vitals?.map ?? p.latest_vitals?.MAP);
            const spo2Value = safeNumeric(p.latest_vitals?.spo2 ?? p.latest_vitals?.SpO2);
            const rrValue = safeNumeric(p.latest_vitals?.rr ?? p.latest_vitals?.RR);
            const hr = hrValue !== null ? Math.round(hrValue) : null;
            const map = mapValue !== null ? Math.round(mapValue) : null;
            const spo2 = spo2Value !== null ? Math.round(spo2Value) : null;
            const rr = rrValue !== null ? Math.round(rrValue) : null;
            const riskLevel = p.risk_level || 'LOW RISK';
            const isHighRisk = riskLevel.includes('HIGH');

            return (
              <div
                key={p.patient_id}
                onClick={() => handleSelectBed(p.patient_id)}
                className={`p-4 rounded-xl border text-left cursor-pointer transition-all hover:scale-[1.01] flex flex-col justify-between shadow-lg ${
                  hasAlert
                    ? 'bg-red-950/30 border-red-700/80 hover:border-red-500'
                    : isSurveillance
                    ? 'bg-amber-950/20 border-amber-700/70 hover:border-amber-500'
                    : 'bg-slate-900/90 border-slate-800 hover:border-sky-500'
                }`}
              >
                <div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <span className="text-sm font-bold text-white">{p.name || `Bed ${p.patient_id}`}</span>
                      <span className="text-[10px] font-mono text-slate-400">ID: {p.patient_id}</span>
                    </div>
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase ${
                      hasAlert ? 'bg-red-600 text-white animate-pulse' :
                      isSurveillance ? 'bg-amber-500 text-black font-extrabold' :
                      'bg-emerald-600/30 text-emerald-300 border border-emerald-500/40'
                    }`}>
                      {p.status || 'STABLE'}
                    </span>
                  </div>

                  {/* Vitals matrix */}
                  <div className="grid grid-cols-4 gap-2 mt-3 pt-3 border-t border-slate-800/80 text-center">
                    <div className="bg-medDark/80 p-2 rounded-lg border border-slate-800">
                      <span className="text-[9px] font-bold text-slate-400 block uppercase">HR</span>
                      <span className="text-base font-black text-slate-100 font-mono mt-0.5 block">{hr > 0 ? hr : '--'}</span>
                    </div>
                    <div className="bg-medDark/80 p-2 rounded-lg border border-slate-800">
                      <span className="text-[9px] font-bold text-slate-400 block uppercase">MAP</span>
                      <span className="text-base font-black text-slate-100 font-mono mt-0.5 block">{map > 0 ? map : '--'}</span>
                    </div>
                    <div className="bg-medDark/80 p-2 rounded-lg border border-slate-800">
                      <span className="text-[9px] font-bold text-slate-400 block uppercase">SpO2</span>
                      <span className="text-base font-black text-slate-100 font-mono mt-0.5 block">{spo2 > 0 ? `${spo2}%` : '--'}</span>
                    </div>
                    <div className="bg-medDark/80 p-2 rounded-lg border border-slate-800">
                      <span className="text-[9px] font-bold text-slate-400 block uppercase">RR</span>
                      <span className="text-base font-black text-slate-100 font-mono mt-0.5 block">{rr > 0 ? rr : '--'}</span>
                    </div>
                  </div>
                </div>

                <div className="mt-3 pt-2.5 border-t border-slate-800/80 flex items-center justify-between text-xs">
                  <div className="flex items-center space-x-1.5">
                    <span className={`h-2 w-2 rounded-full ${isHighRisk ? 'bg-red-400' : 'bg-emerald-400'}`}></span>
                    <span className="text-slate-400 text-[11px]">Risk:</span>
                    <span className={`font-bold text-[11px] ${isHighRisk ? 'text-red-300' : 'text-emerald-300'}`}>
                      {riskLevel}
                    </span>
                  </div>
                  <span className="text-[10px] text-sky-400 hover:underline font-semibold flex items-center gap-1">
                    Surveillance View →
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* ===================================================================== */}
      {/* 3. ACTIVE REAL-TIME ALERT FEED                                        */}
      {/* ===================================================================== */}
      <section className="bg-medCard border border-medBorder rounded-2xl p-5 shadow-xl space-y-3">
        <div className="flex items-center justify-between border-b border-medBorder pb-3">
          <div className="flex items-center space-x-2.5">
            <span className="h-2.5 w-2.5 rounded-full bg-red-400 animate-ping"></span>
            <h2 className="text-base font-bold text-white uppercase tracking-tight">
              Real-Time Clinical Alert Feed
            </h2>
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
              {activeAlerts.length} Active
            </span>
          </div>
          <span className="text-[11px] text-slate-400 font-mono">
            Directly from AlertManager Lifecycle
          </span>
        </div>

        <div className="space-y-2.5">
          {activeAlerts.length > 0 ? (
            activeAlerts.map(al => (
              <div
                key={al.alert_id}
                className="bg-slate-900/90 border border-red-700/70 p-4 rounded-xl flex flex-col md:flex-row md:items-center justify-between gap-3 shadow-md"
              >
                <div className="space-y-1">
                  <div className="flex items-center space-x-2">
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase ${
                      al.severity === 'critical' ? 'bg-red-600 text-white' : 'bg-amber-600 text-black'
                    }`}>
                      {al.severity}
                    </span>
                    <span className="text-xs font-bold text-white font-mono">Bed {al.patient_id}</span>
                    <span className="text-[11px] text-slate-400 font-mono">• Active for {al.duration_seconds}s</span>
                  </div>
                  <div className="text-sm font-semibold text-slate-100">{al.message}</div>
                  <div className="text-xs text-slate-400">
                    Recommended Action: <b className="text-slate-200">{al.recommended_action}</b>
                  </div>
                </div>

                <div className="flex items-center space-x-2 self-end md:self-center">
                  {al.state === 'NEW' && (
                    <button
                      onClick={() => handleAcknowledgeAlert(al.alert_id)}
                      className="px-3 py-1.5 bg-sky-600 hover:bg-sky-500 text-white text-xs font-bold rounded-lg shadow transition"
                    >
                      Acknowledge
                    </button>
                  )}
                  <button
                    onClick={() => handleResolveAlert(al.alert_id)}
                    className="px-3 py-1.5 bg-emerald-700 hover:bg-emerald-600 text-white text-xs font-bold rounded-lg shadow transition"
                  >
                    Resolve
                  </button>
                </div>
              </div>
            ))
          ) : (
            <div className="p-6 bg-slate-900/50 rounded-xl border border-slate-800 text-center text-xs text-slate-400">
              <span className="text-emerald-400 font-bold block mb-1 text-sm">✓ All Inpatient Beds Operating Within Normal Parameters</span>
              No active clinical escalations or unacknowledged alarm triggers across the unit.
            </div>
          )}
        </div>
      </section>

      {/* ===================================================================== */}
      {/* 4. 5-AGENT HEALTH MATRIX                                              */}
      {/* ===================================================================== */}
      <section className="bg-medCard border border-medBorder rounded-2xl p-5 shadow-xl space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-medBorder pb-3 gap-2">
          <div>
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              5-Agent Autonomous System Health Matrix
              <span className="text-xs font-normal text-slate-400">
                (Click any agent to inspect execution traces)
              </span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Live heartbeat telemetry and fault supervisor recovery states across cooperating workers.
            </p>
          </div>
          <span className="text-xs font-mono text-slate-400">
            Supervisor Interval: 0.5s | Heartbeat Timeout: 4.0s
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
          {agentDefs.map(agent => {
            const data = agentHealthData[agent.key] || {};
            const isHealthy = data.status === 'healthy';
            const hb = data.last_heartbeat_ago_seconds;

            return (
              <div
                key={agent.key}
                onClick={() => handleSelectAgent(agent.key)}
                className="bg-slate-900/90 border border-slate-800 hover:border-sky-500 rounded-xl p-3.5 flex flex-col justify-between cursor-pointer transition shadow-md"
              >
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-[10px] font-mono font-bold text-sky-400 px-1.5 py-0.5 rounded bg-sky-950 border border-sky-800">
                      {agent.type}
                    </span>
                    <span className={`h-2.5 w-2.5 rounded-full ${isHealthy ? 'bg-emerald-400' : 'bg-red-400 animate-ping'}`}></span>
                  </div>
                  <div className="text-xs font-bold text-slate-100">{agent.label}</div>
                  <p className="text-[10px] text-slate-400 mt-0.5 line-clamp-1">{agent.role}</p>

                  <div className="mt-3 pt-2.5 border-t border-slate-800 text-[10px] text-slate-400 space-y-1">
                    <div className="flex justify-between">
                      <span>Status:</span>
                      <b className={`font-mono uppercase ${isHealthy ? 'text-emerald-400' : 'text-red-400'}`}>
                        {data.status || 'HEALTHY'}
                      </b>
                    </div>
                    <div className="flex justify-between">
                      <span>Heartbeat:</span>
                      <b className="font-mono text-slate-200">
                        {hb != null ? `${hb}s ago` : 'active'}
                      </b>
                    </div>
                    <div className="flex justify-between">
                      <span>Failures:</span>
                      <b className="font-mono text-slate-200">{data.failure_count || 0}</b>
                    </div>
                    <div className="flex justify-between">
                      <span>Restarts:</span>
                      <b className="font-mono text-slate-200">{data.restart_count || 0}</b>
                    </div>
                  </div>
                </div>

                <div className="mt-3 text-[10px] text-sky-400 hover:underline font-semibold pt-1 text-right">
                  Inspect Trace →
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* ===================================================================== */}
      {/* 5. RUNTIME ADAPTIVE EXECUTION METRICS                                  */}
      {/* ===================================================================== */}
      <section className="bg-medCard border border-medBorder rounded-2xl p-5 shadow-xl space-y-4">
        <div className="flex items-center justify-between border-b border-medBorder pb-3">
          <div>
            <h2 className="text-base font-bold text-white">Adaptive Execution & Surveillance Metrics</h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Reflects runtime gating: routine observations bypass computationally intensive agent chains.
            </p>
          </div>
          <span className="text-xs font-mono text-slate-400">System Metrics Tracker</span>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
          <div className="bg-slate-900/90 p-3.5 rounded-xl border border-slate-800">
            <span className="text-[10px] font-bold text-slate-400 block uppercase">Total Evaluated Cycles</span>
            <span className="text-2xl font-black text-white font-mono mt-1 block">
              {metrics.total_observations_evaluated || 0}
            </span>
          </div>

          <div className="bg-slate-900/90 p-3.5 rounded-xl border border-slate-800">
            <span className="text-[10px] font-bold text-slate-400 block uppercase">Routine Bypassed Cycles</span>
            <span className="text-2xl font-black text-sky-400 font-mono mt-1 block">
              {metrics.routine_bypassed_cycles || 0}
            </span>
            <span className="text-[10px] text-slate-500 mt-0.5 block">Lightweight Tier 1 check</span>
          </div>

          <div className="bg-slate-900/90 p-3.5 rounded-xl border border-slate-800">
            <span className="text-[10px] font-bold text-slate-400 block uppercase">Escalated Anomaly Cycles</span>
            <span className="text-2xl font-black text-amber-400 font-mono mt-1 block">
              {metrics.escalated_cycles || 0}
            </span>
            <span className="text-[10px] text-slate-500 mt-0.5 block">Activated multi-agent chain</span>
          </div>

          <div className="bg-slate-900/90 p-3.5 rounded-xl border border-slate-800">
            <span className="text-[10px] font-bold text-slate-400 block uppercase">Stabilizations / Recoveries</span>
            <span className="text-2xl font-black text-emerald-400 font-mono mt-1 block">
              {metrics.recoveries_detected || 0}
            </span>
            <span className="text-[10px] text-slate-500 mt-0.5 block">Auto-resolved alerts</span>
          </div>
        </div>
      </section>
    </main>
  );
};

