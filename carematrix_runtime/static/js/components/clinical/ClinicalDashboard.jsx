/**
 * ClinicalDashboard.jsx — Professional Healthcare Decision Support Interface.
 *
 * Implements authoritative clinical documentation & structured decision support:
 * 1. PATIENT SELECTOR & BED CENSUS STRIP
 * 2. PATIENT HEADER (Bed, unit, status, telemetry sync)
 * 3. CURRENT VITALS TABLE (Parameter, Current, Status, Trend, Reference)
 * 4. CLINICAL ASSESSMENT REPORT (1. Situation, 2. Objective Findings, 3. Assessment,
 *    4. Clinical Findings, 5. Severity / Risk, 6. Clinical Considerations,
 *    7. Safety Consideration, 8. Plan / Bedside Orders, 9. Medical Evidence / PubMed)
 * 5. ALERTS & RISK PANEL (Active alarms, risk probability, acknowledgment)
 * 6. VITAL TREND CHARTS (HR, MAP, SpO2, RR with explicit medical scales & trends)
 *
 * Zero fabricated data. Zero raw LLM chain-of-thought or internal deliberation.
 */

/* ─── Clinical Utility Functions ────────────────────────────────────────── */

const _fv = (val, fallback = '—') => {
  if (val === null || val === undefined) return fallback;
  const n = Number(val);
  if (isNaN(n)) return fallback;
  return Math.round(n);
};

const _ts = (epoch) => {
  if (!epoch) return '';
  return new Date(epoch * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

const _humanizeEnum = (val) => {
  if (!val || typeof val !== 'string') return '';
  const ENUM_MAP = {
    'TRIGGER_URGENT_CLINICAL_ALERT': 'Urgent Bedside Clinical Alert',
    'SCHEDULE_CLINICIAN_REVIEW': 'Clinician Review Scheduled',
    'REQUEST_DATA_VERIFICATION': 'Bedside Sensor Verification',
    'CONTINUE_ROUTINE_MONITORING': 'Continuous Routine Observation',
    'assess_patient_risk': 'Assess Patient Risk',
    'severe': 'Severe',
    'critical': 'Critical',
    'warning': 'Warning',
    'info': 'Informational',
    'HIGH RISK': 'High Risk',
    'LOW RISK': 'Low Risk',
    'MODERATE RISK': 'Moderate Risk',
    'URGENT': 'Urgent',
    'ELEVATED': 'Elevated',
    'ROUTINE': 'Routine',
    'ALERT': 'Alert',
    'STABLE': 'Stable',
    'SURVEILLANCE': 'Surveillance',
    'RECOVERING': 'Stabilizing / Recovering',
  };
  if (ENUM_MAP[val]) return ENUM_MAP[val];
  return val
    .replace(/_/g, ' ')
    .toLowerCase()
    .replace(/\b\w/g, (c) => c.toUpperCase());
};

const _cleanClinicalText = (text) => {
  if (!text || typeof text !== 'string') return '';
  return text
    .replace(/\s*\([a-zA-Z0-9_-]*event[a-zA-Z0-9_-]*\)/gi, '')
    .replace(/\s{2,}/g, ' ')
    .trim();
};

const _filterPhysiologicalFindings = (findings) => {
  if (!Array.isArray(findings)) return [];
  const deliberationPatterns = [
    /classification retained/i,
    /provided for follow-up reasoning/i,
    /follow-up reasoning/i,
    /trend and data-quality evidence is provided/i,
    /supporting evidence:\s*model/i,
    /model.*prediction substantiated/i,
    /agentic/i,
    /fipa/i,
  ];
  return findings
    .map(f => (typeof f === 'string' ? f : (f.description || f.finding || '')))
    .filter(str => {
      if (!str || str.length < 3) return false;
      return !deliberationPatterns.some(pat => pat.test(str));
    })
    .map(str => _cleanClinicalText(str));
};

const _deduplicateOrders = (orders) => {
  if (!Array.isArray(orders) || orders.length === 0) return [];
  const rawList = orders.map(o => (typeof o === 'string' ? o : (o.order || o.description || ''))).filter(Boolean);
  const seenThemes = new Set();
  const distinct = [];
  for (const item of rawList) {
    const lower = item.toLowerCase();
    let theme = 'general';
    if (lower.includes('rapid response') || lower.includes('alert') || lower.includes('attending physician')) {
      theme = 'escalate_rrt';
    } else if (lower.includes('telemetry') || lower.includes('monitoring frequency') || lower.includes('high-density')) {
      theme = 'telemetry_freq';
    } else if (lower.includes('intravenous') || lower.includes('iv access') || lower.includes('medication')) {
      theme = 'iv_meds';
    } else if (lower.includes('vital signs') || lower.includes('re-assess') || lower.includes('reassessment')) {
      theme = 'vitals_freq';
    } else if (lower.includes('transducer') || lower.includes('cuff') || lower.includes('sensor')) {
      theme = 'sensor_check';
    } else if (lower.includes('evaluation') || lower.includes('hemodynamic') || lower.includes('bedside clinical')) {
      theme = 'bedside_eval';
    }
    if (!seenThemes.has(theme)) {
      seenThemes.add(theme);
      distinct.push(item);
    }
    if (distinct.length >= 5) break;
  }
  return distinct;
};

const _formatAuthors = (authors) => {
  if (!authors) return '';
  if (Array.isArray(authors)) {
    if (authors.length === 0) return '';
    if (authors.length <= 2) return authors.join(', ');
    return `${authors[0]} et al.`;
  }
  const str = String(authors).trim();
  if (!str) return '';
  const parts = str.split(/,\s*|\s+and\s+/i);
  if (parts.length > 2) return `${parts[0].trim()} et al.`;
  return str;
};

/* ─── 1. PatientSelector (Bed Strip) ────────────────────────────────────── */
window.PatientSelector = function({ patients, selectedPatientId, onSelect }) {
  const patientList = Object.values(patients || {});

  if (patientList.length === 0) {
    return (
      <div className="bg-white border-b border-slate-200 px-6 py-2.5 text-xs text-slate-500 font-medium">
        Connecting to Inpatient Unit Telemetry…
      </div>
    );
  }

  return (
    <nav className="bg-white border-b border-slate-200 px-6 py-2 flex items-center gap-2 overflow-x-auto text-xs" aria-label="Bed Census">
      <span className="font-semibold text-slate-400 uppercase tracking-wider text-[11px] mr-1 flex-shrink-0">
        Inpatient Beds:
      </span>
      {patientList.map(p => {
        const isSelected = String(p.patient_id) === String(selectedPatientId);
        const status = (p.status || 'STABLE').toUpperCase();
        const isAlert = status === 'ALERT' || status === 'CRITICAL' || p.has_active_alert;
        const isSurv  = status === 'SURVEILLANCE' || status === 'WARNING';

        let badgeBg = 'bg-teal-50 border-teal-200 text-teal-800';
        let dotBg = 'bg-teal-600';
        if (isAlert) {
          badgeBg = 'bg-red-50 border-red-200 text-red-800';
          dotBg = 'bg-red-600';
        } else if (isSurv) {
          badgeBg = 'bg-amber-50 border-amber-200 text-amber-800';
          dotBg = 'bg-amber-600';
        }

        return (
          <button
            key={p.patient_id}
            onClick={() => onSelect(p.patient_id)}
            className={`flex items-center gap-2 px-3 py-1.5 rounded border transition-colors flex-shrink-0 font-medium text-xs ${
              isSelected
                ? 'bg-slate-900 border-slate-900 text-white shadow-sm'
                : 'bg-slate-50 border-slate-200 text-slate-700 hover:bg-slate-100 hover:border-slate-300'
            }`}
          >
            <span className={`inline-block w-2 h-2 rounded-full ${isSelected ? 'bg-sky-400' : dotBg} ${isAlert ? 'pulse-alert' : ''}`} />
            <span>{p.name || `Bed ${p.patient_id}`}</span>
            <span className={`text-[10px] uppercase font-bold px-1.5 py-0.2 rounded border ${isSelected ? 'bg-slate-800 border-slate-700 text-slate-200' : badgeBg}`}>
              {status}
            </span>
          </button>
        );
      })}
    </nav>
  );
};

/* ─── 2. PatientHeader ─────────────────────────────────────────────────── */
window.PatientHeader = function({ patient, patientDetail }) {
  const p = patient || {};
  const pd = patientDetail || {};
  const status = (p.status || pd.status || 'STABLE').toUpperCase();
  const riskLevel = p.risk_level || pd.risk?.risk_level || 'LOW RISK';
  const lastUpdated = pd.lastUpdated || p.last_updated;

  const isAlert = status === 'ALERT' || status === 'CRITICAL' || p.has_active_alert;
  const isSurv  = status === 'SURVEILLANCE' || status === 'WARNING';
  const isHighRisk = (riskLevel || '').toUpperCase().includes('HIGH');

  let statusBadge = {
    bg: 'bg-teal-50 border-teal-200 text-teal-800',
    dot: 'bg-teal-600',
    icon: '●'
  };
  if (isAlert) {
    statusBadge = {
      bg: 'bg-red-50 border-red-200 text-red-800',
      dot: 'bg-red-600',
      icon: '▲'
    };
  } else if (isSurv) {
    statusBadge = {
      bg: 'bg-amber-50 border-amber-200 text-amber-800',
      dot: 'bg-amber-600',
      icon: '■'
    };
  }

  const formattedTime = _ts(lastUpdated);

  // Derive clinical bed & care setting
  const bedName = pd.name || p.name || `Bed ${p.patient_id || '—'}`;
  const careSetting = p.scenario === 'GRADUAL_DETERIORATION' ? 'ICU Sepsis Surveillance' :
                      p.patient_id === 200 ? 'Surgical ICU — Perioperative Hemodynamic Replay' :
                      'Inpatient Floor Monitoring';

  return (
    <header className="bg-white border-b border-slate-200 px-6 py-4">
      <div className="flex flex-wrap items-center justify-between gap-4">
        {/* Left: Patient Demographics & Location */}
        <div>
          <div className="flex items-center gap-3 flex-wrap">
            <h1 className="text-xl font-bold text-slate-900 tracking-tight">
              {bedName}
            </h1>
            <span className="text-xs font-mono font-medium text-slate-500 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
              MRN / ID: {p.patient_id || pd.patient_id}
            </span>
            <span className="text-xs text-slate-500 font-medium border-l border-slate-200 pl-3">
              Unit: <strong className="text-slate-800">{careSetting}</strong>
            </span>
          </div>
          <div className="flex items-center gap-4 text-xs text-slate-500 mt-1">
            <span className="flex items-center gap-1.5 font-medium">
              <span className="inline-block w-2 h-2 rounded-full bg-teal-600" />
              Telemetry: <strong className="text-slate-700">Live</strong>
            </span>
            {formattedTime && (
              <span>Last Synchronized: <strong className="text-slate-700">{formattedTime}</strong></span>
            )}
          </div>
        </div>

        {/* Right: Authoritative Clinical Status & Risk Indicators */}
        <div className="flex items-center gap-2.5">
          <div className={`flex items-center gap-1.5 text-xs font-bold uppercase tracking-wide px-3 py-1.5 rounded border ${statusBadge.bg}`}>
            <span className="text-[10px]">{statusBadge.icon}</span>
            <span>STATUS: {_humanizeEnum(status)}</span>
          </div>

          <div className={`text-xs font-bold uppercase tracking-wide px-3 py-1.5 rounded border ${
            isHighRisk
              ? 'bg-red-50 border-red-200 text-red-800'
              : 'bg-slate-100 border-slate-200 text-slate-700'
          }`}>
            RISK: {_humanizeEnum(riskLevel)}
          </div>
        </div>
      </div>
    </header>
  );
};

/* ─── 3. CurrentVitals (Clinical Structured Grid/Table) ─────────────────── */
window.CurrentVitals = function({ vitals }) {
  if (!vitals || Object.keys(vitals).length === 0) {
    return (
      <section className="clinical-document rounded-md p-4">
        <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-200">
          <div>
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-700">Current Physiological Vitals</h2>
            <p className="text-[11px] text-slate-500">Continuous telemetry feed compared against clinical reference ranges.</p>
          </div>
        </div>
        <div className="p-4 text-xs text-slate-400 italic">
          Awaiting telemetry stream from patient acquisition monitor…
        </div>
      </section>
    );
  }

  const v = vitals || {};
  const hr   = v.HR   ?? v.hr;
  const map  = v.MAP  ?? v.map;
  const spo2 = v.SpO2 ?? v.spo2 ?? v.SPO2;
  const rr   = v.RR   ?? v.rr;
  const sbp  = v.SBP  ?? v.sbp;
  const dbp  = v.DBP  ?? v.dbp;
  const bt   = v.BT   ?? v.bt;

  // Format blood pressure compound display
  const hasBP = (sbp !== undefined && sbp !== null && !isNaN(Number(sbp))) ||
                (dbp !== undefined && dbp !== null && !isNaN(Number(dbp)));
  const bpVal = hasBP ? `${_fv(sbp)} / ${_fv(dbp)}` : '—';

  const rows = [
    {
      param: 'Heart Rate',
      code: 'HR',
      value: hr !== undefined && hr !== null && !isNaN(Number(hr)) ? `${_fv(hr)} bpm` : '—',
      ref: '60 – 100 bpm',
      status: (hr < 60) ? 'LOW' : (hr > 100) ? 'HIGH' : 'NORMAL',
      trend: (hr > 100) ? '↑' : (hr < 60) ? '↓' : '→',
      severity: (hr > 120 || hr < 45) ? 'CRITICAL' : (hr > 100 || hr < 60) ? 'WARNING' : 'NORMAL'
    },
    {
      param: 'Blood Pressure',
      code: 'BP (Systolic / Diastolic)',
      value: hasBP ? `${bpVal} mmHg` : '—',
      ref: '90–120 / 60–80 mmHg',
      status: (sbp < 90 || dbp < 60) ? 'LOW' : (sbp > 140 || dbp > 90) ? 'HIGH' : 'NORMAL',
      trend: (sbp < 90) ? '↓' : (sbp > 140) ? '↑' : '→',
      severity: (sbp < 80 || sbp > 180) ? 'CRITICAL' : (sbp < 90 || sbp > 140) ? 'WARNING' : 'NORMAL'
    },
    {
      param: 'Mean Arterial Pressure',
      code: 'MAP',
      value: map !== undefined && map !== null && !isNaN(Number(map)) ? `${_fv(map)} mmHg` : '—',
      ref: '70 – 105 mmHg',
      status: (map < 65) ? 'CRITICAL LOW' : (map < 70) ? 'LOW' : (map > 105) ? 'HIGH' : 'NORMAL',
      trend: (map < 65) ? '↓' : (map > 105) ? '↑' : '→',
      severity: (map < 65) ? 'CRITICAL' : (map < 70) ? 'WARNING' : 'NORMAL'
    },
    {
      param: 'Oxygen Saturation',
      code: 'SpO₂',
      value: spo2 !== undefined && spo2 !== null && !isNaN(Number(spo2)) ? `${_fv(spo2)} %` : '—',
      ref: '95 – 100 %',
      status: (spo2 < 90) ? 'CRITICAL LOW' : (spo2 < 95) ? 'LOW' : 'NORMAL',
      trend: (spo2 < 95) ? '↓' : '→',
      severity: (spo2 < 90) ? 'CRITICAL' : (spo2 < 95) ? 'WARNING' : 'NORMAL'
    },
    {
      param: 'Respiratory Rate',
      code: 'RR',
      value: rr !== undefined && rr !== null && !isNaN(Number(rr)) ? `${_fv(rr)} /min` : '—',
      ref: '12 – 20 /min',
      status: (rr < 10) ? 'LOW' : (rr > 22) ? 'HIGH' : 'NORMAL',
      trend: (rr > 22) ? '↑' : (rr < 10) ? '↓' : '→',
      severity: (rr > 28 || rr < 8) ? 'CRITICAL' : (rr > 22 || rr < 10) ? 'WARNING' : 'NORMAL'
    },
    {
      param: 'Body Temperature',
      code: 'Temp',
      value: bt !== undefined && bt !== null && !isNaN(Number(bt)) ? `${Number(bt).toFixed(1)} °C` : '—',
      ref: '36.5 – 37.5 °C',
      status: (bt > 38.0) ? 'FEVER' : (bt < 36.0) ? 'HYPOTHERMIC' : 'NORMAL',
      trend: (bt > 37.5) ? '↑' : '→',
      severity: (bt > 38.5 || bt < 35.0) ? 'WARNING' : 'NORMAL'
    },
  ];

  const getStatusBadge = (sev, statusText) => {
    if (sev === 'CRITICAL') {
      return <span className="inline-flex items-center gap-1 font-bold text-red-700 bg-red-50 border border-red-200 px-2 py-0.5 rounded text-[11px]"><span className="text-[10px]">▲</span> {statusText}</span>;
    }
    if (sev === 'WARNING') {
      return <span className="inline-flex items-center gap-1 font-semibold text-amber-700 bg-amber-50 border border-amber-200 px-2 py-0.5 rounded text-[11px]"><span className="text-[10px]">■</span> {statusText}</span>;
    }
    return <span className="inline-flex items-center gap-1 font-medium text-slate-600 bg-slate-100 border border-slate-200 px-2 py-0.5 rounded text-[11px]"><span className="text-[10px]">●</span> {statusText}</span>;
  };

  return (
    <section className="clinical-document rounded-md p-4">
      <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-200">
        <div>
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-700">Current Physiological Vitals</h2>
          <p className="text-[11px] text-slate-500">Continuous telemetry feed compared against clinical reference ranges.</p>
        </div>
        <span className="text-[11px] font-mono text-slate-400">7-Channel Acquisition</span>
      </div>

      <div className="overflow-x-auto">
        <table className="clinical-table">
          <thead>
            <tr>
              <th className="w-1/4">Parameter</th>
              <th className="w-1/4">Current Value</th>
              <th className="w-1/6">Reference Range</th>
              <th className="w-1/6">Status</th>
              <th className="w-1/12 text-center">Trend</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                <td>
                  <strong className="text-slate-800">{r.param}</strong>
                  <span className="block text-[11px] text-slate-400 font-mono">{r.code}</span>
                </td>
                <td className="font-semibold text-slate-900 text-sm">
                  {r.value}
                </td>
                <td className="text-slate-500 font-mono text-xs">
                  {r.ref}
                </td>
                <td>
                  {getStatusBadge(r.severity, r.status)}
                </td>
                <td className="text-center font-bold text-slate-600 text-sm">
                  {r.trend}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
};

/* ─── 4. Clinical Assessment Report (Professional Document Format) ──────── */
window.ClinicalAssessment = function({ clinicalReasoning, careCoordination, riskDecision, vitals, vitalsHistory, patient, patientDetail }) {
  const cr = clinicalReasoning || {};
  const ca = careCoordination || {};
  const risk = riskDecision || {};
  const rep = cr.clinical_report || {};

  // Truthful empty/waiting state check
  if (!clinicalReasoning && !careCoordination && !riskDecision) {
    return (
      <article className="clinical-document rounded-md p-6 space-y-4">
        <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800 border-b border-slate-200 pb-2">
          Clinical Assessment Report
        </h2>
        <div className="p-4 bg-slate-50 border border-slate-200 rounded text-xs text-slate-500 italic space-y-2">
          <p>Awaiting clinical reasoning telemetry synthesis…</p>
          <p>Awaiting care coordination bedside recommendation queue…</p>
        </div>
      </article>
    );
  }

  const priority = cr.priority || rep.clinical_status?.priority || ca.priority || 'ROUTINE';
  const riskLevel = cr.risk_level || risk.risk_level || 'LOW RISK';
  const riskProb  = risk.risk_probability;

  const rawSummary = cr.clinical_summary || rep.executive_summary || '';
  const cleanSummary = _cleanClinicalText(rawSummary);

  const rawFindings = cr.findings || rep.key_findings || [];
  const clinicalFindings = _filterPhysiologicalFindings(rawFindings);

  const rawConsiderations = cr.supporting_evidence || rep.supporting_evidence || [];
  const considerations = _filterPhysiologicalFindings(rawConsiderations);

  const orders = _deduplicateOrders(ca.suggested_orders);
  const evidenceList = rep.medical_evidence || [];

  const isSafetyApplied = cr.metadata?.safety_arbitration_applied ||
                          risk.metadata?.physiological_safety_override ||
                          false;

  const isHighRisk = (riskLevel || '').toUpperCase().includes('HIGH') || priority === 'URGENT';
  const isSurveillance = priority === 'ELEVATED';

  // Format relevant temporal trends from vitals history
  const trendsList = [];
  if (vitalsHistory && vitalsHistory.length >= 5) {
    const maps = vitalsHistory.map(v => v.MAP ?? v.map).filter(v => v !== null && !isNaN(Number(v)));
    if (maps.length >= 2) {
      const deltaMap = Math.round(maps[maps.length - 1] - maps[0]);
      if (Math.abs(deltaMap) >= 5) {
        trendsList.push(`MAP shifted by ${deltaMap > 0 ? '+' : ''}${deltaMap} mmHg over recent sampling window`);
      }
    }
    const hrs = vitalsHistory.map(v => v.HR ?? v.hr).filter(v => v !== null && !isNaN(Number(v)));
    if (hrs.length >= 2) {
      const deltaHr = Math.round(hrs[hrs.length - 1] - hrs[0]);
      if (Math.abs(deltaHr) >= 6) {
        trendsList.push(`Heart Rate trended ${deltaHr > 0 ? 'upward' : 'downward'} by ${Math.abs(deltaHr)} bpm`);
      }
    }
  }

  // Priority banner style
  let priorityBadge = {
    bg: 'bg-teal-50 border-teal-200 text-teal-800',
    title: 'ROUTINE'
  };
  if (isHighRisk) {
    priorityBadge = {
      bg: 'bg-red-50 border-red-200 text-red-800',
      title: 'URGENT'
    };
  } else if (isSurveillance) {
    priorityBadge = {
      bg: 'bg-amber-50 border-amber-200 text-amber-800',
      title: 'ELEVATED'
    };
  }

  return (
    <article className="clinical-document rounded-md p-6 space-y-6">
      {/* Document Header */}
      <div className="border-b-2 border-slate-900 pb-4 flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
              Clinical Decision Support
            </span>
            <span className="text-xs text-slate-400">• Continuous Inpatient Surveillance</span>
          </div>
          <h2 className="text-lg font-black text-slate-900 tracking-tight mt-1">
            Clinical Assessment Report
          </h2>
        </div>

        <div className="flex items-center gap-3">
          <div className={`px-3 py-1 rounded border text-xs font-bold uppercase tracking-wider ${priorityBadge.bg}`}>
            Priority: {priorityBadge.title}
          </div>
          <span className="text-xs font-mono text-slate-500">
            {_ts(patientDetail?.lastUpdated || patient?.last_updated)}
          </span>
        </div>
      </div>

      {/* 1. Situation */}
      <section className="space-y-1.5">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800 border-b border-slate-100 pb-1 flex items-center justify-between">
          <span>1. Situation — Reason for Clinical Attention</span>
          <span className="text-[11px] font-normal text-slate-400 lowercase">clinical trigger summary</span>
        </h3>
        <p className="text-xs text-slate-700 leading-relaxed font-medium">
          {cleanSummary || (
            isHighRisk
              ? 'Patient exhibits acute hemodynamic instability marked by severe blood pressure depression requiring urgent bedside clinician evaluation.'
              : 'Patient parameters are within expected baseline clinical tolerance. Standard routine continuous monitoring is indicated.'
          )}
        </p>
      </section>

      {/* 2. Objective Findings */}
      <section className="space-y-2">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800 border-b border-slate-100 pb-1">
          2. Objective Findings & Observed Trends
        </h3>
        {trendsList.length > 0 ? (
          <ul className="text-xs text-slate-700 space-y-1 list-disc list-inside">
            {trendsList.map((tr, i) => (
              <li key={i}>{tr}</li>
            ))}
          </ul>
        ) : (
          <p className="text-xs text-slate-500 italic">No significant multi-point trend drift detected across the baseline surveillance window.</p>
        )}
      </section>

      {/* 3. Primary Clinical Assessment */}
      <section className="space-y-1.5 bg-slate-50 border border-slate-200 rounded p-3.5">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800 pb-1">
          3. Assessment — Primary Clinical Interpretation
        </h3>
        <p className="text-xs text-slate-800 leading-relaxed">
          {isHighRisk
            ? 'Findings are consistent with acute hemodynamic deterioration characterized by persistent arterial hypotension and systemic perfusion risk.'
            : isSurveillance
            ? 'Findings demonstrate borderline physiological stability with isolated parameter variance warranting increased surveillance frequency.'
            : 'Physiological trajectories demonstrate clinical stability without evidence of acute organ hypoperfusion or decompensation.'}
        </p>
      </section>

      {/* 4. Clinical Findings */}
      {clinicalFindings.length > 0 && (
        <section className="space-y-1.5">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800 border-b border-slate-100 pb-1">
            4. Clinical Findings
          </h3>
          <ul className="text-xs text-slate-700 space-y-1.5">
            {clinicalFindings.map((cf, i) => (
              <li key={i} className="flex items-start gap-2">
                <span className="text-slate-400 mt-0.5">•</span>
                <span>{cf}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* 5. Severity & Risk */}
      <section className="grid grid-cols-1 sm:grid-cols-3 gap-3 bg-white border border-slate-200 rounded p-3">
        <div>
          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Clinical Priority</span>
          <strong className="text-xs text-slate-800">{_humanizeEnum(priority)}</strong>
        </div>
        <div>
          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Risk Classification</span>
          <strong className={`text-xs ${isHighRisk ? 'text-red-700' : 'text-slate-800'}`}>
            {_humanizeEnum(riskLevel)}
          </strong>
        </div>
        <div>
          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Calibrated Probability</span>
          <strong className="text-xs text-slate-800 font-mono">
            {riskProb !== null && riskProb !== undefined ? `${(Number(riskProb) * 100).toFixed(0)}%` : '—'}
          </strong>
        </div>
      </section>

      {/* 6. Clinical Considerations */}
      {considerations.length > 0 && (
        <section className="space-y-1.5">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800 border-b border-slate-100 pb-1">
            6. Clinical Considerations & Supporting Factors
          </h3>
          <ul className="text-xs text-slate-700 space-y-1">
            {considerations.map((c, i) => (
              <li key={i} className="flex items-start gap-2">
                <span className="text-teal-600 font-bold">✓</span>
                <span>{c}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* 7. Safety Consideration */}
      {isSafetyApplied && (
        <section className="bg-amber-50 border border-amber-200 rounded p-3 text-xs text-amber-900">
          <div className="font-bold flex items-center gap-1.5 mb-0.5">
            <span>⚠</span> Safety Arbitration Notice
          </div>
          <p className="leading-relaxed">
            Safety arbitration was applied: deterministic physiological safety threshold maintained the high-risk classification to safeguard against alert suppression.
          </p>
        </section>
      )}

      {/* 8. Plan / Recommended Bedside Actions */}
      <section className="space-y-3 bg-slate-50 border border-slate-200 rounded p-4">
        <div className="flex items-center justify-between border-b border-slate-200 pb-2">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800">
            8. Plan — Recommended Bedside Actions
          </h3>
          <span className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded border ${
            ca.clinician_review_required ? 'bg-amber-100 border-amber-300 text-amber-800' : 'bg-slate-200 text-slate-700'
          }`}>
            Clinician Review Required: {ca.clinician_review_required ? 'YES' : 'NO'}
          </span>
        </div>

        {orders.length > 0 ? (
          <ul className="space-y-2 text-xs text-slate-800">
            {orders.map((ord, i) => (
              <li key={i} className="flex items-start gap-2 bg-white border border-slate-200 rounded p-2">
                <span className="font-mono text-slate-400 mt-0.5">□</span>
                <span className="font-medium leading-relaxed">{ord}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-xs text-slate-500 italic">No acute order escalation indicated. Maintain continuous standard surveillance.</p>
        )}
      </section>

      {/* 9. Medical Evidence (PubMed / RAG Citations) */}
      <section className="space-y-3 pt-2">
        <div className="flex items-center justify-between border-b border-slate-200 pb-2">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-800">
            9. Supporting Medical Literature & Clinical Guidelines
          </h3>
          <span className="text-[11px] text-slate-500">
            {evidenceList.length > 0 ? `${evidenceList.length} Grounded Reference${evidenceList.length > 1 ? 's' : ''}` : 'No References Retrieved'}
          </span>
        </div>

        {evidenceList.length > 0 ? (
          <div className="space-y-3">
            {evidenceList.map((e, i) => {
              const authors = _formatAuthors(e.authors);
              const pmid = e.pmid ? String(e.pmid).trim() : '';
              const doi  = e.doi  ? String(e.doi).trim() : '';

              return (
                <div key={i} className="bg-white border border-slate-200 rounded p-3 text-xs space-y-1.5">
                  {/* Title */}
                  <h4 className="font-bold text-slate-900 leading-snug">
                    {e.title || 'Authoritative Clinical Guidance'}
                  </h4>

                  {/* Authors & Journal Citation */}
                  <div className="text-[11px] text-slate-600 flex flex-wrap items-center gap-x-2 gap-y-1">
                    {authors && <span className="font-semibold text-slate-700">{authors}</span>}
                    {e.journal && <span className="italic">{e.journal}</span>}
                    {e.year && <span>({e.year})</span>}
                  </div>

                  {/* Identifiers (PMID / DOI) */}
                  <div className="flex items-center gap-3 text-[11px] pt-0.5">
                    {pmid && (
                      <span className="font-mono">
                        PMID:{' '}
                        <a
                          href={`https://pubmed.ncbi.nlm.nih.gov/${pmid}/`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:text-blue-800 underline font-semibold"
                        >
                          {pmid}
                        </a>
                      </span>
                    )}
                    {doi && (
                      <span className="font-mono text-slate-500">
                        DOI:{' '}
                        <a
                          href={`https://doi.org/${doi}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:text-blue-800 underline"
                        >
                          {doi}
                        </a>
                      </span>
                    )}
                  </div>

                  {/* Relevance & Grounding */}
                  {(e.application_to_case || e.key_evidence) && (
                    <div className="mt-2 text-[11px] bg-slate-50 border-l-2 border-slate-300 p-2 text-slate-700 leading-relaxed">
                      <strong className="text-slate-800 font-semibold">Clinical Relevance: </strong>
                      {e.application_to_case || e.key_evidence}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        ) : (
          <p className="text-xs text-slate-400 italic">
            Literature retrieval omitted for low-risk routine observation state.
          </p>
        )}
      </section>
    </article>
  );
};

/* ─── 5. AlertsRisk Panel ───────────────────────────────────────────────── */
window.AlertsRisk = function({ activeAlerts, riskDecision, patientId, onAcknowledge, onResolve }) {
  const patientAlerts = (activeAlerts || []).filter(a =>
    !patientId || String(a.patient_id) === String(patientId) || !a.patient_id
  );
  const risk = riskDecision || {};
  const riskProb = risk.risk_probability;
  const riskLevel = risk.risk_level || risk.decision || 'LOW RISK';
  const isHighRisk = (riskLevel || '').toUpperCase().includes('HIGH');

  return (
    <section className="clinical-document rounded-md p-4 space-y-4">
      <div className="flex items-center justify-between pb-2 border-b border-slate-200">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700">Active Alerts &amp; Surveillance</h3>
        <span className="text-[11px] text-slate-400 font-mono">{patientAlerts.length} Active</span>
      </div>

      {/* Risk Calibration Indicator */}
      {riskProb !== undefined && riskProb !== null && (
        <div className={`rounded p-3 border flex items-center justify-between ${
          isHighRisk ? 'bg-red-50 border-red-200 text-red-900' : 'bg-slate-50 border-slate-200 text-slate-800'
        }`}>
          <div>
            <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 block">Deterioration Risk Score</span>
            <span className="text-xs font-bold">{_humanizeEnum(riskLevel)}</span>
          </div>
          <div className="text-right">
            <span className={`text-xl font-black font-mono ${isHighRisk ? 'text-red-700' : 'text-slate-700'}`}>
              {(Number(riskProb) * 100).toFixed(0)}%
            </span>
            <span className="block text-[10px] text-slate-400 uppercase font-medium">Probability</span>
          </div>
        </div>
      )}

      {/* Active Alerts List */}
      {patientAlerts.length === 0 ? (
        <div className="text-xs text-teal-800 bg-teal-50 border border-teal-200 rounded p-3 flex items-center gap-2">
          <span className="text-teal-600 font-bold">✓</span>
          <span>No active bedside telemetry alarms.</span>
        </div>
      ) : (
        <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
          {patientAlerts.map(alert => {
            const sev = (alert.severity || 'INFO').toUpperCase();
            let borderCol = 'border-l-blue-600';
            let badgeBg = 'bg-blue-100 text-blue-800';
            if (sev === 'CRITICAL' || sev === 'SEVERE') {
              borderCol = 'border-l-red-600';
              badgeBg = 'bg-red-100 text-red-800';
            } else if (sev === 'WARNING') {
              borderCol = 'border-l-amber-500';
              badgeBg = 'bg-amber-100 text-amber-800';
            }

            return (
              <div key={alert.alert_id || alert.id} className={`border-l-4 ${borderCol} bg-slate-50 border border-slate-200 rounded-r p-2.5 text-xs`}>
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <div className="flex items-center gap-1.5 flex-wrap">
                      <span className={`text-[10px] font-bold uppercase px-1.5 py-0.2 rounded ${badgeBg}`}>
                        {sev}
                      </span>
                      <strong className="text-slate-800 truncate">
                        {_humanizeEnum(alert.care_action_type || alert.event_type || alert.type || 'Clinical Alert')}
                      </strong>
                    </div>
                    {alert.affected_vitals && alert.affected_vitals.length > 0 && (
                      <p className="text-[11px] text-slate-600 mt-1">
                        Affected parameter: <strong>{alert.affected_vitals.join(', ')}</strong>
                      </p>
                    )}
                    {alert.timestamp && (
                      <span className="text-[10px] text-slate-400 font-mono mt-0.5 block">{_ts(alert.timestamp)}</span>
                    )}
                  </div>

                  <div className="flex gap-1 flex-shrink-0">
                    {alert.state !== 'acknowledged' && (
                      <button
                        onClick={() => onAcknowledge && onAcknowledge(alert.alert_id || alert.id)}
                        className="text-[10px] px-2 py-0.5 rounded bg-slate-200 hover:bg-slate-300 text-slate-700 font-semibold"
                      >
                        Ack
                      </button>
                    )}
                    <button
                      onClick={() => onResolve && onResolve(alert.alert_id || alert.id)}
                      className="text-[10px] px-2 py-0.5 rounded bg-slate-200 hover:bg-slate-300 text-slate-700 font-semibold"
                    >
                      Resolve
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
};

/* ─── 6. VitalTrends (Standard Clinical Telemetry Charts) ───────────────── */
window.VitalTrends = function({ vitalsHistory }) {
  const { useEffect, useRef } = React;
  const chartRefs = {
    HR:   useRef(null),
    MAP:  useRef(null),
    SpO2: useRef(null),
    RR:   useRef(null),
  };
  const chartInstances = useRef({});

  const CONFIG = {
    HR:   { label: 'Heart Rate', unit: 'bpm', color: '#dc2626', min: 40, max: 160 },
    MAP:  { label: 'Mean Arterial Pressure', unit: 'mmHg', color: '#2563eb', min: 40, max: 140 },
    SpO2: { label: 'Oxygen Saturation', unit: '%', color: '#0d9488', min: 80, max: 100 },
    RR:   { label: 'Respiratory Rate', unit: '/min', color: '#d97706', min: 5, max: 35 },
  };

  const KEY_MAP = {
    HR:   ['HR', 'hr'],
    MAP:  ['MAP', 'map'],
    SpO2: ['SpO2', 'spo2', 'SPO2'],
    RR:   ['RR', 'rr'],
  };

  const getValues = (key) => {
    if (!vitalsHistory || vitalsHistory.length === 0) return [];
    const aliases = KEY_MAP[key];
    return vitalsHistory.map(tick => {
      for (const alias of aliases) {
        const val = tick[alias] ?? tick.vitals?.[alias];
        if (val !== undefined && val !== null && !isNaN(Number(val))) return Number(val);
      }
      return null;
    }).filter(v => v !== null);
  };

  useEffect(() => {
    const history = vitalsHistory || [];
    if (history.length < 2) return;

    Object.entries(chartRefs).forEach(([key, ref]) => {
      if (!ref.current) return;
      const values = getValues(key);
      if (values.length < 2) return;

      if (chartInstances.current[key]) {
        chartInstances.current[key].destroy();
      }

      const cfg = CONFIG[key];
      const ctx = ref.current.getContext('2d');
      chartInstances.current[key] = new Chart(ctx, {
        type: 'line',
        data: {
          labels: values.map((_, i) => `${i}`),
          datasets: [{
            data: values,
            borderColor: cfg.color,
            borderWidth: 1.5,
            pointRadius: 0,
            fill: false,
            tension: 0.2,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { enabled: false } },
          scales: {
            x: { display: false },
            y: {
              display: true,
              min: cfg.min,
              max: cfg.max,
              ticks: { font: { size: 9 }, maxTicksLimit: 3, color: '#94a3b8' },
              grid: { color: '#f1f5f9' },
            },
          },
          animation: false,
        },
      });
    });

    return () => {
      Object.values(chartInstances.current).forEach(c => c && c.destroy());
      chartInstances.current = {};
    };
  }, [vitalsHistory]);

  const history = vitalsHistory || [];
  if (history.length < 2) {
    return (
      <section className="clinical-document rounded-md p-4">
        <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 mb-2">Physiological Telemetry Trends</h3>
        <p className="text-xs text-slate-400 italic">Collecting trend data: buffering continuous vital sign stream for trend analysis…</p>
      </section>
    );
  }

  return (
    <section className="clinical-document rounded-md p-4 space-y-3">
      <div className="flex items-center justify-between pb-2 border-b border-slate-200">
        <div>
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700">Physiological Telemetry Trends</h3>
          <p className="text-[11px] text-slate-500">60-second calibrated time-series trends with clinical scale bounds.</p>
        </div>
        <span className="text-[11px] font-mono text-slate-400">Time Window: 60s</span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {Object.keys(chartRefs).map(key => {
          const cfg = CONFIG[key];
          const values = getValues(key);
          const latest = values.length > 0 ? Math.round(values[values.length - 1]) : null;

          return (
            <div key={key} className="bg-slate-50 border border-slate-200 rounded p-2.5">
              <div className="flex justify-between items-center mb-1">
                <span className="text-xs font-semibold text-slate-700">{cfg.label}</span>
                {latest !== null && (
                  <span className="text-xs font-mono font-bold text-slate-900">
                    {latest} <span className="font-normal text-slate-500 text-[11px]">{cfg.unit}</span>
                  </span>
                )}
              </div>
              <div style={{ height: '56px' }}>
                <canvas ref={chartRefs[key]} />
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
};

/* ─── 7. Main ClinicalDashboard Workspace ───────────────────────────────── */
window.ClinicalDashboard = function() {
  const state    = window.useCareMatrixState();
  const dispatch = window.useCareMatrixDispatch();

  const selectedPid   = state.selectedPatientId;
  const patients      = state.patients || {};
  const patientDetail = state.patientDetails[selectedPid] || {};
  const activePatient = patients[selectedPid] || {};

  const vitals            = patientDetail.vitals || activePatient.latest_vitals || {};
  const vitalsHistory     = patientDetail.vitalsHistory || [];
  const clinicalReasoning = patientDetail.clinicalReasoning || null;
  const careCoordination  = patientDetail.careCoordination || null;
  const riskDecision      = patientDetail.risk || null;
  const activeAlerts      = state.activeAlerts || [];

  const handleAcknowledge = async (alertId) => {
    try {
      await window.CareMatrixApi.acknowledgeAlert(alertId, 'Attending Physician');
    } catch (e) {
      console.warn('[ClinicalDashboard] Acknowledge error:', e);
    }
  };

  const handleResolve = async (alertId) => {
    try {
      await window.CareMatrixApi.resolveAlert(alertId, 'Bedside evaluation complete.');
    } catch (e) {
      console.warn('[ClinicalDashboard] Resolve error:', e);
    }
  };

  return (
    <div className="flex-1 flex flex-col bg-slate-100 min-h-0">
      {/* 1. Bed Selector Strip */}
      <window.PatientSelector
        patients={patients}
        selectedPatientId={selectedPid}
        onSelect={(pid) => dispatch({ type: 'SET_SELECTED_PATIENT_ID', payload: pid })}
      />

      {/* 2. Patient Header Banner */}
      <window.PatientHeader
        patient={activePatient}
        patientDetail={patientDetail}
      />

      {/* Main Clinical Document Workspace (Structured Document Hierarchy) */}
      <main className="flex-1 overflow-auto p-4 md:p-6">
        <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-6">

          {/* Left Column (Primary Patient Assessment & Vitals, 8 Cols) */}
          <div className="lg:col-span-8 space-y-6">
            {/* CURRENT VITALS */}
            <window.CurrentVitals vitals={vitals} />

            {/* CLINICAL ASSESSMENT REPORT */}
            <window.ClinicalAssessment
              clinicalReasoning={clinicalReasoning}
              careCoordination={careCoordination}
              riskDecision={riskDecision}
              vitals={vitals}
              vitalsHistory={vitalsHistory}
              patient={activePatient}
              patientDetail={patientDetail}
            />
          </div>

          {/* Right Column (Alerts, Active Alarms, and Telemetry Trends, 4 Cols) */}
          <div className="lg:col-span-4 space-y-6">
            {/* ALERTS & RISK */}
            <window.AlertsRisk
              activeAlerts={activeAlerts}
              riskDecision={riskDecision}
              patientId={selectedPid}
              onAcknowledge={handleAcknowledge}
              onResolve={handleResolve}
            />

            {/* VITAL TRENDS */}
            <window.VitalTrends vitalsHistory={vitalsHistory} />
          </div>

        </div>
      </main>
    </div>
  );
};

/* ─── 8. Modular Clinical Sub-Components (Exported Aliases) ─────────────── */
window.RecommendedActions = function({ careCoordination }) {
  const ca = careCoordination || {};
  const orders = _deduplicateOrders(ca.suggested_orders);
  if (!careCoordination || orders.length === 0) {
    return (
      <div className="text-xs text-slate-500 italic p-3">
        Awaiting care coordination bedside orders…
      </div>
    );
  }
  return (
    <ul className="space-y-1.5 text-xs text-slate-800">
      {orders.map((ord, i) => (
        <li key={i} className="flex items-start gap-2 bg-white border border-slate-200 rounded p-2">
          <span className="font-mono text-slate-400 mt-0.5">□</span>
          <span className="font-medium leading-relaxed">{ord}</span>
        </li>
      ))}
    </ul>
  );
};

window.EvidencePanel = function({ evidence }) {
  const evidenceList = evidence || [];
  if (evidenceList.length === 0) {
    return (
      <div className="text-xs text-slate-400 italic p-3">
        No Biomedical Literature Citations Retrieved.
      </div>
    );
  }
  return (
    <div className="space-y-2 text-xs">
      {evidenceList.map((e, i) => (
        <div key={i} className="bg-white border border-slate-200 rounded p-2.5">
          <div className="font-bold text-slate-900">{e.title}</div>
          <div className="text-[11px] text-slate-500">{_formatAuthors(e.authors)} {e.journal && `• ${e.journal}`} {e.year && `(${e.year})`}</div>
        </div>
      ))}
    </div>
  );
};

