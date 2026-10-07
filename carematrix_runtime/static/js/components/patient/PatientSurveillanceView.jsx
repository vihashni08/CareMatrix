/**
 * Patient Surveillance View — Dedicated Patient Telemetry, Waveform,
 * and 5-Agent Decision Summary Cards.
 */

window.PatientSurveillanceView = function() {
  const state = window.useCareMatrixState();
  const dispatch = window.useCareMatrixDispatch();

  const selectedPid = state.selectedPatientId;
  const patientDetail = state.patientDetails[selectedPid] || {};
  const activePatient = state.patients[selectedPid] || {};

  const vitals = patientDetail.vitals || activePatient.latest_vitals || {};
  const vitalsHistory = patientDetail.vitalsHistory || [];
  const careAction = patientDetail.careCoordination;
  const riskDecision = patientDetail.risk;
  const dataAnalysis = patientDetail.dataAnalysis;
  const clinicalReasoning = patientDetail.clinicalReasoning;
  const monitoringEvent = patientDetail.monitoring;

  const riskModelName = riskDecision?.model_name || patientDetail.risk_model_name ||
    activePatient.risk_model_name || state.system.riskModelName || 'Model loading';
  const riskModelLabel = window.CareMatrixFormatting.formatModelName(riskModelName);

  const clinicalReport = clinicalReasoning?.clinical_report || null;
  const rep = clinicalReport || {};
  const reportStatus = rep.clinical_status || {
    risk_level: clinicalReasoning?.risk_level || riskDecision?.risk_level || 'LOW RISK',
    priority: clinicalReasoning?.priority || 'ROUTINE',
    confidence: clinicalReasoning?.confidence || 0.95,
    data_reliability: clinicalReasoning?.data_reliability || 'HIGH',
    evidence_consistency: clinicalReasoning?.evidence_consistency || 'SUPPORTING',
  };

  const isHighRisk = (reportStatus.risk_level || '').includes('HIGH');
  const isUrgent = reportStatus.priority === 'URGENT' || activePatient?.status === 'ALERT';
  const isElevated = reportStatus.priority === 'ELEVATED' || activePatient?.status === 'SURVEILLANCE';
  const isDataCompromised = reportStatus.data_reliability === 'COMPROMISED';
  const isConflicting = reportStatus.evidence_consistency === 'CONFLICTING';

  const reportMetadata = clinicalReasoning?.metadata || {};
  const isSafetyArbitrated = reportMetadata?.safety_arbitration_applied ||
    (clinicalReasoning?.metadata?.safety_arbitration_applied) ||
    (rep?.executive_summary || '').startsWith('[SAFETY ARBITRATION]');

  // Alert actions
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

  const toggleSection = (key) => dispatch({ type: 'TOGGLE_SECTION', payload: key });
  const setAllSections = (val) => dispatch({ type: 'SET_ALL_SECTIONS', payload: val });
  const toggleOrder = (pid, idx) => dispatch({ type: 'TOGGLE_ORDER_CHECK', payload: { patientId: pid, index: idx } });

  const navigateToAgent = (agentName) => {
    dispatch({ type: 'SET_SELECTED_AGENT', payload: agentName });
    dispatch({ type: 'SET_ACTIVE_TAB', payload: 'agent_inspection' });
  };

  return (
    <main className="flex-1 p-6 space-y-6 max-w-7xl mx-auto w-full">
      {/* ===================================================================== */}
      {/* 1. PATIENT HEADER BANNER                                              */}
      {/* ===================================================================== */}
      <div className="bg-medCard border border-medBorder rounded-2xl p-5 shadow-xl relative overflow-hidden">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center space-x-3">
              <h2 className="text-2xl font-black text-white tracking-tight">
                {patientDetail.name || `Bed ${selectedPid}`}
              </h2>
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                CASE ID: {selectedPid}
              </span>
              {patientDetail.data_source && (
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-teal-950 text-teal-300 border border-teal-800">
                  SOURCE: {patientDetail.data_source.toUpperCase()}
                </span>
              )}
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Active Bedside Surveillance • Coordinated 5-Agent Event Processing
            </p>
          </div>

          {/* Clinical Hierarchy Badges */}
          <div className="flex flex-wrap items-center gap-2">
            <div className={`px-3 py-1.5 rounded-xl border font-bold text-xs flex items-center space-x-1.5 shadow ${
              isHighRisk
                ? 'bg-red-950/80 text-red-200 border-red-700 animate-pulse'
                : 'bg-emerald-950/60 text-emerald-300 border-emerald-700'
            }`}>
              <span className={`h-2 w-2 rounded-full ${isHighRisk ? 'bg-red-400' : 'bg-emerald-400'}`}></span>
              <span>RISK: {reportStatus.risk_level}</span>
            </div>

            <div className={`px-3 py-1.5 rounded-xl border font-bold text-xs uppercase ${
              isUrgent
                ? 'bg-red-600 text-white border-red-500 shadow-md shadow-red-600/30'
                : isElevated
                ? 'bg-amber-500 text-black border-amber-400 font-extrabold shadow-md shadow-amber-500/20'
                : 'bg-slate-800 text-slate-300 border-slate-700'
            }`}>
              PRIORITY: {reportStatus.priority}
            </div>

            <div className={`px-3 py-1.5 rounded-xl border font-bold text-xs ${
              isConflicting
                ? 'bg-amber-950/80 text-amber-300 border-amber-700'
                : 'bg-emerald-950/60 text-emerald-300 border-emerald-700'
            }`}>
              EVIDENCE: {reportStatus.evidence_consistency}
            </div>

            <div className="px-3 py-1.5 rounded-xl border border-slate-700 bg-slate-900 text-xs font-mono text-slate-200">
              CONFIDENCE: <b className="text-sky-400">{(reportStatus.confidence * 100).toFixed(0)}%</b>
            </div>
          </div>
        </div>
      </div>

      {/* ===================================================================== */}
      {/* 2. LIVE VITALS & CHART.JS WAVEFORM                                    */}
      {/* ===================================================================== */}
      <window.PatientVitalsSection
        vitals={vitals}
        vitalsHistory={vitalsHistory}
        dataAnalysis={dataAnalysis}
      />

      {/* ===================================================================== */}
      {/* 3. 5-AGENT DECISION SUMMARY CARDS                                     */}
      {/* ===================================================================== */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* Card 1: Monitoring Agent */}
        <div className="bg-medCard border border-medBorder rounded-xl p-4.5 flex flex-col justify-between shadow-lg">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-[11px] font-bold text-sky-400 uppercase tracking-wider">
                Monitoring Agent
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-sky-950 text-sky-300 border border-sky-800">
                Tier 1 Anomaly
              </span>
            </div>

            <div className="text-sm font-bold text-white">
              Decision: <span className="font-mono text-sky-300">{monitoringEvent?.recommended_action || 'CONTINUE_MONITORING'}</span>
            </div>

            <div className="mt-2.5 text-xs text-slate-300 space-y-1">
              <div>Affected: <b className="text-slate-100">{monitoringEvent?.affected_vitals?.join(', ') || 'None (Normal)'}</b></div>
              <div>Severity: <b className="text-slate-100 uppercase">{monitoringEvent?.severity || 'Normal'}</b></div>
              <div>Signal Quality: <b className="text-slate-100 font-mono">{monitoringEvent?.signal_quality?.HR || 'Good'}</b></div>
            </div>

            {/* Tool execution badge */}
            <div className="mt-3 p-2 bg-slate-900/90 rounded-lg border border-slate-800 text-[11px]">
              <span className="text-slate-400">Tool Calls: </span>
              <b className="text-sky-400 font-mono">
                {patientDetail.agenticTraces?.monitoring?.length || 0} executed
              </b>
              <span className="text-slate-500 text-[10px] ml-1.5">(Preprocess, Baseline, Trend)</span>
            </div>
          </div>

          <button
            onClick={() => navigateToAgent('MonitoringAgent')}
            className="mt-3 text-xs text-sky-400 hover:text-sky-300 font-semibold text-left flex items-center gap-1"
          >
            Inspect Monitoring Trace →
          </button>
        </div>

        {/* Card 2: Risk Agent */}
        <div className="bg-medCard border border-medBorder rounded-xl p-4.5 flex flex-col justify-between shadow-lg">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-[11px] font-bold text-indigo-400 uppercase tracking-wider">
                Risk Agent
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-800">
                Trained ML Model
              </span>
            </div>

            <div className="text-sm font-bold text-white">
              Risk: <span className={`font-mono ${isHighRisk ? 'text-red-400' : 'text-emerald-400'}`}>
                {riskDecision?.risk_level || 'LOW RISK'}
              </span>
            </div>

            <div className="mt-2.5 text-xs text-slate-300 space-y-1">
              <div>Probability: <b className="text-slate-100 font-mono">
                {riskDecision?.risk_probability != null ? `${(riskDecision.risk_probability * 100).toFixed(1)}%` : '--'}
              </b></div>
              <div>Model: <b className="text-slate-100">{riskModelLabel}</b></div>
              <div>Threshold: <b className="text-slate-100 font-mono">0.65</b></div>
            </div>

            {/* Safety Floor Override indicator */}
            <div className="mt-3 p-2 bg-slate-900/90 rounded-lg border border-slate-800 text-[11px]">
              <span className="text-slate-400">Safety Floor Override: </span>
              <b className={`font-mono ${riskDecision?.risk_probability >= 0.85 && (vitals.map < 65) ? 'text-red-400' : 'text-slate-300'}`}>
                {riskDecision?.risk_probability >= 0.85 && (vitals.map < 65) ? 'ENFORCED (MAP < 65)' : 'PASSIVE'}
              </b>
            </div>
          </div>

          <button
            onClick={() => navigateToAgent('RiskAgent')}
            className="mt-3 text-xs text-indigo-400 hover:text-indigo-300 font-semibold text-left flex items-center gap-1"
          >
            Inspect Risk Predictions →
          </button>
        </div>

        {/* Card 3: Data Analysis Agent */}
        <div className="bg-medCard border border-medBorder rounded-xl p-4.5 flex flex-col justify-between shadow-lg">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-[11px] font-bold text-purple-400 uppercase tracking-wider">
                Data Analysis Agent
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-purple-950 text-purple-300 border border-purple-800">
                Temporal Trajectory
              </span>
            </div>

            <div className="text-sm font-bold text-white">
              Data Quality: <span className={`font-mono ${dataAnalysis?.data_quality_flag ? 'text-amber-400' : 'text-emerald-400'}`}>
                {dataAnalysis?.data_quality_flag ? 'FLAGGED' : 'VALID'}
              </span>
            </div>

            <div className="mt-2.5 text-xs text-slate-300 space-y-1">
              <div>Pattern: <b className="text-slate-100 truncate block">{dataAnalysis?.pattern_identified?.[0] || 'Stable dynamics'}</b></div>
              <div>MAP Slope: <b className="text-slate-100 font-mono">{dataAnalysis?.trend_metrics?.MAP?.trend || 'stable'}</b></div>
              <div>Consistency: <b className="text-slate-100">{reportStatus.evidence_consistency}</b></div>
            </div>

            {/* Negotiation State */}
            <div className="mt-3 p-2 bg-slate-900/90 rounded-lg border border-slate-800 text-[11px]">
              <span className="text-slate-400">Agent Challenge: </span>
              <b className="text-purple-300 font-mono">
                {patientDetail.negotiationTrace?.length > 0 ? `Active (${patientDetail.negotiationTrace.length} rounds)` : 'Consensus Reached'}
              </b>
            </div>
          </div>

          <button
            onClick={() => navigateToAgent('DataAnalysisAgent')}
            className="mt-3 text-xs text-purple-400 hover:text-purple-300 font-semibold text-left flex items-center gap-1"
          >
            Inspect Negotiation Trace →
          </button>
        </div>

        {/* Card 4: Clinical Reasoning Agent */}
        <div className="bg-medCard border border-medBorder rounded-xl p-4.5 flex flex-col justify-between shadow-lg">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-[11px] font-bold text-pink-400 uppercase tracking-wider">
                Clinical Reasoning Agent
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-pink-950 text-pink-300 border border-pink-800">
                PubMed RAG Grounded
              </span>
            </div>

            <div className="text-sm font-bold text-white">
              Mode: <span className="font-mono text-pink-300">{reportMetadata?.reasoning_mode || 'DETERMINISTIC'}</span>
            </div>

            <div className="mt-2.5 text-xs text-slate-300 space-y-1">
              <div>RAG Literature: <b className="text-teal-300 font-mono">
                {rep.medical_evidence?.length ? `${rep.medical_evidence.length} citations` : 'Adaptive Bypass'}
              </b></div>
              <div>Safety Arbitration: <b className={`font-mono ${isSafetyArbitrated ? 'text-red-400' : 'text-emerald-400'}`}>
                {isSafetyArbitrated ? 'OVERRIDE APPLIED' : 'VERIFIED'}
              </b></div>
              <div>Confidence: <b className="text-sky-400 font-mono">{(reportStatus.confidence * 100).toFixed(0)}%</b></div>
            </div>

            <div className="mt-3 p-2 bg-slate-900/90 rounded-lg border border-slate-800 text-[11px]">
              <span className="text-slate-400">Agentic Tool Calls: </span>
              <b className="text-pink-400 font-mono">
                {patientDetail.agenticTraces?.clinicalReasoning?.length || 0} executed
              </b>
            </div>
          </div>

          <button
            onClick={() => navigateToAgent('ClinicalReasoningAgent')}
            className="mt-3 text-xs text-pink-400 hover:text-pink-300 font-semibold text-left flex items-center gap-1"
          >
            Inspect Clinical Reasoning →
          </button>
        </div>

        {/* Card 5: Care Coordination Agent */}
        <div className="bg-medCard border border-medBorder rounded-xl p-4.5 flex flex-col justify-between shadow-lg md:col-span-2 lg:col-span-2">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-[11px] font-bold text-emerald-400 uppercase tracking-wider">
                Care Coordination Agent
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800">
                Bedside Tasking
              </span>
            </div>

            <div className="text-sm font-bold text-white">
              Pathway: <span className="text-sky-300 font-mono">{careAction?.escalation_pathway || 'Continuous Surveillance'}</span>
            </div>

            <div className="mt-2.5 text-xs text-slate-300 space-y-1">
              <div>Action: <b className="text-slate-100">{careAction?.action_type || 'CONTINUE_ROUTINE_MONITORING'}</b></div>
              <div>Clinician Review Required: <b className={`font-mono ${careAction?.clinician_review_required ? 'text-red-400' : 'text-slate-400'}`}>
                {careAction?.clinician_review_required ? 'YES (ACTION REQUIRED)' : 'NO'}
              </b></div>
              <div>Orders Suggested: <b className="text-slate-100">{careAction?.suggested_orders?.length || 0} tasks</b></div>
            </div>

            <div className="mt-3 p-2 bg-slate-900/90 rounded-lg border border-slate-800 text-[11px]">
              <span className="text-slate-400">Clinical Reason: </span>
              <span className="text-slate-200 italic">{careAction?.reason || 'Hemodynamic stability maintained.'}</span>
            </div>
          </div>

          <button
            onClick={() => navigateToAgent('CareCoordinationAgent')}
            className="mt-3 text-xs text-emerald-400 hover:text-emerald-300 font-semibold text-left flex items-center gap-1"
          >
            Inspect Care Actions →
          </button>
        </div>
      </div>

      {/* ===================================================================== */}
      {/* 4. CLINICAL REPORT & CARE ACTION PLAN                                 */}
      {/* ===================================================================== */}
      <window.ClinicalReportSection
        clinicalReport={clinicalReport}
        clinicalReasoning={clinicalReasoning}
        riskDecision={riskDecision}
        reportMetadata={reportMetadata}
        expandedSections={state.ui.expandedSections}
        toggleSection={toggleSection}
        setAllSections={setAllSections}
      />

      <window.CareCoordinationSection
        careAction={careAction}
        activeAlerts={patientDetail.alerts || []}
        selectedPid={selectedPid}
        orderChecked={state.ui.orderChecked}
        onToggleOrder={toggleOrder}
        onAcknowledgeAlert={handleAcknowledgeAlert}
        onResolveAlert={handleResolveAlert}
        isUrgent={isUrgent}
        isElevated={isElevated}
      />
    </main>
  );
};

