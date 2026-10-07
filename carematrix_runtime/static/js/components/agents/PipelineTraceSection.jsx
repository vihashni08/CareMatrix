/**
 * Pipeline Execution Nodes and Safety Arbitration Components.
 */

window.PipelineTraceSection = function({
  monitoringEvent,
  riskDecision,
  riskModelLabel,
  dataAnalysis,
  clinicalReasoning,
  careAction,
  reportMetadata,
  reportStatus,
  isConflicting,
  isAdaptiveRAGSkipped,
  pipelineState
}) {
  return (
    <div className="bg-medCard border border-medBorder rounded-2xl p-5 shadow-xl space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-medBorder pb-3 gap-2">
        <div>
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            Adaptive Multi-Agent Execution Pipeline
            <span className="text-xs font-normal text-slate-400">
              (Cooperating Autonomous Agents)
            </span>
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Dynamically gates computational stages based on predicted risk, evidence consistency, and vital sign stability.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          {isAdaptiveRAGSkipped ? (
            <span className="text-xs font-bold font-mono px-3 py-1 rounded-full bg-sky-950 text-sky-300 border border-sky-800">
              ADAPTIVE GATING: RAG BYPASSED (LOW RISK CONFIDENT)
            </span>
          ) : (
            <span className="text-xs font-bold font-mono px-3 py-1 rounded-full bg-purple-950 text-purple-300 border border-purple-800">
              ADAPTIVE GATING: FULL PIPELINE ENGAGED (RETRIEVAL ACTIVE)
            </span>
          )}
        </div>
      </div>

      {/* 6 Sequential Nodes */}
      <div className="grid grid-cols-1 md:grid-cols-6 gap-3 text-xs">
        {/* 1. Monitoring Agent */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-3 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[10px] font-bold font-mono text-sky-400 px-1.5 py-0.5 rounded bg-sky-950 border border-sky-800">
                1. MONITOR
              </span>
              <span className="text-[10px] font-semibold text-emerald-400 font-mono">
                {pipelineState?.monitoring || 'COMPLETED'}
              </span>
            </div>
            <div className="font-bold text-slate-100">Monitoring Agent</div>
            <p className="text-[11px] text-slate-400 mt-1">
              {monitoringEvent ? `Detected: ${monitoringEvent.affected_vitals?.join(', ') || 'No breaches'}` : 'Vitals surveillance within limits'}
            </p>
          </div>
          <div className="text-[10px] font-mono text-slate-500 mt-2 border-t border-slate-800 pt-1">
            Status: Active Loop
          </div>
        </div>

        {/* 2. Risk Agent */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-3 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[10px] font-bold font-mono text-indigo-400 px-1.5 py-0.5 rounded bg-indigo-950 border border-indigo-800">
                2. RISK
              </span>
              <span className="text-[10px] font-semibold text-emerald-400 font-mono">
                {pipelineState?.risk || 'COMPLETED'}
              </span>
            </div>
            <div className="font-bold text-slate-100">Risk Agent (ML)</div>
            <p className="text-[11px] text-slate-400 mt-1">
              {riskDecision ? `Prob: ${(riskDecision.risk_probability * 100).toFixed(1)}% (${riskDecision.risk_level})` : 'Prediction complete'}
            </p>
          </div>
          <div className="text-[10px] font-mono text-slate-500 mt-2 border-t border-slate-800 pt-1">
            Model: {riskModelLabel}
          </div>
        </div>

        {/* 3. Data Analysis Agent */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-3 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[10px] font-bold font-mono text-purple-400 px-1.5 py-0.5 rounded bg-purple-950 border border-purple-800">
                3. ANALYSIS
              </span>
              <span className="text-[10px] font-semibold text-emerald-400 font-mono">
                {pipelineState?.dataAnalysis || 'COMPLETED'}
              </span>
            </div>
            <div className="font-bold text-slate-100">Data Analysis</div>
            <p className="text-[11px] text-slate-400 mt-1 line-clamp-2">
              {dataAnalysis?.pattern_identified?.[0] || 'Dynamic window slope analysis'}
            </p>
          </div>
          <div className="text-[10px] font-mono text-slate-500 mt-2 border-t border-slate-800 pt-1">
            Slopes & Deltas
          </div>
        </div>

        {/* 4. Cross-Agent Verification */}
        <div className={`rounded-xl p-3 flex flex-col justify-between border ${
          isConflicting ? 'bg-amber-950/40 border-amber-700/80' : 'bg-slate-900/90 border-slate-800'
        }`}>
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[10px] font-bold font-mono text-amber-400 px-1.5 py-0.5 rounded bg-amber-950 border border-amber-800">
                4. VERIFY
              </span>
              <span className={`text-[10px] font-bold font-mono ${isConflicting ? 'text-amber-400' : 'text-emerald-400'}`}>
                {reportStatus?.evidence_consistency || 'CONSISTENT'}
              </span>
            </div>
            <div className="font-bold text-slate-100">Cross-Agent Check</div>
            <p className="text-[11px] text-slate-400 mt-1">
              {isConflicting ? 'Risk vs Trend discordance flagged' : 'Corroborating physiological evidence'}
            </p>
          </div>
          <div className="text-[10px] font-mono text-slate-500 mt-2 border-t border-slate-800 pt-1">
            Integrity Check
          </div>
        </div>

        {/* 5. Clinical Reasoning Agent */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-3 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[10px] font-bold font-mono text-pink-400 px-1.5 py-0.5 rounded bg-pink-950 border border-pink-800">
                5. REASON
              </span>
              <span className="text-[10px] font-semibold text-emerald-400 font-mono">
                {pipelineState?.clinicalReasoning || 'COMPLETED'}
              </span>
            </div>
            <div className="font-bold text-slate-100">Clinical Reasoning</div>
            <p className="text-[11px] text-slate-400 mt-1">
              Mode: <b className="text-pink-300 font-mono">{reportMetadata?.reasoning_mode || clinicalReasoning?.metadata?.reasoning_mode || 'DETERMINISTIC'}</b>
            </p>
          </div>
          <div className="text-[10px] font-mono text-slate-500 mt-2 border-t border-slate-800 pt-1">
            {isAdaptiveRAGSkipped ? 'RAG: Skipped' : 'RAG: Active'}
          </div>
        </div>

        {/* 6. Care Coordination Agent */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-3 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[10px] font-bold font-mono text-emerald-400 px-1.5 py-0.5 rounded bg-emerald-950 border border-emerald-800">
                6. ACTION
              </span>
              <span className="text-[10px] font-semibold text-emerald-400 font-mono">
                {pipelineState?.careCoordination || 'COMPLETED'}
              </span>
            </div>
            <div className="font-bold text-slate-100">Care Coordination</div>
            <p className="text-[11px] text-slate-400 mt-1 line-clamp-2">
              {careAction?.action_type || 'CONTINUE_ROUTINE_MONITORING'}
            </p>
          </div>
          <div className="text-[10px] font-mono text-slate-500 mt-2 border-t border-slate-800 pt-1">
            Priority: {careAction?.priority || 'ROUTINE'}
          </div>
        </div>
      </div>
    </div>
  );
};

window.SafetyArbitrationPanel = function({
  riskDecision,
  riskModelLabel,
  reportStatus,
  isConflicting,
  isDataCompromised,
  dataAnalysis,
  isSafetyArbitrated,
  clinicalReasoning
}) {
  return (
    <div className="bg-medCard border border-medBorder rounded-2xl p-5 shadow-xl space-y-4">
      <div className="flex items-center justify-between border-b border-medBorder pb-3">
        <div>
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            Cross-Agent Verification & Deterministic Safety Arbitration
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Direct contrast of predictive ML model risk against analytical temporal trajectory findings.
          </p>
        </div>
        <span className={`text-xs px-3 py-1 rounded-full font-bold uppercase ${
          isConflicting
            ? 'bg-amber-950 text-amber-300 border border-amber-700 animate-pulse'
            : 'bg-emerald-950 text-emerald-300 border border-emerald-700'
        }`}>
          {reportStatus?.evidence_consistency === 'SUPPORTING' ? '✓ Evidence Supporting' : `⚠ ${reportStatus?.evidence_consistency}`}
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Left: Risk Agent Conclusion */}
        <div className="bg-slate-900 p-4 rounded-xl border border-indigo-900/60 flex flex-col justify-between">
          <div>
            <div className="text-[11px] font-bold text-indigo-400 uppercase tracking-wider mb-1">
              Risk Agent Prediction (ML)
            </div>
            <div className="text-xl font-extrabold text-white">
              {riskDecision?.risk_level || reportStatus?.risk_level}
            </div>
            <div className="text-xs text-slate-400 mt-1">
              Probability: <b className="text-slate-200 font-mono">
                {riskDecision?.risk_probability != null ? `${(riskDecision.risk_probability * 100).toFixed(1)}%` : 'N/A'}
              </b>
            </div>
            <div className="text-[11px] text-slate-400 mt-1">
              Model: <span className="font-mono text-slate-300">{riskModelLabel}</span>
            </div>
          </div>
          <div className="text-[10px] text-slate-500 border-t border-slate-800 pt-2 mt-3">
            Independent static feature inference
          </div>
        </div>

        {/* Center: Verification Determination */}
        <div className="bg-slate-900 p-4 rounded-xl border border-slate-800 flex flex-col justify-between items-center text-center">
          <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-1">
            Cross-Agent Consistency
          </div>
          <div className={`text-2xl font-black mt-1 ${
            isConflicting ? 'text-amber-400' : 'text-emerald-400'
          }`}>
            {reportStatus?.evidence_consistency}
          </div>
          <div className="text-xs text-slate-300 mt-2 px-2">
            {isConflicting
              ? 'Discordance detected between ML risk and temporal vital trends.'
              : 'Analytical trajectory confirms model risk classification.'}
          </div>
          <div className="text-[10px] font-mono text-slate-400 border-t border-slate-800 pt-2 mt-3 w-full">
            Reliability: <b className={isDataCompromised ? 'text-amber-400' : 'text-emerald-400'}>{reportStatus?.data_reliability}</b>
          </div>
        </div>

        {/* Right: Data Analysis Agent Trajectory */}
        <div className="bg-slate-900 p-4 rounded-xl border border-purple-900/60 flex flex-col justify-between">
          <div>
            <div className="text-[11px] font-bold text-purple-400 uppercase tracking-wider mb-1">
              Data Analysis Trajectory
            </div>
            <div className="text-sm font-bold text-white line-clamp-2">
              {dataAnalysis?.pattern_identified?.[0] || 'Physiological metrics stable across window.'}
            </div>
            <div className="text-xs text-slate-400 mt-2 space-y-0.5">
              {dataAnalysis?.trend_metrics?.MAP && (
                <div>MAP: <b className="text-slate-200">{dataAnalysis.trend_metrics.MAP.trend}</b> (latest: {dataAnalysis.trend_metrics.MAP.latest_value})</div>
              )}
              {dataAnalysis?.trend_metrics?.HR && (
                <div>HR: <b className="text-slate-200">{dataAnalysis.trend_metrics.HR.trend}</b> (latest: {dataAnalysis.trend_metrics.HR.latest_value})</div>
              )}
            </div>
          </div>
          <div className="text-[10px] text-slate-500 border-t border-slate-800 pt-2 mt-3">
            Windowed rate-of-change analysis
          </div>
        </div>
      </div>

      {/* Safety Override Alert Box */}
      {isSafetyArbitrated ? (
        <div className="bg-red-950/40 border border-red-700/80 rounded-xl p-3.5 flex items-start space-x-3">
          <span className="text-lg">🛡️</span>
          <div className="text-xs text-red-200 flex-1">
            <div className="font-bold uppercase tracking-wider text-red-300">
              Deterministic Safety Arbitration Override Active
            </div>
            <p className="mt-0.5 text-slate-300">
              {clinicalReasoning?.metadata?.final_priority_source === 'DETERMINISTIC_SAFETY_ARBITRATION'
                ? 'The reasoning engine detected an attempt to downgrade priority contrary to deterministic safety rules. The deterministic safety baseline has overridden the decision to protect patient safety.'
                : 'Deterministic safety rules actively serving as final boundary constraint.'}
            </p>
          </div>
        </div>
      ) : (
        <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-3 flex items-center justify-between text-xs text-slate-400">
          <div className="flex items-center space-x-2">
            <span className="text-sm">🛡️</span>
            <span>
              <b className="text-slate-300">Deterministic Safety Arbitration:</b> Passive surveillance mode (no override triggered — model outputs within safety envelopes).
            </span>
          </div>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800">
            SAFETY ENVELOPE VERIFIED
          </span>
        </div>
      )}
    </div>
  );
};

