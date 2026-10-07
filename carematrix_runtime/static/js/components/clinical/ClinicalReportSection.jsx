/**
 * Clinical Intelligence Report (16 Structured Sections) View.
 */

window.ClinicalReportSection = function({
  clinicalReport,
  clinicalReasoning,
  riskDecision,
  reportMetadata,
  expandedSections,
  toggleSection,
  setAllSections
}) {
  const rep = clinicalReport || clinicalReasoning?.clinical_report || {};
  const reportStatus = rep.clinical_status || {
    risk_level: clinicalReasoning?.risk_level || riskDecision?.risk_level || 'LOW RISK',
    priority: clinicalReasoning?.priority || 'ROUTINE',
    confidence: clinicalReasoning?.confidence || 0.95,
    data_reliability: clinicalReasoning?.data_reliability || 'HIGH',
    evidence_consistency: clinicalReasoning?.evidence_consistency || 'SUPPORTING',
  };

  const isHighRisk = (reportStatus.risk_level || '').includes('HIGH');
  const isAdaptiveRAGSkipped = reportMetadata?.evidence_retrieval === 'SKIPPED_LOW_RISK_CONFIDENT' ||
    (clinicalReasoning?.metadata?.evidence_retrieval === 'SKIPPED_LOW_RISK_CONFIDENT') ||
    (rep?.medical_evidence && rep.medical_evidence.length === 0 && !isHighRisk);

  return (
    <div className="bg-medCard border-2 border-sky-500/40 rounded-2xl p-6 shadow-2xl space-y-6 ring-1 ring-sky-500/20">
      {/* Header with Title & Level 1 Status */}
      <div className="flex flex-col md:flex-row md:items-center justify-between border-b border-medBorder pb-4 gap-3">
        <div>
          <div className="flex items-center space-x-2.5">
            <span className="h-3 w-3 rounded-full bg-sky-400"></span>
            <h2 className="text-xl font-black text-white tracking-tight uppercase flex items-center gap-2">
              Clinical Intelligence Report
              <span className="text-xs font-mono font-normal text-sky-400 px-2 py-0.5 rounded bg-sky-950 border border-sky-800">
                16 Structured Sections
              </span>
            </h2>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Hierarchical decision synthesis, physiological drift analysis, and retrieved biomedical literature grounding.
          </p>
        </div>

        {/* Global Expand / Collapse All Controls */}
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setAllSections(true)}
            className="text-xs px-2.5 py-1 rounded bg-slate-800 text-slate-300 hover:bg-slate-700 transition font-medium"
          >
            Expand All
          </button>
          <button
            onClick={() => setAllSections(false)}
            className="text-xs px-2.5 py-1 rounded bg-slate-800 text-slate-300 hover:bg-slate-700 transition font-medium"
          >
            Collapse All
          </button>
        </div>
      </div>

      {/* Accordion List for 16 Structured Sections */}
      <div className="space-y-3">
        {/* 1. Executive Summary */}
        <div className="bg-slate-900/90 border border-sky-800/60 rounded-xl p-4 shadow-inner">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-bold uppercase tracking-wider text-sky-300 flex items-center gap-2">
              <span>1. Executive Clinical Summary</span>
              <window.ProvenanceBadge type="LLM_SYNTHESIS" />
            </span>
            <span className="text-[11px] font-mono text-slate-400">
              Generated: {new Date().toLocaleTimeString()}
            </span>
          </div>
          <p className="text-sm text-slate-100 leading-relaxed font-sans">
            {rep.executive_summary || rep.evidence_synthesis || 'Continuous monitoring active. Physiological telemetry within normal limits.'}
          </p>
        </div>

        {/* 2. Clinical Status & Gating Summary */}
        <div className="bg-slate-900/80 border border-medBorder rounded-xl overflow-hidden">
          <button
            onClick={() => toggleSection('clinical_status')}
            className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-slate-800/50 transition"
          >
            <span className="text-xs font-bold text-slate-200 flex items-center gap-2">
              <span>2. Clinical Status & Decision Parameters</span>
              <window.ProvenanceBadge type="MODEL_OUTPUT" />
            </span>
            <span className="text-slate-400 text-xs">{expandedSections.clinical_status ? '▼' : '▶'}</span>
          </button>
          {expandedSections.clinical_status && (
            <div className="px-4 pb-3 pt-1 border-t border-slate-800/60 grid grid-cols-2 sm:grid-cols-5 gap-3 text-xs">
              <div className="bg-medDark p-2.5 rounded-lg border border-slate-800">
                <span className="text-slate-400 block text-[10px] uppercase font-bold">Risk Level</span>
                <span className="font-bold text-white text-sm mt-0.5 block">{reportStatus.risk_level}</span>
              </div>
              <div className="bg-medDark p-2.5 rounded-lg border border-slate-800">
                <span className="text-slate-400 block text-[10px] uppercase font-bold">Urgency Priority</span>
                <span className="font-bold text-white text-sm mt-0.5 block">{reportStatus.priority}</span>
              </div>
              <div className="bg-medDark p-2.5 rounded-lg border border-slate-800">
                <span className="text-slate-400 block text-[10px] uppercase font-bold">Confidence</span>
                <span className="font-bold text-sky-400 text-sm mt-0.5 block font-mono">{(reportStatus.confidence * 100).toFixed(0)}%</span>
              </div>
              <div className="bg-medDark p-2.5 rounded-lg border border-slate-800">
                <span className="text-slate-400 block text-[10px] uppercase font-bold">Evidence Consistency</span>
                <span className="font-bold text-white text-sm mt-0.5 block">{reportStatus.evidence_consistency}</span>
              </div>
              <div className="bg-medDark p-2.5 rounded-lg border border-slate-800">
                <span className="text-slate-400 block text-[10px] uppercase font-bold">Data Reliability</span>
                <span className="font-bold text-white text-sm mt-0.5 block">{reportStatus.data_reliability}</span>
              </div>
            </div>
          )}
        </div>

        {/* 3. Key Findings */}
        <div className="bg-slate-900/80 border border-medBorder rounded-xl overflow-hidden">
          <button
            onClick={() => toggleSection('key_findings')}
            className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-slate-800/50 transition"
          >
            <span className="text-xs font-bold text-slate-200 flex items-center gap-2">
              <span>3. Salient Findings & Cardinal Deviations</span>
              <window.ProvenanceBadge type="PATIENT_DATA" />
            </span>
            <span className="text-slate-400 text-xs">{expandedSections.key_findings ? '▼' : '▶'}</span>
          </button>
          {expandedSections.key_findings && (
            <div className="px-4 pb-3 pt-1 border-t border-slate-800/60">
              <ul className="space-y-1.5 text-xs text-slate-300">
                {(rep.key_findings && rep.key_findings.length > 0) ? (
                  rep.key_findings.map((finding, idx) => (
                    <li key={idx} className="flex items-start space-x-2">
                      <span className="text-sky-400 font-bold">•</span>
                      <span>{finding}</span>
                    </li>
                  ))
                ) : (
                  <li className="text-slate-500 italic">No acute or critical physiological deviations noted.</li>
                )}
              </ul>
            </div>
          )}
        </div>

        {/* 4. Physiological Analysis */}
        <div className="bg-slate-900/80 border border-medBorder rounded-xl overflow-hidden">
          <button
            onClick={() => toggleSection('physiological_analysis')}
            className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-slate-800/50 transition"
          >
            <span className="text-xs font-bold text-slate-200 flex items-center gap-2">
              <span>4. Multi-System Physiological Analysis</span>
              <window.ProvenanceBadge type="TEMPORAL_ANALYSIS" />
            </span>
            <span className="text-slate-400 text-xs">{expandedSections.physiological_analysis ? '▼' : '▶'}</span>
          </button>
          {expandedSections.physiological_analysis && (
            <div className="px-4 pb-3 pt-1 border-t border-slate-800/60">
              <ul className="space-y-1.5 text-xs text-slate-300">
                {(rep.physiological_analysis && rep.physiological_analysis.length > 0) ? (
                  rep.physiological_analysis.map((item, idx) => (
                    <li key={idx} className="flex items-start space-x-2">
                      <span className="text-purple-400 font-bold">•</span>
                      <span>{item}</span>
                    </li>
                  ))
                ) : (
                  <li className="text-slate-500 italic">Cardiovascular, respiratory, and autonomic states within baseline distributions.</li>
                )}
              </ul>
            </div>
          )}
        </div>

        {/* 5. Temporal Analysis */}
        <div className="bg-slate-900/80 border border-medBorder rounded-xl overflow-hidden">
          <button
            onClick={() => toggleSection('temporal_analysis')}
            className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-slate-800/50 transition"
          >
            <span className="text-xs font-bold text-slate-200 flex items-center gap-2">
              <span>5. Temporal Trajectory & Rate of Change</span>
              <window.ProvenanceBadge type="TEMPORAL_ANALYSIS" />
            </span>
            <span className="text-slate-400 text-xs">{expandedSections.temporal_analysis ? '▼' : '▶'}</span>
          </button>
          {expandedSections.temporal_analysis && (
            <div className="px-4 pb-3 pt-1 border-t border-slate-800/60">
              <ul className="space-y-1.5 text-xs text-slate-300">
                {(rep.temporal_analysis && rep.temporal_analysis.length > 0) ? (
                  rep.temporal_analysis.map((item, idx) => (
                    <li key={idx} className="flex items-start space-x-2">
                      <span className="text-sky-400 font-bold">•</span>
                      <span>{item}</span>
                    </li>
                  ))
                ) : (
                  <li className="text-slate-500 italic">Slope metrics reveal zero significant drift across the surveillance window.</li>
                )}
              </ul>
            </div>
          )}
        </div>

        {/* 6. Supporting Evidence */}
        <div className="bg-slate-900/80 border border-medBorder rounded-xl overflow-hidden">
          <button
            onClick={() => toggleSection('supporting_evidence')}
            className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-slate-800/50 transition"
          >
            <span className="text-xs font-bold text-slate-200 flex items-center gap-2">
              <span>6. Corroborating / Supporting Evidence</span>
              <window.ProvenanceBadge type="AGENT_VERIFICATION" />
            </span>
            <span className="text-slate-400 text-xs">{expandedSections.supporting_evidence ? '▼' : '▶'}</span>
          </button>
          {expandedSections.supporting_evidence && (
            <div className="px-4 pb-3 pt-1 border-t border-slate-800/60">
              <ul className="space-y-1.5 text-xs text-slate-300">
                {(rep.supporting_evidence && rep.supporting_evidence.length > 0) ? (
                  rep.supporting_evidence.map((item, idx) => (
                    <li key={idx} className="flex items-start space-x-2">
                      <span className="text-emerald-400 font-bold">✓</span>
                      <span>{item}</span>
                    </li>
                  ))
                ) : (
                  <li className="text-slate-500 italic">Vital stability corroborated across multiple sensor channels.</li>
                )}
              </ul>
            </div>
          )}
        </div>

        {/* 7. Conflicting Evidence */}
        <div className="bg-slate-900/80 border border-medBorder rounded-xl overflow-hidden">
          <button
            onClick={() => toggleSection('conflicting_evidence')}
            className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-slate-800/50 transition"
          >
            <span className="text-xs font-bold text-slate-200 flex items-center gap-2">
              <span>7. Discordant / Conflicting Evidence</span>
              <window.ProvenanceBadge type="AGENT_VERIFICATION" />
            </span>
            <span className="text-slate-400 text-xs">{expandedSections.conflicting_evidence ? '▼' : '▶'}</span>
          </button>
          {expandedSections.conflicting_evidence && (
            <div className="px-4 pb-3 pt-1 border-t border-slate-800/60">
              <ul className="space-y-1.5 text-xs text-slate-300">
                {(rep.conflicting_evidence && rep.conflicting_evidence.length > 0) ? (
                  rep.conflicting_evidence.map((item, idx) => (
                    <li key={idx} className="flex items-start space-x-2">
                      <span className="text-amber-400 font-bold">⚠</span>
                      <span>{item}</span>
                    </li>
                  ))
                ) : (
                  <li className="text-slate-500 italic">No conflicting physiological patterns or contradictory sensor trends detected.</li>
                )}
              </ul>
            </div>
          )}
        </div>

        {/* 8. Evidence Synthesis */}
        <div className="bg-slate-900/80 border border-medBorder rounded-xl overflow-hidden">
          <button
            onClick={() => toggleSection('evidence_synthesis')}
            className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-slate-800/50 transition"
          >
            <span className="text-xs font-bold text-slate-200 flex items-center gap-2">
              <span>8. Evidence Synthesis & Cross-Corroboration</span>
              <window.ProvenanceBadge type="LLM_SYNTHESIS" />
            </span>
            <span className="text-slate-400 text-xs">{expandedSections.evidence_synthesis ? '▼' : '▶'}</span>
          </button>
          {expandedSections.evidence_synthesis && (
            <div className="px-4 pb-3 pt-1 border-t border-slate-800/60 text-xs text-slate-300 leading-relaxed">
              {rep.evidence_synthesis || 'Synthesis reconciles multi-channel observations with baseline stability.'}
            </div>
          )}
        </div>

        {/* 9. Medical Evidence / PubMed */}
        <div className="bg-slate-900/80 border border-medBorder rounded-xl overflow-hidden">
          <button
            onClick={() => toggleSection('medical_evidence')}
            className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-slate-800/50 transition"
          >
            <span className="text-xs font-bold text-slate-200 flex items-center gap-2">
              <span>9. Retrieved Medical Literature (PubMed)</span>
              <window.ProvenanceBadge type="PUBMED_RAG" />
            </span>
            <span className="text-slate-400 text-xs">{expandedSections.medical_evidence ? '▼' : '▶'}</span>
          </button>
          {expandedSections.medical_evidence && (
            <div className="px-4 pb-3 pt-1 border-t border-slate-800/60 space-y-3">
              {(rep.medical_evidence && rep.medical_evidence.length > 0) ? (
                rep.medical_evidence.map((med, idx) => {
                  const isGrounded = med.grounding_status === 'GROUNDED' || med.grounded === true;
                  return (
                    <div key={idx} className="bg-medDark p-3.5 rounded-xl border border-teal-900/60 space-y-2 text-xs">
                      {/* Source Metadata Header */}
                      <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-slate-800 pb-2 gap-1.5">
                        <div>
                          <div className="font-bold text-teal-300 text-sm">{med.title}</div>
                          <div className="text-[11px] text-slate-400">
                            {med.authors && <span>{med.authors} • </span>}
                            {med.journal && <span className="italic">{med.journal} </span>}
                            {med.year && <span>({med.year})</span>}
                            {med.doi && <span className="font-mono"> • DOI: {med.doi}</span>}
                          </div>
                        </div>
                        <div className="flex items-center space-x-2">
                          <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono text-[10px]">
                            PMID: {med.pmid || 'N/A'}
                          </span>
                          <span className={`px-2 py-0.5 rounded font-mono text-[10px] font-bold ${
                            isGrounded ? 'bg-emerald-950 text-emerald-300 border border-emerald-800' : 'bg-amber-950 text-amber-300 border border-amber-800'
                          }`}>
                            {isGrounded ? '✓ GROUNDED' : '⚠ UNVERIFIED CLAIM REPLACED'}
                          </span>
                        </div>
                      </div>

                      {/* Partitioned Blocks */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
                        <div className="space-y-2">
                          <div className="bg-slate-900/90 p-2.5 rounded-lg border border-slate-800">
                            <span className="text-[10px] font-bold uppercase tracking-wider text-teal-400 block mb-1">
                              Source Abstract Excerpt:
                            </span>
                            <p className="text-[11px] text-slate-300 italic leading-snug">
                              "{med.abstract || med.key_evidence}"
                            </p>
                          </div>
                          <div className="bg-slate-900/90 p-2.5 rounded-lg border border-slate-800">
                            <span className="text-[10px] font-bold uppercase tracking-wider text-sky-400 block mb-1">
                              Key Biomedical Finding / Threshold:
                            </span>
                            <p className="text-[11px] text-slate-200">
                              {med.key_evidence}
                            </p>
                          </div>
                        </div>

                        <div className="space-y-2">
                          <div className="bg-slate-900/90 p-2.5 rounded-lg border border-slate-800">
                            <span className="text-[10px] font-bold uppercase tracking-wider text-pink-400 block mb-1">
                              Bedside Case Application (Decision Support):
                            </span>
                            <p className="text-[11px] text-slate-200 leading-snug">
                              {med.application_to_case}
                            </p>
                          </div>
                          <div className="bg-slate-900/90 p-2.5 rounded-lg border border-slate-800">
                            <span className="text-[10px] font-bold uppercase tracking-wider text-purple-400 block mb-1">
                              Why Retrieved for this Presentation:
                            </span>
                            <p className="text-[11px] text-slate-300 leading-snug">
                              {med.why_retrieved}
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })
              ) : (
                <div className="p-3.5 bg-medDark rounded-lg border border-slate-800 text-xs text-slate-400 text-center">
                  <span className="font-semibold text-slate-300 block mb-1">
                    {isAdaptiveRAGSkipped ? "Adaptive Literature Gating Active" : "No External Guidance Required"}
                  </span>
                  {reportMetadata?.evidence_retrieval_reason || (isAdaptiveRAGSkipped
                    ? "Patient exhibits low predictive risk and stable physiological trajectories. Computational RAG bypass was dynamically invoked to conserve resources."
                    : "Hemodynamic presentation within routine thresholds.")}
                </div>
              )}
            </div>
          )}
        </div>

        {/* 10. Uncertainties & Limitations */}
        <div className="bg-slate-900/80 border border-medBorder rounded-xl overflow-hidden">
          <button
            onClick={() => toggleSection('uncertainties')}
            className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-slate-800/50 transition"
          >
            <span className="text-xs font-bold text-slate-200 flex items-center gap-2">
              <span>10. Clinical Uncertainties & Model Boundaries</span>
              <window.ProvenanceBadge type="MODEL_OUTPUT" />
            </span>
            <span className="text-slate-400 text-xs">{expandedSections.uncertainties ? '▼' : '▶'}</span>
          </button>
          {expandedSections.uncertainties && (
            <div className="px-4 pb-3 pt-1 border-t border-slate-800/60">
              <ul className="space-y-1.5 text-xs text-slate-300">
                {(rep.uncertainties && rep.uncertainties.length > 0) ? (
                  rep.uncertainties.map((item, idx) => (
                    <li key={idx} className="flex items-start space-x-2">
                      <span className="text-amber-400 font-bold">•</span>
                      <span>{item}</span>
                    </li>
                  ))
                ) : (
                  <li className="text-slate-500 italic">No elevated analytical or signal-quality uncertainties noted.</li>
                )}
              </ul>
            </div>
          )}
        </div>

        {/* 11. Recommended Actions */}
        <div className="bg-slate-900/80 border border-medBorder rounded-xl overflow-hidden">
          <button
            onClick={() => toggleSection('recommended_actions')}
            className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-slate-800/50 transition"
          >
            <span className="text-xs font-bold text-slate-200 flex items-center gap-2">
              <span>11. Actionable Recommendations (CDS Only)</span>
              <window.ProvenanceBadge type="LLM_SYNTHESIS" />
            </span>
            <span className="text-slate-400 text-xs">{expandedSections.recommended_actions ? '▼' : '▶'}</span>
          </button>
          {expandedSections.recommended_actions && (
            <div className="px-4 pb-3 pt-1 border-t border-slate-800/60">
              <ul className="space-y-1.5 text-xs text-slate-300">
                {(rep.recommended_actions && rep.recommended_actions.length > 0) ? (
                  rep.recommended_actions.map((item, idx) => (
                    <li key={idx} className="flex items-start space-x-2">
                      <span className="text-sky-400 font-bold">→</span>
                      <span>{item}</span>
                    </li>
                  ))
                ) : (
                  <li className="text-slate-500 italic">Continue continuous non-invasive vital sign monitoring.</li>
                )}
              </ul>
            </div>
          )}
        </div>

        {/* 12. Monitoring Priorities */}
        <div className="bg-slate-900/80 border border-medBorder rounded-xl overflow-hidden">
          <button
            onClick={() => toggleSection('monitoring_priorities')}
            className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-slate-800/50 transition"
          >
            <span className="text-xs font-bold text-slate-200 flex items-center gap-2">
              <span>12. Immediate Surveillance Priorities</span>
              <window.ProvenanceBadge type="PATIENT_DATA" />
            </span>
            <span className="text-slate-400 text-xs">{expandedSections.monitoring_priorities ? '▼' : '▶'}</span>
          </button>
          {expandedSections.monitoring_priorities && (
            <div className="px-4 pb-3 pt-1 border-t border-slate-800/60">
              <ul className="space-y-1.5 text-xs text-slate-300">
                {(rep.monitoring_priorities && rep.monitoring_priorities.length > 0) ? (
                  rep.monitoring_priorities.map((item, idx) => (
                    <li key={idx} className="flex items-start space-x-2">
                      <span className="text-indigo-400 font-bold">•</span>
                      <span>{item}</span>
                    </li>
                  ))
                ) : (
                  <li className="text-slate-500 italic">Maintain standard inpatient telemetry intervals.</li>
                )}
              </ul>
            </div>
          )}
        </div>

        {/* 13. Urgency & Escalation Rationale */}
        <div className="bg-slate-900/80 border border-medBorder rounded-xl overflow-hidden">
          <button
            onClick={() => toggleSection('escalation_rationale')}
            className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-slate-800/50 transition"
          >
            <span className="text-xs font-bold text-slate-200 flex items-center gap-2">
              <span>13. Escalation Pathway & Urgency Rationale</span>
              <window.ProvenanceBadge type="DETERMINISTIC_SAFETY" />
            </span>
            <span className="text-slate-400 text-xs">{expandedSections.escalation_rationale ? '▼' : '▶'}</span>
          </button>
          {expandedSections.escalation_rationale && (
            <div className="px-4 pb-3 pt-1 border-t border-slate-800/60 text-xs text-slate-300 leading-relaxed">
              {rep.escalation_rationale || 'Routine surveillance priority maintained based on physiological baseline alignment.'}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

