/**
 * Utility functions for event normalization, sanitization, and pipeline state derivation.
 */

window.CareMatrixEventUtils = {
  /**
   * Normalizes a sanitized tool trace step for frontend consumption.
   * Ensures raw LLM thought is never exposed or retained.
   */
  normalizeToolStep: function(rawStep) {
    if (!rawStep || typeof rawStep !== 'object') return null;
    return {
      iteration: rawStep.iteration || 1,
      action: rawStep.action || 'call_tool',
      toolName: rawStep.tool_name || rawStep.toolName || '',
      toolArgs: rawStep.tool_args || rawStep.toolArgs || {},
      status: rawStep.status || 'success',
      confidence: typeof rawStep.confidence === 'number' ? rawStep.confidence : null,
      rationale: rawStep.rationale || '',
      error: rawStep.error || null,
      timestamp: rawStep.timestamp || Date.now() / 1000,
    };
  },

  /**
   * Normalizes an entire tool trace list.
   */
  normalizeToolTrace: function(rawTrace) {
    if (!Array.isArray(rawTrace)) return [];
    return rawTrace
      .map(step => window.CareMatrixEventUtils.normalizeToolStep(step))
      .filter(Boolean);
  },

  /**
   * Derives pipeline stage execution state based on received agent events and tool traces.
   * Possible states: IDLE | ACTIVE | TOOL_CALLING | WAITING | COMPLETED | FALLBACK | ERROR
   */
  derivePipelineState: function(patientRecord) {
    if (!patientRecord) {
      return {
        observation: 'IDLE',
        monitoring: 'IDLE',
        risk: 'IDLE',
        dataAnalysis: 'IDLE',
        clinicalReasoning: 'IDLE',
        careCoordination: 'IDLE',
      };
    }

    const res = {
      observation: patientRecord.vitals && Object.keys(patientRecord.vitals).length > 0 ? 'ACTIVE' : 'IDLE',
      monitoring: 'IDLE',
      risk: 'IDLE',
      dataAnalysis: 'IDLE',
      clinicalReasoning: 'IDLE',
      careCoordination: 'IDLE',
    };

    // Monitoring state
    if (patientRecord.monitoring) {
      const isFallback = !!(patientRecord.monitoring.metadata?.fallback_error || patientRecord.monitoring.metadata?.deterministic_fallback);
      res.monitoring = isFallback ? 'FALLBACK' : 'COMPLETED';
    } else if (patientRecord.agenticTraces?.monitoring?.length > 0) {
      res.monitoring = 'TOOL_CALLING';
    }

    // Risk state
    if (patientRecord.risk) {
      res.risk = patientRecord.risk.status === 'failed' ? 'ERROR' : 'COMPLETED';
    }

    // Data Analysis state
    if (patientRecord.dataAnalysis) {
      const isFallback = !!(patientRecord.dataAnalysis.metadata?.fallback_error);
      res.dataAnalysis = isFallback ? 'FALLBACK' : 'COMPLETED';
    } else if (patientRecord.agenticTraces?.dataAnalysis?.length > 0) {
      res.dataAnalysis = 'TOOL_CALLING';
    }

    // Clinical Reasoning state
    if (patientRecord.clinicalReasoning) {
      const isFallback = !!(patientRecord.clinicalReasoning.metadata?.fallback_error);
      res.clinicalReasoning = isFallback ? 'FALLBACK' : 'COMPLETED';
    } else if (patientRecord.agenticTraces?.clinicalReasoning?.length > 0) {
      res.clinicalReasoning = 'TOOL_CALLING';
    }

    // Care Coordination state
    if (patientRecord.careCoordination) {
      res.careCoordination = 'COMPLETED';
    }

    return res;
  },

  /**
   * Extracts patient_id / case_id safely from an event payload.
   */
  extractPatientId: function(data) {
    if (!data || typeof data !== 'object') return null;
    const id = data.patient_id !== undefined ? data.patient_id : data.case_id;
    if (id === undefined || id === null) return null;
    const num = parseInt(id, 10);
    return isNaN(num) ? id : num;
  },

  /**
   * Creates a bounded, structured audit entry for the Agentic Event History.
   */
  createAuditEntry: function(eventType, data) {
    const pid = window.CareMatrixEventUtils.extractPatientId(data);
    const timeStr = new Date().toLocaleTimeString();

    let actionOrTool = '';
    let summaryText = '';
    let status = 'COMPLETED';

    if (eventType === 'vital_tick') {
      actionOrTool = 'TELEMETRY_SAMPLE';
      const hr = Math.round(data.hr || data.HR || 0);
      const map = Math.round(data.map || data.MAP || 0);
      summaryText = `HR: ${hr} bpm | MAP: ${map} mmHg`;
    } else if (eventType === 'monitoring_alert') {
      actionOrTool = data.event_type || 'ALERT';
      const vitals = Array.isArray(data.affected_vitals) ? data.affected_vitals.join(', ') : 'vitals';
      summaryText = `${data.severity?.toUpperCase()} alert on ${vitals} (${data.recommended_action || ''})`;
      if (data.metadata?.fallback_error) status = 'FALLBACK';
    } else if (eventType === 'risk_prediction') {
      actionOrTool = data.model_name || 'RISK_MODEL';
      summaryText = `${data.risk_level} (${Math.round((data.risk_probability || 0) * 100)}%)`;
    } else if (eventType === 'agent_negotiation') {
      actionOrTool = data.performative || 'CHALLENGE';
      summaryText = `Round ${data.challenge_round || 1}: ${data.reason || 'Verification requested'}`;
    } else if (eventType === 'data_analysis') {
      actionOrTool = data.pattern_identified?.[0] || 'SLOPE_ANALYSIS';
      summaryText = `Data quality: ${data.data_quality_flag ? 'FLAGGED' : 'NORMAL'}`;
      if (data.metadata?.fallback_error) status = 'FALLBACK';
    } else if (eventType === 'clinical_reasoning') {
      actionOrTool = data.metadata?.reasoning_mode || 'REASONING';
      summaryText = `${data.priority} priority: ${data.clinical_summary || 'Clinical evaluation'}`;
      if (data.metadata?.fallback_error) status = 'FALLBACK';
    } else if (eventType === 'care_coordination') {
      actionOrTool = data.action_type || 'CARE_ACTION';
      summaryText = `${data.priority} pathway: ${data.escalation_pathway || 'Standard'}`;
    } else if (eventType === 'heartbeat') {
      actionOrTool = data.agent_name || 'HEARTBEAT';
      summaryText = `Status: ${data.status || 'healthy'}`;
    } else if (eventType === 'agent_failure') {
      actionOrTool = data.agent_name || 'AGENT_FAILURE';
      summaryText = `Error: ${data.error_message || 'Agent fault'}`;
      status = 'ERROR';
    } else if (eventType === 'agent_recovery') {
      actionOrTool = data.agent_name || 'AGENT_RECOVERY';
      summaryText = `Recovered: ${data.details || 'Agent restarted'}`;
    } else if (eventType === 'clinician_feedback') {
      actionOrTool = data.action || 'FEEDBACK';
      summaryText = `${data.clinician_id || 'Clinician'}: ${data.reason || data.action}`;
    } else {
      actionOrTool = eventType;
      summaryText = typeof data === 'object' ? JSON.stringify(data).slice(0, 80) : String(data);
    }

    return {
      timestamp: timeStr,
      epoch: Date.now() / 1000,
      patient_id: pid,
      type: eventType,
      agent: data.source || data.agent_name || eventType,
      action: actionOrTool,
      status: status,
      summary: summaryText,
      data: data,
    };
  }
};

