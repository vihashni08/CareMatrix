/**
 * Central Event Reducer for CareMatrix.
 * 
 * Directly applies incoming real-time SSE events and REST hydration payloads
 * into the single normalized state model.
 * 
 * Strict Multi-Patient Isolation:
 * Events for patient X ONLY mutate patientDetails[X] and patients[X].
 * System events NEVER mutate patient records.
 */

window.careMatrixReducer = function(state, action) {
  if (!state) return window.createInitialState();

  const limits = window.CareMatrixConfig?.limits || {
    maxGlobalEvents: 150,
    maxVitalsHistory: 40,
  };

  switch (action.type) {
    // ------------------------------------------------------------------------
    // SSE Status Updates
    // ------------------------------------------------------------------------
    case 'SET_SSE_STATUS': {
      return {
        ...state,
        system: {
          ...state.system,
          sseStatus: action.payload,
        }
      };
    }

    // ------------------------------------------------------------------------
    // UI Local State Actions
    // ------------------------------------------------------------------------
    case 'SET_ACTIVE_TAB': {
      return {
        ...state,
        ui: { ...state.ui, activeTab: action.payload }
      };
    }

    case 'SET_SELECTED_AGENT': {
      return {
        ...state,
        ui: { ...state.ui, selectedAgentName: action.payload }
      };
    }

    case 'SET_SELECTED_PATIENT_ID': {
      return {
        ...state,
        selectedPatientId: Number(action.payload)
      };
    }

    case 'TOGGLE_SECTION': {
      const key = action.payload;
      return {
        ...state,
        ui: {
          ...state.ui,
          expandedSections: {
            ...state.ui.expandedSections,
            [key]: !state.ui.expandedSections[key],
          }
        }
      };
    }

    case 'SET_ALL_SECTIONS': {
      const val = Boolean(action.payload);
      const next = {};
      Object.keys(state.ui.expandedSections).forEach(k => { next[k] = val; });
      return {
        ...state,
        ui: { ...state.ui, expandedSections: next }
      };
    }

    case 'TOGGLE_ORDER_CHECK': {
      const { patientId, index } = action.payload;
      const key = `${patientId}_${index}`;
      return {
        ...state,
        ui: {
          ...state.ui,
          orderChecked: {
            ...state.ui.orderChecked,
            [key]: !state.ui.orderChecked[key],
          }
        }
      };
    }

    case 'SET_DATASET_CONTROLS': {
      return {
        ...state,
        ui: { ...state.ui, ...action.payload }
      };
    }

    case 'CLEAR_EVENT_HISTORY': {
      return {
        ...state,
        eventHistory: []
      };
    }

    // ------------------------------------------------------------------------
    // REST Hydration Actions
    // ------------------------------------------------------------------------
    case 'HYDRATE_SYSTEM_HEALTH': {
      const h = action.payload || {};
      return {
        ...state,
        system: {
          ...state.system,
          status: h.status || (h.all_healthy ? 'healthy' : 'degraded'),
          allHealthy: Boolean(h.all_healthy),
          totalAgents: h.total_agents || 5,
          riskModelName: h.risk_model_name || state.system.riskModelName,
          supervisor: h.supervisor || state.system.supervisor,
          agents: h.agents || (h.supervisor?.agents) || state.system.agents,
          failureLog: h.failure_log || (h.supervisor?.failure_log) || state.system.failureLog,
          runtimeMetrics: h.runtime_metrics || state.system.runtimeMetrics,
          activeAlertCount: h.active_alert_count !== undefined ? h.active_alert_count : state.system.activeAlertCount,
        }
      };
    }

    case 'HYDRATE_PATIENTS_LIST': {
      const list = Array.isArray(action.payload) ? action.payload : [];
      const updatedPatients = { ...state.patients };
      const updatedDetails = { ...state.patientDetails };

      list.forEach(p => {
        const pid = p.patient_id;
        updatedPatients[pid] = {
          ...(updatedPatients[pid] || {}),
          ...p,
        };

        if (!updatedDetails[pid]) {
          updatedDetails[pid] = window.createInitialPatientDetail(pid, p.name);
        }
        updatedDetails[pid].status = p.status || updatedDetails[pid].status;
        updatedDetails[pid].scenario = p.scenario || updatedDetails[pid].scenario;
        if (p.data_source) updatedDetails[pid].data_source = p.data_source;
        if (p.latest_vitals) updatedDetails[pid].vitals = p.latest_vitals;
      });

      return {
        ...state,
        patients: updatedPatients,
        patientDetails: updatedDetails,
      };
    }

    case 'HYDRATE_PATIENT_DETAIL': {
      const detail = action.payload;
      if (!detail || !detail.patient_id) return state;
      const pid = detail.patient_id;

      const currentRecord = state.patientDetails[pid] || window.createInitialPatientDetail(pid, detail.name);

      const updatedRecord = {
        ...currentRecord,
        name: detail.name || currentRecord.name,
        data_source: detail.data_source || currentRecord.data_source,
        status: detail.status || currentRecord.status,
        scenario: detail.scenario || currentRecord.scenario,
        vitals: detail.latest_vitals || currentRecord.vitals,
        monitoring: detail.latest_monitoring_event || currentRecord.monitoring,
        risk: detail.latest_risk_decision || currentRecord.risk,
        dataAnalysis: detail.latest_data_analysis || currentRecord.dataAnalysis,
        clinicalReasoning: detail.latest_clinical_reasoning || currentRecord.clinicalReasoning,
        careCoordination: detail.latest_care_action || currentRecord.careCoordination,
        alerts: detail.active_alerts || currentRecord.alerts,
        lastUpdated: Date.now() / 1000,
      };

      // Extract tool traces from metadata if present
      if (updatedRecord.monitoring?.metadata?.tool_trace) {
        updatedRecord.agenticTraces.monitoring = window.CareMatrixEventUtils.normalizeToolTrace(
          updatedRecord.monitoring.metadata.tool_trace
        );
      }
      if (updatedRecord.dataAnalysis?.metadata?.tool_trace) {
        updatedRecord.agenticTraces.dataAnalysis = window.CareMatrixEventUtils.normalizeToolTrace(
          updatedRecord.dataAnalysis.metadata.tool_trace
        );
      }
      if (updatedRecord.clinicalReasoning?.metadata?.tool_trace) {
        updatedRecord.agenticTraces.clinicalReasoning = window.CareMatrixEventUtils.normalizeToolTrace(
          updatedRecord.clinicalReasoning.metadata.tool_trace
        );
      }

      // Extract cross-agent negotiation trace from backend clinical reasoning metadata if present
      const negDiag = updatedRecord.clinicalReasoning?.metadata?.negotiation_trace?.dialogue;
      if (Array.isArray(negDiag) && negDiag.length > 0 && (!updatedRecord.negotiationTrace || updatedRecord.negotiationTrace.length === 0)) {
        updatedRecord.negotiationTrace = negDiag.map(d => ({
          round: d.round || 1,
          performative: d.performative || 'CHALLENGE',
          reason: d.statement || d.reason || '',
          requested_checks: d.requested_checks || [],
          agent: d.agent || 'RiskAgent',
          decision: d.decision || '',
          timestamp: Date.now() / 1000,
        }));
      } else if (updatedRecord.dataAnalysis?.performative === 'CHALLENGE' && (!updatedRecord.negotiationTrace || updatedRecord.negotiationTrace.length === 0)) {
        updatedRecord.negotiationTrace = [{
          round: updatedRecord.dataAnalysis.challenge_round || 1,
          performative: updatedRecord.dataAnalysis.performative,
          reason: updatedRecord.dataAnalysis.metadata?.challenge_reason || 'Observed trends conflict with risk assessment',
          requested_checks: updatedRecord.dataAnalysis.executed_checks || [],
          agent: 'DataAnalysisAgent',
          decision: updatedRecord.dataAnalysis.risk_level || '',
          timestamp: updatedRecord.dataAnalysis.timestamp || (Date.now() / 1000),
        }];
      }

      updatedRecord.pipelineState = window.CareMatrixEventUtils.derivePipelineState(updatedRecord);

      return {
        ...state,
        patientDetails: {
          ...state.patientDetails,
          [pid]: updatedRecord,
        },
        patients: {
          ...state.patients,
          [pid]: {
            ...(state.patients[pid] || {}),
            patient_id: pid,
            name: updatedRecord.name,
            status: updatedRecord.status,
            scenario: updatedRecord.scenario,
            risk_level: updatedRecord.risk?.risk_level || (state.patients[pid]?.risk_level) || 'LOW RISK',
            latest_vitals: updatedRecord.vitals,
            data_source: updatedRecord.data_source,
          }
        }
      };
    }

    case 'HYDRATE_PATIENT_VITALS': {
      const { patientId, vitals } = action.payload;
      if (!patientId || !Array.isArray(vitals)) return state;
      const pid = Number(patientId);

      const record = state.patientDetails[pid] || window.createInitialPatientDetail(pid);
      return {
        ...state,
        patientDetails: {
          ...state.patientDetails,
          [pid]: {
            ...record,
            vitalsHistory: vitals.slice(-limits.maxVitalsHistory),
            vitals: vitals.length > 0 ? vitals[vitals.length - 1] : record.vitals,
          }
        }
      };
    }

    case 'HYDRATE_PATIENT_MEMORY': {
      const { patientId, memory } = action.payload;
      if (!patientId || !memory) return state;
      const pid = Number(patientId);
      const record = state.patientDetails[pid] || window.createInitialPatientDetail(pid);

      return {
        ...state,
        patientDetails: {
          ...state.patientDetails,
          [pid]: {
            ...record,
            memory: memory,
          }
        }
      };
    }

    case 'HYDRATE_ALERTS': {
      const alerts = Array.isArray(action.payload) ? action.payload : [];
      return {
        ...state,
        activeAlerts: alerts,
        system: {
          ...state.system,
          activeAlertCount: alerts.length,
        }
      };
    }

    // ------------------------------------------------------------------------
    // Real-Time SSE Ingestion (Zero-Refetch State Mutations)
    // ------------------------------------------------------------------------
    case 'SSE_EVENT_RECEIVED': {
      const { type: eventType, data } = action.payload;
      if (!eventType || !data) return state;

      // 1. Append bounded global event audit entry
      const auditEntry = window.CareMatrixEventUtils.createAuditEntry(eventType, data);
      const updatedHistory = [auditEntry, ...state.eventHistory].slice(0, limits.maxGlobalEvents);

      // Handle system-level events (Connection established, heartbeats, supervisor failures/recoveries)
      if (eventType === 'connection_established') {
        const d = data || {};
        const newState = {
          ...state,
          eventHistory: updatedHistory,
          system: {
            ...state.system,
            sseStatus: 'CONNECTED',
            activeAlertCount: d.active_alerts ? d.active_alerts.length : state.system.activeAlertCount,
          },
          activeAlerts: d.active_alerts || state.activeAlerts,
        };
        if (d.patients) {
          return window.careMatrixReducer(newState, { type: 'HYDRATE_PATIENTS_LIST', payload: d.patients });
        }
        return newState;
      }

      if (eventType === 'heartbeat') {
        const agentName = data.agent_name;
        const currentAgents = { ...state.system.agents };
        if (agentName) {
          currentAgents[agentName] = {
            ...(currentAgents[agentName] || {}),
            status: data.status || 'healthy',
            last_heartbeat_ago_seconds: 0.0,
          };
        }
        return {
          ...state,
          eventHistory: updatedHistory,
          system: {
            ...state.system,
            agents: currentAgents,
          }
        };
      }

      if (eventType === 'agent_failure') {
        const failureList = [data, ...(state.system.failureLog || [])].slice(0, 50);
        return {
          ...state,
          eventHistory: updatedHistory,
          system: {
            ...state.system,
            status: 'degraded',
            allHealthy: false,
            failureLog: failureList,
          }
        };
      }

      if (eventType === 'agent_recovery') {
        return {
          ...state,
          eventHistory: updatedHistory,
          system: {
            ...state.system,
            status: 'healthy',
          }
        };
      }

      if (eventType === 'alert_updated') {
        // Update alert manager state
        const updatedAlerts = state.activeAlerts.map(a => (a.alert_id === data.alert_id ? { ...a, ...data } : a))
          .filter(a => a.state !== 'RESOLVED');
        const alertPid = window.CareMatrixEventUtils.extractPatientId(data);
        const nextDetails = { ...state.patientDetails };
        if (alertPid !== null && nextDetails[alertPid]) {
          nextDetails[alertPid] = {
            ...nextDetails[alertPid],
            alerts: (nextDetails[alertPid].alerts || [])
              .map(a => (a.alert_id === data.alert_id ? { ...a, ...data } : a))
              .filter(a => a.state !== 'RESOLVED'),
          };
        }
        return {
          ...state,
          eventHistory: updatedHistory,
          activeAlerts: updatedAlerts,
          patientDetails: nextDetails,
          system: {
            ...state.system,
            activeAlertCount: updatedAlerts.length,
          }
        };
      }

      // 2. Identify target patient safely
      const pid = window.CareMatrixEventUtils.extractPatientId(data);
      if (pid === null) {
        // Non-patient event processed at system level
        return {
          ...state,
          eventHistory: updatedHistory,
        };
      }

      // Patient-scoped updates with strict multi-bed isolation
      const currentDetail = state.patientDetails[pid] || window.createInitialPatientDetail(pid);
      const currentSummary = state.patients[pid] || { patient_id: pid, name: `Bed ${pid}` };

      let nextDetail = { ...currentDetail };
      let nextSummary = { ...currentSummary };

      // Handle specific patient-scoped SSE event types
      if (eventType === 'vital_tick') {
        const obs = data;
        const vitalsHistory = [...currentDetail.vitalsHistory, obs].slice(-limits.maxVitalsHistory);
        nextDetail.vitals = obs;
        nextDetail.vitalsHistory = vitalsHistory;
        nextDetail.scenario = obs.scenario || nextDetail.scenario;
        nextDetail.lastUpdated = Date.now() / 1000;

        nextSummary.latest_vitals = obs;
        nextSummary.scenario = obs.scenario || nextSummary.scenario;
        nextSummary.last_updated = Date.now() / 1000;
      }
      else if (eventType === 'monitoring_alert') {
        nextDetail.monitoring = data;
        if (data.metadata?.tool_trace) {
          nextDetail.agenticTraces.monitoring = window.CareMatrixEventUtils.normalizeToolTrace(data.metadata.tool_trace);
        }
        const eventTypeStr = String(data.event_type || '').toLowerCase();
        if (eventTypeStr.includes('recovery') || eventTypeStr.includes('recovered')) {
          nextDetail.status = 'RECOVERING';
          nextSummary.status = 'RECOVERING';
        } else if (eventTypeStr.includes('started') || eventTypeStr.includes('active')) {
          nextDetail.status = 'ALERT';
          nextSummary.status = 'ALERT';
        }
      }
      else if (eventType === 'risk_prediction') {
        nextDetail.risk = data;
        nextSummary.risk_level = data.risk_level;
        nextSummary.risk_probability = data.risk_probability;
        if (data.risk_level && data.risk_level.includes('HIGH')) {
          nextDetail.status = 'ALERT';
          nextSummary.status = 'ALERT';
        }
      }
      else if (eventType === 'agent_negotiation') {
        const negEntry = {
          round: data.challenge_round || 1,
          performative: data.performative || 'CHALLENGE',
          reason: data.reason || data.metadata?.challenge_reason || '',
          requested_checks: data.requested_checks || data.executed_checks || [],
          agent: data.source || (data.performative === 'AGREE' ? 'RiskAgent' : 'DataAnalysisAgent'),
          decision: data.decision || data.risk_level || '',
          timestamp: data.timestamp || (Date.now() / 1000),
        };
        nextDetail.negotiationTrace = [...(nextDetail.negotiationTrace || []), negEntry].slice(-20);
      }
      else if (eventType === 'data_analysis') {
        nextDetail.dataAnalysis = data;
        if (data.metadata?.tool_trace) {
          nextDetail.agenticTraces.dataAnalysis = window.CareMatrixEventUtils.normalizeToolTrace(data.metadata.tool_trace);
        }
      }
      else if (eventType === 'clinical_reasoning') {
        nextDetail.clinicalReasoning = data;
        if (data.metadata?.tool_trace) {
          nextDetail.agenticTraces.clinicalReasoning = window.CareMatrixEventUtils.normalizeToolTrace(data.metadata.tool_trace);
        }
        if (data.priority === 'URGENT') {
          nextDetail.status = 'ALERT';
          nextSummary.status = 'ALERT';
        } else if (data.priority === 'ELEVATED' && nextDetail.status !== 'ALERT') {
          nextDetail.status = 'SURVEILLANCE';
          nextSummary.status = 'SURVEILLANCE';
        }
      }
      else if (eventType === 'care_coordination') {
        nextDetail.careCoordination = data;
        nextSummary.care_action_priority = data.priority;
        nextSummary.care_action_type = data.action_type;
      }
      else if (eventType === 'clinician_feedback') {
        const ep = {
          action: data.action,
          clinician_id: data.clinician_id,
          reason: data.reason || data.resolution_notes || '',
          timestamp: Date.now() / 1000,
        };
        nextDetail.memory = {
          ...nextDetail.memory,
          count: (nextDetail.memory.count || 0) + 1,
          episodes: [ep, ...(nextDetail.memory.episodes || [])].slice(0, 30),
        };
      }

      // Re-derive patient pipeline status
      nextDetail.pipelineState = window.CareMatrixEventUtils.derivePipelineState(nextDetail);

      return {
        ...state,
        eventHistory: updatedHistory,
        patientDetails: {
          ...state.patientDetails,
          [pid]: nextDetail,
        },
        patients: {
          ...state.patients,
          [pid]: nextSummary,
        }
      };
    }

    default:
      return state;
  }
};

