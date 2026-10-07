/**
 * Central Initial State Model for CareMatrix Clinician Dashboard.
 * 
 * Provides a normalized source of truth across:
 * - system (health, supervisor status, SSE connectivity, runtime metrics)
 * - patients (dictionary of normalized patient summaries keyed by patient_id)
 * - patientDetails (dictionary of normalized per-bed rich states)
 * - eventHistory (bounded chronological event list for audit trail)
 * - ui (activeTab: 'command_center' | 'patient_surveillance' | 'agent_inspection')
 */

window.createInitialState = function() {
  return {
    system: {
      status: 'healthy',
      allHealthy: true,
      totalAgents: 5,
      riskModelName: 'RandomForestClassifier',
      supervisor: {
        all_healthy: true,
        total_agents: 5,
        agents: {},
        failure_log_count: 0,
      },
      agents: {},
      failureLog: [],
      runtimeMetrics: {
        total_observations_evaluated: 0,
        routine_bypassed_cycles: 0,
        escalated_cycles: 0,
        recoveries_detected: 0,
      },
      activeAlertCount: 0,
      sseStatus: 'DISCONNECTED', // 'CONNECTED' | 'RECONNECTING' | 'DISCONNECTED'
      lastTickTime: null,
    },
    // Normalized patient summaries: { [patient_id]: { patient_id, name, status, scenario, risk_level, latest_vitals, ... } }
    patients: {},
    // Currently selected bed ID
    selectedPatientId: 101,
    // Normalized per-bed details: { [patient_id]: { patient_id, vitals, vitalsHistory, monitoring, risk, dataAnalysis, clinicalReasoning, careCoordination, alerts, agenticTraces, negotiationTrace, pipelineState, memory } }
    patientDetails: {},
    // Bounded global event history for the audit/research timeline
    eventHistory: [],
    // Active alerts list from AlertManager
    activeAlerts: [],
    // UI state
    ui: {
      // Primary navigation views:
      activeTab: 'clinical_dashboard', // 'clinical_dashboard' | 'command_center' | 'patient_surveillance' | 'agent_inspection'
      selectedAgentName: 'MonitoringAgent',
      expandedSections: {
        clinical_status: true,
        key_findings: true,
        physiological_analysis: false,
        temporal_analysis: false,
        risk_interpretation: true,
        supporting_evidence: true,
        conflicting_evidence: true,
        evidence_synthesis: true,
        medical_evidence: true,
        clinical_interpretation: true,
        uncertainties: true,
        recommended_actions: true,
        monitoring_priorities: true,
        escalation_rationale: true,
      },
      orderChecked: {},
      availableDatasets: { vitaldb: [] },
      selectedAdapter: 'vitaldb',
      selectedCaseId: '',
      loadingDataset: false,
      datasetLoadMsg: '',
      triggeringScenario: false,
    }
  };
};

/**
 * Creates a blank normalized patient record container.
 */
window.createInitialPatientDetail = function(patientId, name) {
  return {
    patient_id: patientId,
    name: name || `Patient ${patientId}`,
    data_source: 'simulated',
    status: 'STABLE',
    scenario: 'STABLE',
    vitals: {},
    vitalsHistory: [],
    monitoring: null,
    risk: null,
    dataAnalysis: null,
    clinicalReasoning: null,
    careCoordination: null,
    alerts: [],
    agenticTraces: {
      monitoring: [],
      dataAnalysis: [],
      clinicalReasoning: [],
    },
    negotiationTrace: [],
    pipelineState: {
      observation: 'IDLE',
      monitoring: 'IDLE',
      risk: 'IDLE',
      dataAnalysis: 'IDLE',
      clinicalReasoning: 'IDLE',
      careCoordination: 'IDLE',
    },
    memory: {
      count: 0,
      episodes: [],
      context_prompt: '',
    },
    lastUpdated: Date.now() / 1000,
  };
};
