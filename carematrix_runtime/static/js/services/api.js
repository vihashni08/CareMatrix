/**
 * Dedicated REST API service for CareMatrix.
 * Centralizes all HTTP endpoints used for initial hydration, patient switching,
 * manual actions, and dataset inspection.
 */

window.CareMatrixApi = {
  fetchHealth: async function() {
    const res = await fetch('/api/health');
    if (!res.ok) throw new Error(`Health fetch failed with status ${res.status}`);
    return await res.json();
  },

  fetchPatients: async function() {
    const res = await fetch('/api/patients');
    if (!res.ok) throw new Error(`Patients fetch failed with status ${res.status}`);
    return await res.json();
  },

  fetchPatientDetail: async function(patientId) {
    const res = await fetch(`/api/patients/${patientId}`);
    if (!res.ok) throw new Error(`Patient ${patientId} detail failed with status ${res.status}`);
    return await res.json();
  },

  fetchClinicalReport: async function(patientId) {
    const res = await fetch(`/api/patients/${patientId}/clinical-report`);
    if (!res.ok) throw new Error(`Clinical report fetch failed with status ${res.status}`);
    return await res.json();
  },

  fetchVitalsHistory: async function(patientId, limit = 40) {
    const res = await fetch(`/api/patients/${patientId}/vitals?limit=${limit}`);
    if (!res.ok) throw new Error(`Vitals history fetch failed with status ${res.status}`);
    return await res.json();
  },

  fetchAlerts: async function(patientId = null) {
    const url = patientId ? `/api/alerts?patient_id=${patientId}` : '/api/alerts';
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Alerts fetch failed with status ${res.status}`);
    return await res.json();
  },

  fetchPatientMemory: async function(patientId) {
    const res = await fetch(`/api/patients/${patientId}/memory`);
    if (!res.ok) throw new Error(`Memory fetch failed with status ${res.status}`);
    return await res.json();
  },

  fetchDatasets: async function() {
    const res = await fetch('/api/datasets');
    if (!res.ok) throw new Error(`Datasets fetch failed with status ${res.status}`);
    return await res.json();
  },

  acknowledgeAlert: async function(alertId, clinicianId = 'Attending Physician') {
    const res = await fetch(`/api/alerts/${alertId}/acknowledge`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ clinician_id: clinicianId }),
    });
    if (!res.ok) throw new Error(`Acknowledge alert failed with status ${res.status}`);
    return await res.json();
  },

  resolveAlert: async function(alertId, reason = 'Bedside evaluation complete; patient stable under monitoring.') {
    const res = await fetch(`/api/alerts/${alertId}/resolve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason: reason }),
    });
    if (!res.ok) throw new Error(`Resolve alert failed with status ${res.status}`);
    return await res.json();
  },

  triggerScenario: async function(patientId, scenario) {
    const res = await fetch('/api/scenarios/trigger', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ patient_id: patientId, scenario: scenario }),
    });
    if (!res.ok) throw new Error(`Trigger scenario failed with status ${res.status}`);
    return await res.json();
  },

  loadDatasetPatient: async function(caseId, adapter = 'vitaldb') {
    const res = await fetch('/api/patients/dataset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        case_id: parseInt(caseId, 10) || caseId,
        adapter: adapter,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `Load dataset failed with status ${res.status}`);
    }
    return await res.json();
  }
};

