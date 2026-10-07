/**
 * Dashboard Router Component.
 *
 * Primary views:
 *   clinical_dashboard  — Clean clinical view (DEFAULT, new)
 *   command_center      — System overview / census
 *   patient_surveillance — Deep bed dive
 *   agent_inspection    — Tool traces, negotiation, supervisor health
 */

window.DashboardRouter = function() {
  const state = window.useCareMatrixState();
  const currentTab = state.ui.activeTab || 'clinical_dashboard';

  return (
    <>
      {/* Show bed selector strip in non-clinical views */}
      {(currentTab === 'patient_surveillance' || currentTab === 'agent_inspection') && (
        <window.BedSelectorStrip />
      )}

      {currentTab === 'clinical_dashboard' && (
        <window.ClinicalDashboard />
      )}

      {currentTab === 'command_center' && (
        <window.CommandCenterView />
      )}

      {currentTab === 'patient_surveillance' && (
        <window.PatientSurveillanceView />
      )}

      {currentTab === 'agent_inspection' && (
        <window.AgentInspectionView />
      )}
    </>
  );
};
