/**
 * Top Header and Navigation Bar for CareMatrix Hospital Telemetry & Decision Support.
 *
 * Professional healthcare styling aligned with the global clinical design system:
 * Navy brand container, legible typography, clear active states, and clinical telemetry sync.
 */

window.DashboardHeader = function() {
  const state = window.useCareMatrixState();
  const dispatch = window.useCareMatrixDispatch();
  const { system, activeAlerts, ui } = state;
  const isSseConnected = system.sseStatus === 'CONNECTED';

  const navItems = [
    { id: 'clinical_dashboard', label: 'CLINICAL VIEW' },
    { id: 'command_center', label: 'COMMAND CENTER' },
    { id: 'patient_surveillance', label: 'PATIENT SURVEILLANCE' },
    { id: 'agent_inspection', label: 'AGENT INSPECTION' },
  ];

  return (
    <header className="bg-slate-900 border-b border-slate-800 px-6 py-3 flex flex-wrap items-center justify-between shadow-sm sticky top-0 z-50 gap-3">
      {/* Brand & Subtitle */}
      <div className="flex items-center space-x-3.5">
        <div className="h-9 w-9 rounded bg-blue-600 flex items-center justify-center font-bold text-white text-sm tracking-wider shadow-sm">
          CM
        </div>
        <div>
          <div className="flex items-center gap-2">
            <span className="text-base font-bold tracking-tight text-white">
              CAREMATRIX
            </span>
            <span className="text-[10px] font-mono font-semibold text-slate-300 bg-slate-800 border border-slate-700 px-1.5 py-0.5 rounded">
              CDS Engine v2.0
            </span>
          </div>
          <p className="text-[11px] text-slate-400">Continuous Inpatient Monitoring &amp; Decision Support</p>
        </div>
      </div>

      {/* Navigation Tabs */}
      <nav className="flex items-center bg-slate-950 border border-slate-800 rounded p-1" aria-label="System Views">
        {navItems.map(item => {
          const isActive = ui.activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => dispatch({ type: 'SET_ACTIVE_TAB', payload: item.id })}
              className={`px-3.5 py-1.5 rounded text-xs font-semibold tracking-wide transition-colors ${
                isActive
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
              }`}
            >
              {item.label}
            </button>
          );
        })}
      </nav>

      {/* Live System Status Badges */}
      <div className="flex items-center space-x-3 text-xs">
        {/* SSE Stream Status */}
        <div className="flex items-center space-x-2 bg-slate-950 px-2.5 py-1.5 rounded border border-slate-800">
          <span className={`inline-block w-2 h-2 rounded-full ${isSseConnected ? 'bg-teal-400' : 'bg-red-400'}`} />
          <span className="text-slate-400 font-medium">Telemetry:</span>
          <span className={`font-semibold font-mono ${isSseConnected ? 'text-teal-400' : 'text-red-400'}`}>
            {isSseConnected ? 'CONNECTED' : (system.sseStatus || 'CONNECTING')}
          </span>
        </div>

        {/* Active Alert Counter */}
        <div className={`px-2.5 py-1.5 rounded font-bold flex items-center gap-1.5 border ${
          activeAlerts.length > 0
            ? 'bg-red-950/80 text-red-300 border-red-700 pulse-alert'
            : 'bg-slate-950 text-slate-300 border-slate-800'
        }`}>
          <span className={`inline-block w-2 h-2 rounded-full ${activeAlerts.length > 0 ? 'bg-red-400' : 'bg-teal-500'}`} />
          <span>{activeAlerts.length} Active Alert{activeAlerts.length === 1 ? '' : 's'}</span>
        </div>
      </div>
    </header>
  );
};

window.BedSelectorStrip = function() {
  const state = window.useCareMatrixState();
  const dispatch = window.useCareMatrixDispatch();
  const patientList = Object.values(state.patients);
  const selectedPid = state.selectedPatientId;
  const safeNumeric = (value) => {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  };

  return (
    <section className="bg-slate-950 border-b border-slate-800 px-6 py-2 flex items-center space-x-2.5 overflow-x-auto text-xs" aria-label="Bed Selection Strip">
      <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider pr-1">Beds:</span>
      {patientList.map(p => {
        const isSelected = p.patient_id === selectedPid;
        const hasAlert = p.has_active_alert || p.status === 'ALERT';
        const isSurveillance = p.status === 'SURVEILLANCE';
        const hrValue = safeNumeric(p.latest_vitals?.hr ?? p.latest_vitals?.HR);
        const mapValue = safeNumeric(p.latest_vitals?.map ?? p.latest_vitals?.MAP);
        const hr = hrValue !== null ? Math.round(hrValue) : null;
        const map = mapValue !== null ? Math.round(mapValue) : null;

        return (
          <button
            key={p.patient_id}
            onClick={() => dispatch({ type: 'SET_SELECTED_PATIENT_ID', payload: p.patient_id })}
            className={`flex items-center space-x-2.5 px-3 py-1.5 rounded border text-left transition-colors ${
              isSelected
                ? 'bg-slate-800 border-slate-700 text-white shadow-sm ring-1 ring-blue-500'
                : 'bg-slate-900 border-slate-800 text-slate-300 hover:border-slate-700'
            }`}
          >
            <div className="flex flex-col">
              <div className="flex items-center space-x-2">
                <span className="font-bold text-slate-100">{p.name || `Bed ${p.patient_id}`}</span>
                <span className={`text-[10px] px-1.5 py-0.2 rounded font-bold uppercase ${
                  hasAlert ? 'bg-red-900/80 text-red-300 border border-red-700' :
                  isSurveillance ? 'bg-amber-900/80 text-amber-300 border border-amber-700' :
                  'bg-teal-950 text-teal-300 border border-teal-800'
                }`}>
                  {p.status || 'STABLE'}
                </span>
              </div>
              <div className="flex items-center space-x-2 text-[11px] text-slate-400 mt-0.5">
                <span>HR: <strong className="text-slate-200">{hr || '—'}</strong></span>
                <span>MAP: <strong className="text-slate-200">{map || '—'}</strong></span>
              </div>
            </div>
          </button>
        );
      })}
    </section>
  );
};
