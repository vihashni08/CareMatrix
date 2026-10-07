/**
 * Root App Component for CareMatrix Clinician Dashboard.
 */

window.App = function() {
  return (
    <window.CareMatrixProvider>
      <window._AppInner />
    </window.CareMatrixProvider>
  );
};

window._AppInner = function() {
  const state = window.useCareMatrixState();
  const activeTab = state.ui.activeTab || 'clinical_dashboard';
  const isClinical = activeTab === 'clinical_dashboard';

  return (
    <div className={`flex flex-col min-h-screen ${isClinical ? 'bg-gray-50' : 'bg-medDark text-slate-100'}`}>
      <window.DashboardHeader />
      <window.DashboardRouter />
      <footer className="bg-medCard border-t border-medBorder px-6 py-3 text-xs text-slate-400 flex flex-col sm:flex-row items-center justify-between gap-2 mt-auto">
        <div>
          <span className="font-semibold text-slate-300">CareMatrix</span> — Intelligent ICU Monitoring &amp; Decision-Support System
        </div>
        <div className="flex items-center space-x-4">
          <span>Supervisor: <b className="text-emerald-400">Active</b></span>
          <span>•</span>
          <span>Decision Support Only</span>
          <span>•</span>
          <span className="text-slate-500">v2.0 Modular Architecture</span>
        </div>
      </footer>
    </div>
  );
};
