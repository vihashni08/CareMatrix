/**
 * Care Action Plan and Clinical Alert Center Components.
 */

window.CareCoordinationSection = function({
  careAction,
  activeAlerts,
  selectedPid,
  orderChecked,
  onToggleOrder,
  onAcknowledgeAlert,
  onResolveAlert,
  isUrgent,
  isElevated
}) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
      {/* Left: Care Coordination Orders & Action Plan */}
      <div className="lg:col-span-7 bg-medCard border border-medBorder rounded-xl p-5 space-y-4 shadow-xl">
        <div className="flex items-center justify-between border-b border-medBorder pb-3">
          <div>
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              Care Coordination Action Plan
              <span className={`text-[10px] px-2.5 py-0.5 rounded-full font-bold uppercase ${
                isUrgent ? 'bg-red-500 text-white' :
                isElevated ? 'bg-amber-500 text-black' :
                'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
              }`}>
                Priority: {careAction?.priority || 'ROUTINE'}
              </span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Deterministic bedside tasking & decision support workflow translation (Clinical Decision Support Only)
            </p>
          </div>

          {careAction?.clinician_review_required && (
            <span className="text-xs bg-red-950/80 border border-red-700 text-red-300 px-2.5 py-1 rounded-md font-semibold flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-red-400 animate-ping"></span>
              Clinician Review Required
            </span>
          )}
        </div>

        {/* Escalation Pathway Banner */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-lg p-3">
          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
            Assigned Hospital Escalation Pathway:
          </div>
          <div className="text-xs font-bold text-sky-400 mt-0.5">
            {careAction?.escalation_pathway || 'Standard Continuous Floor Surveillance Protocol'}
          </div>
          <div className="text-xs text-slate-300 mt-1 italic">
            Clinical Reason: {careAction?.reason || 'Vitals and analytical trend indicators within normal limits.'}
          </div>
        </div>

        {/* Suggested Actionable Orders Checklist */}
        <div>
          <div className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-2 flex items-center justify-between">
            <span>Recommended Clinician Review & Decision Support Checklist:</span>
            <span className="text-[10px] text-slate-500 font-mono">Requires Bedside Clinician Verification</span>
          </div>
          <div className="space-y-1.5">
            {(careAction?.suggested_orders && careAction.suggested_orders.length > 0) ? (
              careAction.suggested_orders.map((ord, idx) => {
                const isChecked = Boolean(orderChecked[`${selectedPid}_${idx}`]);
                return (
                  <label
                    key={idx}
                    className={`flex items-start space-x-2.5 p-2.5 rounded-lg border text-xs cursor-pointer transition ${
                      isChecked
                        ? 'bg-emerald-950/20 border-emerald-800/40 text-slate-400 line-through'
                        : 'bg-slate-900/80 border-medBorder hover:border-slate-700 text-slate-200'
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={isChecked}
                      onChange={() => onToggleOrder(selectedPid, idx)}
                      className="mt-0.5 rounded border-slate-700 bg-slate-800 text-sky-500 focus:ring-sky-500"
                    />
                    <span className="flex-1">{ord}</span>
                  </label>
                );
              })
            ) : (
              <div className="text-xs text-slate-400 p-3 bg-slate-900 rounded-lg border border-slate-800">
                Maintain continuous standard observation. No active clinical order escalation required.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Right: Clinical Alert Center & Lifecycle Timeline */}
      <div className="lg:col-span-5 bg-medCard border border-medBorder rounded-xl p-5 space-y-4 flex flex-col justify-between shadow-xl">
        <div>
          <div className="flex items-center justify-between border-b border-medBorder pb-3">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              Clinical Alert Center
              <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-slate-800 text-slate-300">
                Bed {selectedPid}
              </span>
            </h3>
            <span className="text-[10px] text-slate-400 font-mono">NEW → ACTIVE → ACK → RESOLVED</span>
          </div>

          {/* Active Alerts for this patient */}
          <div className="mt-3 space-y-2">
            {activeAlerts && activeAlerts.length > 0 ? (
              activeAlerts.map(al => (
                <div
                  key={al.alert_id}
                  className={`p-3.5 rounded-lg border flex flex-col space-y-2 ${
                    al.state === 'NEW' ? 'bg-red-950/50 border-red-700 text-red-100' :
                    al.state === 'ACTIVE' ? 'bg-amber-950/40 border-amber-700 text-amber-100' :
                    'bg-slate-900 border-slate-700 text-slate-300'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded uppercase ${
                      al.state === 'NEW' ? 'bg-red-600 text-white animate-pulse' :
                      al.state === 'ACTIVE' ? 'bg-amber-600 text-black' :
                      'bg-sky-600 text-white'
                    }`}>
                      [{al.state}] {al.severity}
                    </span>
                    <span className="text-[11px] text-slate-400 font-mono">
                      Active Duration: {al.duration_seconds}s
                    </span>
                  </div>

                  <div className="text-xs">
                    <div className="font-semibold">{al.message}</div>
                    <div className="text-[11px] text-slate-400 mt-1">
                      Action Required: <b className="text-slate-200">{al.recommended_action}</b>
                    </div>
                  </div>

                  {/* Clinician Action Buttons */}
                  <div className="flex items-center justify-end space-x-2 pt-2 border-t border-slate-800/80">
                    {al.state === 'NEW' && (
                      <button
                        onClick={() => onAcknowledgeAlert(al.alert_id)}
                        className="px-3 py-1 bg-sky-600 hover:bg-sky-500 text-white rounded text-xs font-semibold shadow transition"
                      >
                        Acknowledge Alert
                      </button>
                    )}
                    <button
                      onClick={() => onResolveAlert(al.alert_id)}
                      className="px-3 py-1 bg-emerald-700 hover:bg-emerald-600 text-white rounded text-xs font-semibold shadow transition"
                    >
                      Resolve Alert
                    </button>
                  </div>
                </div>
              ))
            ) : (
              <div className="p-4 bg-slate-900/60 rounded-lg border border-slate-800 text-xs text-slate-400 text-center">
                <span className="text-emerald-400 font-bold block mb-1">✓ No Active Inpatient Alerts</span>
                All monitored physiological parameters are operating within configured baseline envelopes.
              </div>
            )}
          </div>
        </div>

        <div className="text-[11px] text-slate-500 border-t border-slate-800 pt-3 flex justify-between items-center">
          <span>Alert Lifecycle Protocol: Tier 2 Guardrails</span>
          <span className="font-mono text-slate-400">Escalation Pacing: 5.0s</span>
        </div>
      </div>
    </div>
  );
};

