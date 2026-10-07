/**
 * React Context and Store hook for CareMatrix Dashboard.
 */

const { createContext, useContext, useReducer, useEffect, useRef } = React;

window.CareMatrixStateContext = createContext(null);
window.CareMatrixDispatchContext = createContext(null);

window.CareMatrixProvider = function({ children }) {
  const [state, dispatch] = useReducer(
    window.careMatrixReducer,
    null,
    window.createInitialState
  );

  // Initialize SSE service once
  useEffect(() => {
    const sseService = window.CareMatrixSseService(dispatch);
    sseService.connect();

    return () => {
      sseService.disconnect();
    };
  }, []);

  // Hydrate global system data on mount
  useEffect(() => {
    const hydrateGlobal = async () => {
      try {
        const [health, patientsRes, alertsRes] = await Promise.all([
          window.CareMatrixApi.fetchHealth().catch(e => console.warn('Health init error:', e)),
          window.CareMatrixApi.fetchPatients().catch(e => console.warn('Patients init error:', e)),
          window.CareMatrixApi.fetchAlerts().catch(e => console.warn('Alerts init error:', e)),
        ]);

        if (health) dispatch({ type: 'HYDRATE_SYSTEM_HEALTH', payload: health });
        if (patientsRes?.patients) dispatch({ type: 'HYDRATE_PATIENTS_LIST', payload: patientsRes.patients });
        if (alertsRes?.active) dispatch({ type: 'HYDRATE_ALERTS', payload: alertsRes.active });
      } catch (err) {
        console.error('[CareMatrix Store] Initial hydration error:', err);
      }
    };

    hydrateGlobal();
  }, []);

  // Hydrate patient-specific baseline when selectedPatientId changes
  const prevPidRef = useRef(null);
  useEffect(() => {
    const pid = state.selectedPatientId;
    if (!pid || pid === prevPidRef.current) return;
    prevPidRef.current = pid;

    const hydrateSelectedPatient = async () => {
      try {
        const [detail, vitalsRes, memoryRes] = await Promise.all([
          window.CareMatrixApi.fetchPatientDetail(pid).catch(e => console.warn(`Detail fetch error for ${pid}:`, e)),
          window.CareMatrixApi.fetchVitalsHistory(pid, 40).catch(e => console.warn(`Vitals history error for ${pid}:`, e)),
          window.CareMatrixApi.fetchPatientMemory(pid).catch(e => console.warn(`Memory fetch error for ${pid}:`, e)),
        ]);

        if (detail) dispatch({ type: 'HYDRATE_PATIENT_DETAIL', payload: detail });
        if (vitalsRes?.vitals) dispatch({ type: 'HYDRATE_PATIENT_VITALS', payload: { patientId: pid, vitals: vitalsRes.vitals } });
        if (memoryRes) dispatch({ type: 'HYDRATE_PATIENT_MEMORY', payload: { patientId: pid, memory: memoryRes } });
      } catch (err) {
        console.error(`[CareMatrix Store] Patient ${pid} hydration error:`, err);
      }
    };

    hydrateSelectedPatient();
  }, [state.selectedPatientId]);

  return (
    <window.CareMatrixStateContext.Provider value={state}>
      <window.CareMatrixDispatchContext.Provider value={dispatch}>
        {children}
      </window.CareMatrixDispatchContext.Provider>
    </window.CareMatrixStateContext.Provider>
  );
};

window.useCareMatrixState = function() {
  const context = useContext(window.CareMatrixStateContext);
  if (!context) throw new Error('useCareMatrixState must be used within CareMatrixProvider');
  return context;
};

window.useCareMatrixDispatch = function() {
  const context = useContext(window.CareMatrixDispatchContext);
  if (!context) throw new Error('useCareMatrixDispatch must be used within CareMatrixProvider');
  return context;
};

