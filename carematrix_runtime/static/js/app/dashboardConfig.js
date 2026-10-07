/**
 * CareMatrix Dashboard Configuration & Theme constants.
 */

window.CareMatrixConfig = {
  theme: {
    colors: {
      medDark: '#090d16',
      medCard: '#0f172a',
      medCardLight: '#1e293b',
      medBorder: '#1e293b',
      medBorderLight: '#334155',
      medPrimary: '#0ea5e9',
      medAlert: '#ef4444',
      medWarning: '#f59e0b',
      medSuccess: '#10b981',
      medPurple: '#8b5cf6',
      medIndigo: '#6366f1',
      medTeal: '#14b8a6',
      medMuted: '#64748b',
    },
    // Semantic status tokens
    statusTokens: {
      HEALTHY: { bg: 'bg-emerald-950/60', text: 'text-emerald-400', border: 'border-emerald-700/60', dot: 'bg-emerald-400' },
      STABLE: { bg: 'bg-emerald-950/60', text: 'text-emerald-400', border: 'border-emerald-700/60', dot: 'bg-emerald-400' },
      SURVEILLANCE: { bg: 'bg-amber-950/60', text: 'text-amber-400', border: 'border-amber-700/60', dot: 'bg-amber-400' },
      ALERT: { bg: 'bg-red-950/70', text: 'text-red-300', border: 'border-red-700/80', dot: 'bg-red-400' },
      DEGRADED: { bg: 'bg-amber-950/60', text: 'text-amber-400', border: 'border-amber-700/60', dot: 'bg-amber-400' },
      CRITICAL: { bg: 'bg-red-950/80', text: 'text-red-200', border: 'border-red-600', dot: 'bg-red-500' },
      RECOVERING: { bg: 'bg-teal-950/60', text: 'text-teal-300', border: 'border-teal-700/60', dot: 'bg-teal-400' },
      TOOL_CALLING: { bg: 'bg-sky-950/70', text: 'text-sky-300', border: 'border-sky-600', dot: 'bg-sky-400' },
      COMPLETED: { bg: 'bg-slate-900', text: 'text-emerald-400', border: 'border-slate-800', dot: 'bg-emerald-400' },
      FALLBACK: { bg: 'bg-amber-950/80', text: 'text-amber-300', border: 'border-amber-700', dot: 'bg-amber-400' },
      ERROR: { bg: 'bg-red-950/90', text: 'text-red-300', border: 'border-red-700', dot: 'bg-red-500' },
      IDLE: { bg: 'bg-slate-900/50', text: 'text-slate-500', border: 'border-slate-800/80', dot: 'bg-slate-600' },
    }
  },
  limits: {
    maxGlobalEvents: 150,
    maxPatientEvents: 50,
    maxVitalsHistory: 40,
  },
  endpoints: {
    health: '/api/health',
    patients: '/api/patients',
    alerts: '/api/alerts',
    datasets: '/api/datasets',
    scenariosTrigger: '/api/scenarios/trigger',
    patientDataset: '/api/patients/dataset',
    sseStream: '/api/events/stream',
  }
};
