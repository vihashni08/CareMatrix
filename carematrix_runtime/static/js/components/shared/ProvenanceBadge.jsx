/**
 * Shared Provenance Badge Component.
 */

window.ProvenanceBadge = function({ type }) {
  const badges = {
    'PATIENT_DATA': { label: 'PATIENT DATA', bg: 'bg-emerald-950/80', text: 'text-emerald-300', border: 'border-emerald-700/60' },
    'MODEL_OUTPUT': { label: 'MODEL PREDICTION', bg: 'bg-indigo-950/80', text: 'text-indigo-300', border: 'border-indigo-700/60' },
    'TEMPORAL_ANALYSIS': { label: 'TEMPORAL DRIFT', bg: 'bg-purple-950/80', text: 'text-purple-300', border: 'border-purple-700/60' },
    'AGENT_VERIFICATION': { label: 'CROSS-AGENT VERIFICATION', bg: 'bg-amber-950/80', text: 'text-amber-300', border: 'border-amber-700/60' },
    'PUBMED_RAG': { label: 'PUBMED EVIDENCE', bg: 'bg-teal-950/80', text: 'text-teal-300', border: 'border-teal-700/60' },
    'LLM_SYNTHESIS': { label: 'LLM SYNTHESIS', bg: 'bg-pink-950/80', text: 'text-pink-300', border: 'border-pink-700/60' },
    'DETERMINISTIC_SAFETY': { label: 'DETERMINISTIC SAFETY', bg: 'bg-red-950/80', text: 'text-red-300', border: 'border-red-700/60' },
  };
  const b = badges[type] || { label: type, bg: 'bg-slate-900', text: 'text-slate-300', border: 'border-slate-700' };
  return (
    <span className={`text-[10px] font-mono font-semibold px-2 py-0.5 rounded border ${b.bg} ${b.text} ${b.border}`}>
      {b.label}
    </span>
  );
};

