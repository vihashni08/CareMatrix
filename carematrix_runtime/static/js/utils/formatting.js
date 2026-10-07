/**
 * Formatting and display helper utilities for CareMatrix.
 */

window.CareMatrixFormatting = {
  formatVital: function(val, fallback = '--') {
    if (val === null || val === undefined || isNaN(val)) return fallback;
    return Math.round(Number(val));
  },

  formatPercent: function(val) {
    if (val === null || val === undefined || isNaN(val)) return '0%';
    return `${Math.round(Number(val) * 100)}%`;
  },

  formatTime: function(epoch) {
    if (!epoch) return new Date().toLocaleTimeString();
    return new Date(epoch * 1000).toLocaleTimeString();
  },

  formatModelName: function(name) {
    if (!name) return 'Model loading';
    return name.replace(/([a-z])([A-Z])/g, '$1 $2');
  },

  getRiskBadgeClasses: function(riskLevel) {
    const isHigh = (riskLevel || '').includes('HIGH');
    return isHigh
      ? 'bg-red-950/80 text-red-200 border-red-700 animate-pulse'
      : 'bg-emerald-950/60 text-emerald-300 border-emerald-700';
  },

  getPriorityBadgeClasses: function(priority, status) {
    if (priority === 'URGENT' || status === 'ALERT') {
      return 'bg-red-600 text-white border-red-500 shadow-md shadow-red-600/30';
    }
    if (priority === 'ELEVATED' || status === 'SURVEILLANCE') {
      return 'bg-amber-500 text-black border-amber-400 font-extrabold shadow-md shadow-amber-500/20';
    }
    return 'bg-slate-800 text-slate-300 border-slate-700';
  }
};

