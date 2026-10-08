/**
 * Patient Vitals and Chart.js Waveform Section.
 */

const { useEffect, useRef } = React;

window.PatientVitalsSection = function({ vitals, vitalsHistory, dataAnalysis }) {
  const chartRef = useRef(null);
  const chartInstance = useRef(null);

  useEffect(() => {
    if (!chartRef.current) return;
    const ctx = chartRef.current.getContext('2d');

    const history = Array.isArray(vitalsHistory) ? vitalsHistory : [];
    const labels = history.map((_, i) => `${i + 1}s`);
    const hrData = history.map(v => v.hr || v.HR || 0);
    const mapData = history.map(v => v.map || v.MAP || 0);
    const spo2Data = history.map(v => v.spo2 || v.SpO2 || 0);

    if (!chartInstance.current) {
      chartInstance.current = new Chart(ctx, {
        type: 'line',
        data: {
          labels,
          datasets: [
            {
              label: 'Heart Rate (BPM)',
              data: hrData,
              borderColor: '#ef4444',
              backgroundColor: 'rgba(239, 68, 68, 0.08)',
              borderWidth: 2,
              tension: 0.35,
              pointRadius: 2,
              yAxisID: 'yHR',
            },
            {
              label: 'Mean Arterial Pressure (mmHg)',
              data: mapData,
              borderColor: '#0ea5e9',
              backgroundColor: 'rgba(14, 165, 233, 0.08)',
              borderWidth: 2,
              tension: 0.35,
              pointRadius: 2,
              yAxisID: 'yMAP',
            },
            {
              label: 'Oxygen Saturation SpO2 (%)',
              data: spo2Data,
              borderColor: '#10b981',
              backgroundColor: 'rgba(16, 185, 129, 0.08)',
              borderWidth: 2,
              tension: 0.35,
              pointRadius: 2,
              yAxisID: 'ySpO2',
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          scales: {
            x: {
              grid: { color: '#1e293b' },
              ticks: { color: '#64748b', font: { size: 10 } }
            },
            yHR: {
              type: 'linear',
              position: 'left',
              min: 40,
              max: 160,
              grid: { color: '#1e293b' },
              ticks: { color: '#ef4444', font: { size: 10 } },
              title: { display: true, text: 'HR (BPM)', color: '#ef4444', font: { size: 10, weight: 'bold' } }
            },
            yMAP: {
              type: 'linear',
              position: 'right',
              min: 40,
              max: 130,
              grid: { drawOnChartArea: false },
              ticks: { color: '#0ea5e9', font: { size: 10 } },
              title: { display: true, text: 'MAP (mmHg)', color: '#0ea5e9', font: { size: 10, weight: 'bold' } }
            },
            ySpO2: {
              type: 'linear',
              position: 'right',
              min: 80,
              max: 102,
              grid: { drawOnChartArea: false },
              ticks: { color: '#10b981', font: { size: 10 } },
              display: false,
            }
          },
          plugins: {
            legend: {
              labels: { color: '#cbd5e1', font: { size: 11 } }
            }
          }
        }
      });
    } else {
      chartInstance.current.data.labels = labels;
      chartInstance.current.data.datasets[0].data = hrData;
      chartInstance.current.data.datasets[1].data = mapData;
      chartInstance.current.data.datasets[2].data = spo2Data;
      chartInstance.current.update('none');
    }
  }, [vitalsHistory]);

  const safeNumeric = (value) => {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  };
  const hr = safeNumeric(vitals?.hr ?? vitals?.HR);
  const map = safeNumeric(vitals?.map ?? vitals?.MAP);
  const spo2 = safeNumeric(vitals?.spo2 ?? vitals?.SpO2);
  const rr = safeNumeric(vitals?.rr ?? vitals?.RR);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
      {/* 4 Vital Sign Indicator Cards */}
      <div className="lg:col-span-5 grid grid-cols-2 gap-3">
        {/* Heart Rate */}
        <div className={`p-4 rounded-xl border transition-all ${
          hr > 105 || (hr < 55 && hr > 0)
            ? 'bg-red-950/40 border-red-700/80 text-red-200 shadow-lg shadow-red-950/40'
            : 'bg-medCard border-medBorder text-slate-100'
        }`}>
          <div className="flex justify-between items-center text-xs font-semibold text-slate-400">
            <span>HEART RATE</span>
            <span className="text-[10px] font-mono text-slate-500">60-100 BPM</span>
          </div>
          <div className="text-3xl font-black mt-1 font-mono tracking-tight flex items-baseline justify-between">
            <span>{hr !== null ? Math.round(hr) : '—'}</span>
            <span className="text-xs font-normal text-slate-400">bpm</span>
          </div>
          <div className="text-[11px] text-slate-400 mt-2 flex items-center justify-between border-t border-slate-800/80 pt-1.5">
            <span>Telemetry Lead II</span>
            <span className="font-mono text-sky-400">
              {dataAnalysis?.trend_metrics?.HR?.trend || 'stable'}
            </span>
          </div>
        </div>

        {/* Mean Arterial Pressure */}
        <div className={`p-4 rounded-xl border transition-all ${
          map < 65 && map > 0
            ? 'bg-red-950/40 border-red-700/80 text-red-200 shadow-lg shadow-red-950/40'
            : 'bg-medCard border-medBorder text-slate-100'
        }`}>
          <div className="flex justify-between items-center text-xs font-semibold text-slate-400">
            <span>MEAN ART PRESSURE</span>
            <span className="text-[10px] font-mono text-slate-500">70-105 mmHg</span>
          </div>
          <div className="text-3xl font-black mt-1 font-mono tracking-tight flex items-baseline justify-between">
            <span>{map !== null ? Math.round(map) : '—'}</span>
            <span className="text-xs font-normal text-slate-400">mmHg</span>
          </div>
          <div className="text-[11px] text-slate-400 mt-2 flex items-center justify-between border-t border-slate-800/80 pt-1.5">
            <span>Arterial Line</span>
            <span className="font-mono text-sky-400">
              {dataAnalysis?.trend_metrics?.MAP?.trend || 'stable'}
            </span>
          </div>
        </div>

        {/* SpO2 */}
        <div className={`p-4 rounded-xl border transition-all ${
          spo2 < 92 && spo2 > 0
            ? 'bg-red-950/40 border-red-700/80 text-red-200 shadow-lg shadow-red-950/40'
            : 'bg-medCard border-medBorder text-slate-100'
        }`}>
          <div className="flex justify-between items-center text-xs font-semibold text-slate-400">
            <span>OXYGEN SATURATION</span>
            <span className="text-[10px] font-mono text-slate-500">&gt;=95%</span>
          </div>
          <div className="text-3xl font-black mt-1 font-mono tracking-tight flex items-baseline justify-between">
            <span>{spo2 !== null ? `${Math.round(spo2)}%` : '—'}</span>
            <span className="text-xs font-normal text-slate-400">SpO2</span>
          </div>
          <div className="text-[11px] text-slate-400 mt-2 flex items-center justify-between border-t border-slate-800/80 pt-1.5">
            <span>Pulse Oximetry</span>
            <span className="font-mono text-sky-400">
              {dataAnalysis?.trend_metrics?.SpO2?.trend || 'stable'}
            </span>
          </div>
        </div>

        {/* Resp Rate */}
        <div className={`p-4 rounded-xl border transition-all ${
          rr > 22
            ? 'bg-amber-950/40 border-amber-700/80 text-amber-200 shadow-lg shadow-amber-950/40'
            : 'bg-medCard border-medBorder text-slate-100'
        }`}>
          <div className="flex justify-between items-center text-xs font-semibold text-slate-400">
            <span>RESPIRATORY RATE</span>
            <span className="text-[10px] font-mono text-slate-500">12-20 /min</span>
          </div>
          <div className="text-3xl font-black mt-1 font-mono tracking-tight flex items-baseline justify-between">
            <span>{rr !== null ? Math.round(rr) : '—'}</span>
            <span className="text-xs font-normal text-slate-400">/min</span>
          </div>
          <div className="text-[11px] text-slate-400 mt-2 flex items-center justify-between border-t border-slate-800/80 pt-1.5">
            <span>Impedance Wave</span>
            <span className="font-mono text-sky-400">
              {dataAnalysis?.trend_metrics?.RR?.trend || 'stable'}
            </span>
          </div>
        </div>
      </div>

      {/* Multi-channel Waveform */}
      <div className="lg:col-span-7 bg-medCard border border-medBorder rounded-xl p-4 flex flex-col justify-between shadow-lg">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-2">
            <span className="h-2 w-2 rounded-full bg-sky-400 animate-pulse"></span>
            <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider">
              Continuous Physiological Trajectory (Rolling 40-Second Window)
            </h3>
          </div>
          <span className="text-[10px] font-mono text-slate-400">Telemetry Sampling: 1.0s</span>
        </div>
        <div className="flex-1 min-h-[200px] w-full">
          <canvas ref={chartRef}></canvas>
        </div>
      </div>
    </div>
  );
};

