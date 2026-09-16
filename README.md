# CareMatrix

## Phase 1 – Monitoring Agent

CareMatrix Phase 1 performs continuous patient-specific physiological monitoring
and deviation detection using the [VitalDB Open Dataset](https://vitaldb.net/dataset).
It works with one-second samples from four vital signs: HR, MAP, SpO2, and RR.

The Monitoring Agent detects physiological deviations. It does not perform final
patient risk classification, diagnosis, treatment recommendation, or clinical
decision-making.

## Pipeline

```text
VitalDB
  ↓
Data Loading
  ↓
Preprocessing
  ↓
Patient Baseline
  ↓
Deviation Detection
  ↓
Trend Detection
  ↓
Signal-Quality Assessment
  ↓
Multi-Vital Detection
  ↓
Persistence Check
  ↓
Alert Lifecycle
  ↓
Monitoring Alert
  ↓
Future Risk Prediction Agent (not implemented in Phase 1)
```

For each vital, the agent calculates a rolling median from the previous 60
one-second samples (minimum 10 earlier samples). A sample is a candidate alert
when at least two vitals exceed their configured relative-deviation thresholds:

| Vital | Relative deviation threshold |
| --- | ---: |
| HR | 20% |
| MAP | 20% |
| SpO2 | 5% |
| RR | 25% |

Candidate alerts must remain present for 10 consecutive one-second samples
before one structured, non-diagnostic monitoring event is emitted for that run.
The early baseline-establishment samples never generate alerts.

## Monitoring enhancements

- **Trend detection:** a lightweight robust comparison of the beginning and end
  of a recent vital-sign window labels each vital as `increasing`, `decreasing`,
  `stable`, or `insufficient_data`. It is descriptive only.
- **Signal quality:** invalid values and possible isolated abrupt spikes are
  labelled `insufficient_data` or `suspicious`; possible-artifact readings are
  not deleted and the original input is retained as `raw_df` in pipeline
  results. Signal quality is supporting metadata: a persistent multi-vital
  event can still generate an alert when a valid reading is suspicious or data
  are sparse/fill-derived. Only explicitly invalid measurements are excluded.
- **Alert lifecycle:** an alert begins only after the existing persistence rule,
  remains `alert_active` without duplicates, and generates `alert_recovered`
  when the multi-vital deviation condition ends. A new persistent event after
  recovery can start a new alert. There is no maximum alert count or per-patient
  alert cap: every independent persistent monitoring event can generate an
  alert.

All thresholds are configurable prototype parameters in
`monitoring_agent/config.py`. They are not clinically validated. VitalDB is an
intraoperative/perioperative open dataset. CareMatrix's Monitoring Agent does
not perform risk prediction, diagnosis, treatment recommendation, or clinical
decision-making; a future Risk Prediction Agent may consume its events.

## Continuous physiological event tracking

A monitoring alert represents a continuous physiological event, not just the
single values present when persistence is first confirmed. Each independent
event receives a deterministic ID such as `case_4_event_001` and maintains a
compact summary for every affected vital: initial, latest, minimum, maximum,
peak relative deviation, current trend, signal quality, and duration.

The same ID is retained from `alert_started`, through `alert_active` and
`recovering`, to `alert_recovered`. Recovery is confirmed only after five
consecutive non-candidate samples (configurable in `config.py`). The agent does
not retain a complete VitalDB time series inside event objects. Duplicate start
alerts are prevented for one continuous event, while every independent event
remains eligible to generate an alert for future Risk Prediction Agent handoff.

## Installation

From the `CareMatrix` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows PowerShell
pip install -r requirements.txt
```

## Run the Monitoring Agent

Run the live VitalDB test. It finds a suitable case automatically, processes it,
and saves the selected case only to `data/processed/patient_<case_id>.csv`.

```bash
python run_monitoring.py
```

Run the deterministic offline persistence test (no VitalDB download/API needed):

```bash
python run_monitoring.py --synthetic
```

## Generate plots

Create simple vital-versus-time plots with rolling baselines and persistent
alert timestamps:

```bash
python notebooks/03_visualize_monitoring.py
```

For the offline synthetic test case:

```bash
python notebooks/03_visualize_monitoring.py --synthetic
```

The additional small scripts can check VitalDB connectivity and inspect a case:

```bash
python notebooks/01_test_vitaldb.py
python notebooks/02_load_patient.py
python notebooks/04_test_monitoring_enhancements.py
```

## Project layout

- `monitoring_agent/preprocessing.py`: conservative numerical and missing-value handling.
- `monitoring_agent/baseline.py`: patient-specific rolling-median baselines.
- `monitoring_agent/deviation_detector.py`: relative deviations, thresholds, and persistence.
- `monitoring_agent/trend_detector.py`: descriptive robust trend labels.
- `monitoring_agent/signal_quality.py`: possible artifact and data-quality labels.
- `monitoring_agent/alert_state.py`: alert start, active, recovery, and cooldown states.
- `monitoring_agent/config.py`: centralized prototype configuration.
- `monitoring_agent/alert.py`: future-agent-compatible monitoring event dictionaries.
- `monitoring_agent/monitoring_agent.py`: VitalDB loading and pipeline orchestration.
- `run_monitoring.py`: live-case runner and offline synthetic verification.
- `notebooks/03_visualize_monitoring.py`: matplotlib visualisation script.

This is intentionally only the Monitoring Agent. A future Risk Prediction Agent
may consume its alert dictionaries, but is not included in Phase 1.
