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
Multi-Vital Detection
  ↓
Persistence Check
  ↓
Monitoring Alert
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
```

## Project layout

- `monitoring_agent/preprocessing.py`: conservative numerical and missing-value handling.
- `monitoring_agent/baseline.py`: patient-specific rolling-median baselines.
- `monitoring_agent/deviation_detector.py`: relative deviations, thresholds, and persistence.
- `monitoring_agent/alert.py`: future-agent-compatible monitoring event dictionaries.
- `monitoring_agent/monitoring_agent.py`: VitalDB loading and pipeline orchestration.
- `run_monitoring.py`: live-case runner and offline synthetic verification.
- `notebooks/03_visualize_monitoring.py`: matplotlib visualisation script.

This is intentionally only the Monitoring Agent. A future Risk Prediction Agent
may consume its alert dictionaries, but is not included in Phase 1.
