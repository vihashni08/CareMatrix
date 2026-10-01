# CareMatrix — Intelligent Healthcare Decision Support Using Multi-Agent Systems

CareMatrix is an autonomous, event-driven multi-agent clinical decision support architecture for continuous inpatient physiological monitoring, risk prediction, analytical validation, clinical evidence synthesis, and care workflow coordination.

---

## 1. Multi-Agent Architecture

The CareMatrix pipeline consists of 5 autonomous agents operating under an independent supervisor:

```text
                  ┌───────────────────────────────┐
                  │    AgentSupervisor Monitor    │ (Thread liveness, heartbeats,
                  │ (Heartbeats & Auto-Recovery)  │  state snapshots, auto-restart)
                  └───────────────┬───────────────┘
                                  │
Continuous Telemetry Stream       │
(Multi-Bed ICU / Floor Simulator) │
              │                   │
              ▼                   │
    ┌───────────────────┐         │
    │  Monitoring Agent │◄────────┤ (Continuous baseline, multi-vital deviation,
    └─────────┬─────────┘         │  adaptive execution gatekeeper)
              │ (Escalation / Recovery)
              ▼                   │
    ┌───────────────────┐         │
    │    Risk Agent     │◄────────┤ (RandomForestClassifier, threshold 0.16,
    └─────────┬─────────┘         │  retries, adaptive check requests)
              │ (Risk Decision Event)
              ▼                   │
    ┌───────────────────┐         │
    │Data Analysis Agent│◄────────┤ (Variance, trend slopes, signal noise check,
    └─────────┬─────────┘         │  cross-agent consistency verification)
              │ (Analytical Evidence Event)
              ▼                   │
    ┌───────────────────┐         │
    │Clinical Reasoning │◄────────┤ (Medical RAG guideline retrieval +
    │      Agent        │         │  Gemini 2.5 Flash / Deterministic reasoning)
    └─────────┬─────────┘         │
              │ (Clinical Reasoning Event)
              ▼                   │
    ┌───────────────────┐         │
    │ Care Coordination │◄────────┘ (Deterministic clinical action policies,
    │      Agent        │            triage priority, orders checklist, deduplication)
    └─────────┬─────────┘
              │ (Care Coordination Event)
              ▼
   ┌───────────────────────────────────────────────┐
   │ Centralized State & Alert Lifecycle Manager   │
   │ (NEW → ACTIVE → ACKNOWLEDGED → RESOLVED)      │
   └──────────────────────┬────────────────────────┘
                          │
                          ▼
            REST API & Real-Time SSE Stream
                          │
                          ▼
             Modern Clinician Dashboard
```

### The 5 Autonomous Agents

1. **Monitoring Agent**:
   - Performs patient-specific physiological monitoring using rolling baseline buffers (HR, MAP, SpO2, RR, BT).
   - Enforces persistence durations before triggering escalations.
   - **Adaptive Execution Gatekeeper**: Completely bypasses downstream ML/LLM execution when vitals are stable, conserving computational resources.

2. **Risk Agent**:
   - Evaluates multi-parameter physiological windows using a trained machine learning model (`RandomForestClassifier`, decision threshold: `0.16`).
   - Requests analytical verification when predictions border decision boundaries or exhibit high variability.

3. **Data Analysis Agent**:
   - Computes statistical metrics (variance, rate of change, correlation) without diagnostic interpretation.
   - Detects compromised data quality (sensor disconnect, excessive motion artifact) and marks cross-agent consistency flags (`CONSISTENT`, `CONFLICTING`, `UNCERTAIN`).

4. **Clinical Reasoning Agent**:
   - Synthesizes physiological trends, ML predictions, and statistical evidence.
   - Incorporates **Medical RAG** (Retrieval-Augmented Generation) against verified clinical guidelines.
   - Invokes **Google Gemini 2.5 Flash** (with robust deterministic fallback if offline) to produce structured clinical summaries, findings, and priority rankings (`URGENT`, `ELEVATED`, `ROUTINE`).

5. **Care Coordination Agent** *(Non-diagnostic workflow orchestration)*:
   - Evaluates validated clinical reasoning evidence against transparent, deterministic clinical action policies.
   - Generates actionable clinical workflow tasks:
     - `TRIGGER_URGENT_CLINICAL_ALERT`: Immediate bedside clinician notification & rapid response activation.
     - `SCHEDULE_CLINICIAN_REVIEW`: Routine or elevated physician evaluation within 30 minutes.
     - `REQUEST_DATA_VERIFICATION`: Bedside sensor inspection and manual vital signs check (prevents false-alarm panic during sensor disconnects).
     - `CONTINUE_ROUTINE_MONITORING`: Standard inpatient surveillance.
   - Compiles structured **Actionable Order Sets** (IV access, labs, 12-lead ECG, telemetry frequency).
   - Enforces **Task & Alert Deduplication** to eliminate alarm fatigue.

6. **Agent Supervisor**:
   - Independent background monitor thread tracking thread liveness and periodic heartbeats.
   - Automatically detects stale or frozen workers and orchestrates graceful restarts with state restoration.

---

## 2. Continuous Telemetry Stream & Realistic Scenarios

The `carematrix_runtime` simulates multi-bed inpatient telemetry across 7 physiological scenarios:

| Scenario | Physiological Dynamics | Clinical Response |
| :--- | :--- | :--- |
| `STABLE` | Normal vitals with micro-variations | Gatekeeper bypasses downstream ML/LLM |
| `GRADUAL_DETERIORATION` | Tachycardia + hypotension + tachypnea | High risk → URGENT alert & Rapid Response order set |
| `SUDDEN_ABNORMALITY` | Acute collapse (extreme desat / drop in MAP) | Immediate bedside clinical alert |
| `RECOVERY` | Normalization of physiological indicators | Automatically transitions alerts to `RESOLVED` |
| `NOISY_SENSOR` | Severe high-frequency variance / spikes | Flags `DATA_QUALITY_COMPROMISED` → Data verification request |
| `MISSING_DATA` | Leads disconnected / partial missing channels | Bedside sensor check order set |
| `PERSISTENT_ABNORMALITY` | Sustained elevation / depression | Deduplication suppresses duplicate alarm spam |

---

## 3. Web Dashboard & REST API

The prototype provides a modern, responsive Clinician Dashboard served at `http://localhost:5050/`.
**Zero Node.js/npm dependencies** — operates purely via Python Flask + CDN-delivered React 18, Tailwind CSS, and Chart.js.

### Key Dashboard Capabilities:
- **Multi-Bed Strip**: Live overview of Bed 101, Bed 102, Bed 103 with live status badges.
- **Continuous Telemetry Monitor**: Large vital signs monitors (HR, MAP, SpO2, RR) with color-coded warning thresholds.
- **Real-Time Rolling Waveform**: Smooth Chart.js canvas plotting multi-trace vital trends via Server-Sent Events.
- **5-Agent Autonomous Trace**: Transparent view into each agent's live state, model probability, and reasoning citations.
- **Actionable Order Set**: Interactive checklist of suggested orders generated by the Care Coordination Agent.
- **Clinical Alert Center**: Interactive alert state machine (`NEW → ACTIVE → ACKNOWLEDGED → RESOLVED`) with 1-click clinician acknowledgment.
- **Live Scenario Injector**: Inject any of the 7 clinical scenarios in real time with immediate physiological response.

### API Endpoints:
- `GET /api/health`: Supervisor snapshot across all 5 agents and runtime metrics.
- `GET /api/metrics`: Operational metrics, agent latencies, and adaptive bypass ratios.
- `GET /api/patients`: Summary list of all monitored beds.
- `GET /api/patients/<id>`: Full multi-agent provenance and state for a specific patient.
- `GET /api/patients/<id>/vitals?limit=60`: Time series telemetry buffer for charting.
- `GET /api/alerts`: Active and historical clinical alerts.
- `POST /api/alerts/<id>/acknowledge`: Clinician acknowledgment of an alert.
- `POST /api/alerts/<id>/resolve`: Manual or clinical resolution of an alert.
- `POST /api/scenarios/trigger`: Dynamically switch clinical scenarios (`{"patient_id": 101, "scenario": "GRADUAL_DETERIORATION"}`).
- `GET /api/events/stream`: Server-Sent Events (SSE) live push stream.

---

## 4. Empirical Evaluation & Mode Comparison

CareMatrix includes an automated research evaluation suite that benchmarks the 5-agent pipeline across all 7 clinical scenarios and quantifies the computational workload reduction achieved by the Adaptive Execution Gatekeeper:

```bash
python3 run_evaluation.py
```

### Empirical Results Summary:

| Scenario | Detection Step | Full 5-Agent Chain | Action Generated | Alert Lifecycle |
| :--- | :---: | :---: | :--- | :--- |
| `STABLE` | None (Bypassed) | Bypassed | `NONE` | 0 Alerts (Gatekeeper active) |
| `GRADUAL_DETERIORATION` | Step 52 | Executed | `REQUEST_DATA_VERIFICATION` | 1 Active Alert |
| `SUDDEN_ABNORMALITY` | Step 24 | Executed | `REQUEST_DATA_VERIFICATION` | 1 Lifecycle Alert |
| `RECOVERY` | Step 20 | Executed | `REQUEST_DATA_VERIFICATION` | Alert Resolved |
| `NOISY_SENSOR` | Step 56 | Executed | `REQUEST_DATA_VERIFICATION` | Flagged & Verified |
| `MISSING_DATA` | None (Safe) | Bypassed | `NONE` | Handled NaN safely |
| `PERSISTENT_ABNORMALITY` | Step 20 | Executed | `REQUEST_DATA_VERIFICATION` | Deduplication active |

### Mode A (Exhaustive) vs Mode B (Adaptive Gatekeeper):
- **Mode A (Always-On Baseline)**: 500 total agent executions
- **Mode B (CareMatrix Adaptive)**: 104 total agent executions
- **Unnecessary Executions Avoided**: 396
- **Workload Reduction**: **79.2%**

---

## 5. How to Run

### 1. Launch Continuous Live Prototype & Clinician Dashboard
```bash
python3 run_live_prototype.py --port 5050 --open-browser
```
Open **http://localhost:5050/** in your browser.

### 2. Run Autonomous Research Evaluation Suite
```bash
python3 run_evaluation.py
```

### 3. Run Terminal Multi-Agent Demonstration (VitalDB Dataset Cases)
Run the full 5-agent pipeline demonstration on synthetic Case 0 or real VitalDB Case 4:
```bash
# Synthetic test case
python3 run_carematrix.py --case-id 0 --samples 100 --delay 0 --agent-demo

# VitalDB Case 4 (Perioperative patient dataset)
python3 run_carematrix.py --case-id 4 --samples 100 --start-sample 800 --delay 0 --agent-demo
```

### 4. Run Test Suite
Run the complete unit, alert lifecycle, server API, multi-patient isolation, and supervisor recovery test suite:
```bash
python3 -m unittest discover -s tests -v
```

---

## 5. Environment Variables (Optional Gemini LLM Integration)

CareMatrix includes deterministic fallback rules for all agents and operates without an internet connection or external API keys.

To enable **Google Gemini 2.5 Flash** for clinical reasoning synthesis:
```bash
export GEMINI_API_KEY="your-gemini-api-key-here"
```
Or create a `.env` file in the root directory:
```bash
echo 'GEMINI_API_KEY="your-gemini-api-key-here"' > .env
```
# CareMatrix---FYP
