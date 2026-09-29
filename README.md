# CareMatrix — Intelligent Healthcare Decision Support Using Multi-Agent Systems

CareMatrix is an autonomous, event-driven multi-agent clinical decision support architecture for continuous inpatient physiological monitoring, risk prediction, analytical validation, clinical evidence synthesis, and care workflow coordination.

---

## 1. Multi-Agent Architecture

CareMatrix consists of **5 autonomous task agents** coordinated by an independent **Agent Supervisor**:

```text
                  ┌───────────────────────────────┐
                  │    AgentSupervisor Monitor    │ (Thread liveness, heartbeats,
                  │ (Heartbeats & Auto-Recovery)  │  state snapshots, auto-restart)
                  └───────────────┬───────────────┘
                                  │
Continuous Telemetry Stream       │
(Real VitalDB Replay / Multi-Bed) │
              │                   │
              ▼                   │
    ┌───────────────────┐         │
    │  Monitoring Agent │◄────────┤ (Continuous rolling baseline, multi-vital deviation,
    └─────────┬─────────┘         │  sensor disconnect detection, adaptive gatekeeper)
              │ (Escalation / Recovery)
              ▼                   │
    ┌───────────────────┐         │
    │    Risk Agent     │◄────────┤ (ExtraTreesClassifier, 54 features, threshold 0.51,
    └─────────┬─────────┘         │  5-minute rolling window feature extraction)
              │ (Risk Decision Event)
              ▼                   │
    ┌───────────────────┐         │
    │Data Analysis Agent│◄────────┤ (Multi-vital trend slopes, artifact detection,
    └─────────┬─────────┘         │  cross-agent consistency verification)
              │ (Analytical Evidence Event)
              ▼                   │
    ┌───────────────────┐         │
    │Clinical Reasoning │◄────────┤ (Medical RAG via PubMed API + Curated Guidelines,
    │      Agent        │         │  Gemini LLM Reasoner + Deterministic Safety Arbiter)
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

### The 5 Autonomous Task Agents

1. **Monitoring Agent**:
   - Performs continuous patient-specific physiological monitoring using rolling baseline buffers (HR, MAP, SpO2, RR).
   - Differentiates physiological deviations from sensor disconnects (`max_missing_samples = 15`).
   - **Adaptive Execution Gatekeeper**: Completely bypasses downstream ML/LLM execution when vitals are stable, conserving computational resources.

2. **Risk Agent**:
   - Evaluates multi-parameter physiological windows using a trained machine learning model (`ExtraTreesClassifier`, 54 features, decision threshold: `0.51`).
   - Extracts real 5-minute statistical window features (mean, std, min, max, slope) from the buffered stream.
   - Demographics (age, sex, BMI, ASA physical status) are retrieved from clinical records with transparent missingness handling.

3. **Data Analysis Agent**:
   - Computes statistical metrics (variance, rate of change, direction) without diagnostic bias.
   - Differentiates isolated transient sensor spikes and high-frequency noise from sustained multi-vital physiological collapse.
   - Performs cross-agent consistency verification (`SUPPORTING`, `CONFLICTING`, `UNCERTAIN`) and issues bounded challenge-response negotiations (capped at 1 round).

4. **Clinical Reasoning Agent**:
   - Synthesizes physiological trajectories, risk predictions, and analytical evidence.
   - Incorporates **Medical RAG** querying the NCBI PubMed E-utilities API in real time, with automatic fallback to curated clinical guidelines (Surviving Sepsis, NEWS2, ASA monitoring guidelines).
   - Formulates non-diagnostic assessments using Google Gemini (with deterministic safety arbitration ensuring the LLM can never downgrade a deterministic safety escalation).

5. **Care Coordination Agent** *(Non-diagnostic workflow orchestration)*:
   - Evaluates validated clinical reasoning evidence against transparent, deterministic clinical action policies.
   - Dispatches actionable workflow pathways:
     - `TRIGGER_URGENT_CLINICAL_ALERT`: Immediate bedside clinician notification & rapid response activation for acute collapse.
     - `SCHEDULE_CLINICIAN_REVIEW`: Scheduled physician clinical review within 30 minutes for discordant evidence or moderate deterioration.
     - `REQUEST_DATA_VERIFICATION`: Bedside sensor inspection, electrode check, and manual vitals measurement for sensor noise or disconnects.
     - `CONTINUE_ROUTINE_MONITORING`: Standard continuous floor/ICU surveillance for stable patients.
   - Compiles structured **Actionable Order Sets** and enforces **Alert Deduplication**.

6. **Agent Supervisor**:
   - Independent background monitor thread tracking agent health and periodic idle heartbeats.
   - Automatically detects stale or frozen workers and orchestrates graceful restarts with state restoration.

---

## 2. Dataset and Reproducible Model Training

CareMatrix trains its risk model strictly on real surgical patient cases from the **VitalDB** open dataset:
- **Cohort**: 160 surgical cases partitioned into strictly disjoint case-level splits: **112 train cases** (1,333 windows), **24 validation cases** (288 windows), and **24 test cases** (282 windows).
- **Vitals**: 7 synchronized physiological tracks (`HR`, `SpO2`, `RR`, `NIBP_SBP`, `NIBP_DBP`, `NIBP_MBP`, `BT`).
- **NIBP Processing**: Bounded forward-fill hold (maximum 300 seconds) with explicit staleness flags (`nibp_stale`, `nibp_age_seconds`) avoiding unphysiological interpolation.
- **Model Selection**: 7 candidate architectures evaluated on the validation split; `ExtraTreesClassifier` selected based on optimal validation PR-AUC (`0.3236`) and ROC-AUC (`0.7711`) at threshold `0.51`.
- **Test Set Performance**: ROC-AUC: **0.7397**, PR-AUC: **0.2285**, Sensitivity: **0.6087**, Specificity: **0.7413**.

To retrain the model and regenerate all validation figures:
```bash
python3 scripts/train_risk_model.py
```

Generated training figures are saved to `docs/figures/`:
- `roc_curve.png`
- `pr_curve.png`
- `confusion_matrix.png`
- `feature_importance.png`
- `validation_model_comparison.png`

---

## 3. Empirical Evaluation Findings

Run the full evaluation benchmark:
```bash
python3 run_evaluation.py
```

### 3.1 Adaptive Gatekeeper (Mode B vs Mode A)
Comparing CareMatrix's Adaptive Gatekeeper against an Always-On Full Pipeline on identical mixed telemetry:
- **Downstream Calls Avoided**: **396 / 500** calls avoided (**79.2% workload reduction**).
- **Throughput Speedup**: **2.0x wall-clock speedup** during mixed monitoring.
- *Figure Reference: `docs/figures/mode_a_vs_mode_b_comparison.png`*

### 3.2 Real VitalDB Test Cohort (24 Cases, 4.0 Hours)
Benchmarked against objective independent clinical deterioration criteria (sustained hypotension MAP < 65 for $\ge 60$s, desaturation SpO2 < 90% for $\ge 30$s, tachycardia HR > 120 for $\ge 60$s):
- **Cohort Sensitivity**: **85.7%** (6 of 7 reference episodes detected).
- **False Alarm Rate**: **3.5 false alarms per monitoring hour**.
- **Mean Advance Warning Lead Time**: **165.0 seconds** prior to reference threshold breach.

### 3.3 Clinical Simulation Scenarios
| Scenario | Abnormality Detected | Detection Step | Risk Probability | Feature Source | Quality | Consistency | Care Action Pathway |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `STABLE` | NO | — | — | none | OK | N/A | `NONE` (Bypassed) |
| `GRADUAL_DETERIORATION` | YES | Step 45 | 0.2225 | window | OK | CONFLICTING | `REQUEST_DATA_VERIFICATION` |
| `SUDDEN_ABNORMALITY` | YES | Step 24 | 0.2256 | window | OK | CONFLICTING | `TRIGGER_URGENT_CLINICAL_ALERT` |
| `RECOVERY` | YES | Step 20 | 0.2323 | window | OK | CONFLICTING | `REQUEST_DATA_VERIFICATION` |
| `NOISY_SENSOR` | YES | Step 41 | 0.2331 | window | FLAGGED | UNCERTAIN | `REQUEST_DATA_VERIFICATION` |
| `MISSING_DATA` | NO | — | — | none | OK | N/A | `NONE` (Bypassed) |
| `PERSISTENT_ABNORMALITY` | YES | Step 20 | 0.2517 | window | OK | CONFLICTING | `TRIGGER_URGENT_CLINICAL_ALERT` |

---

## 4. How to Run

### 1. Launch Continuous Live Prototype & Clinician Dashboard
```bash
python3 run_live_prototype.py --port 5050 --open-browser
```
Open **http://localhost:5050/** in your browser.

### 2. Run Comprehensive Research Evaluation Suite
```bash
python3 run_evaluation.py
```
Outputs `evaluation_results.json`, terminal summary tables, and charts in `docs/figures/`.

### 3. Run Live Multi-Agent Replay on Real VitalDB Test-Split Cases
```bash
# Replay real test-split case 242 (perioperative patient telemetry)
python3 run_carematrix.py --case-id 242 --samples 100 --start-sample 800 --delay 0 --agent-demo
```

### 4. Run Complete Unit and Architecture Test Suite
```bash
python3 -m unittest discover -s tests -v
```

---

## 5. Environment Variables (Optional Gemini LLM & PubMed Integration)

CareMatrix functions completely offline out of the box using deterministic safety rules, local scikit-learn models, and curated clinical guidelines.

To enable live Google Gemini LLM synthesis and NCBI PubMed retrieval:
```bash
# .env file
GEMINI_API_KEY="your-gemini-api-key-here"
NCBI_API_KEY="optional-ncbi-api-key"
NCBI_CONTACT_EMAIL="clinical-team@carematrix.local"
```
