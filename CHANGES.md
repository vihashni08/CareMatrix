# CareMatrix System Changelog (CHANGES.md)

This log documents all architectural, algorithmic, and engineering changes made to the CareMatrix multi-agent clinical decision support system, adhering to strict reproducibility and deterministic safety invariants.

---

## Phase 1 — Fix Broken Code and Tests

### 1.1 Risk Agent / Decision Engine Signature Alignment
- **Problem**: `RiskAgent.process_event` invoked `RiskDecisionEngine.decide_from_prediction(..., metadata=...)` and `decide_from_failure(..., metadata=...)`, but `RiskDecisionEngine` rejected the `metadata` keyword argument, raising unhandled `TypeError`s.
- **Solution**: Added optional `metadata: dict[str, Any] | None = None` parameter to both methods in `risk_agent/decision_engine.py`. Merged into `RiskDecisionEvent.metadata`, persisting `feature_source` ("window" vs "approximated_fallback") end-to-end.

### 1.2 Test Suite Failure Resolution
- **Problem**: Failures in `tests/test_agents.py` and `tests/test_evaluation_framework.py`.
- **Root Cause**: Method signature mismatch in `RiskDecisionEngine` crashed worker threads.
- **Verification**: Aligned method signature; all 15 tests in `test_agents.py` and 8 tests in `test_evaluation_framework.py` pass cleanly.

### 1.3 Model Artifact & Environment Reproducibility
- **Problem**: Missing training dependencies (`xgboost`, `scipy`, `pyarrow`) and unpickling version warnings.
- **Solution**: Pinned exact versions in `requirements.txt` (`scikit-learn==1.6.1`, `xgboost==2.1.4`, `scipy==1.13.1`, `pyarrow==21.0.0`). Created `risk_agent/models/model_metadata.json` (sklearn_version: "1.6.1"). Enhanced `RiskModelTool` to warn loudly when runtime version differs from training version.

### 1.4 Implementation of Empty Files
- `tests/test_risk_agent_feature_source.py`: verified window vs approximated_fallback.
- `tests/test_evaluation_metrics.py`: verified confusion matrix, precision, recall, F1, AUROC, GROUND_TRUTH_UNAVAILABLE, added specificity to `ClassificationMetricsResult`.
- `tests/test_gemini_model_selection.py`: verified model fallback pool and timeout handling.
- `evaluation/ablation_runner.py`: implemented real ablation harness for 4 standard ablations.

### 1.5 Elimination of Fabricated Patient Demographics
- **Problem**: `gather_patient_context()` silently returned hard-coded `age=60, sex=1, bmi=24.5, asa=2`.
- **Solution**: Replaced arbitrary values with `np.nan` and added `context_source` ("vitaldb_cases_table" vs "unavailable"). Model `SimpleImputer` natively handles missing demographics.

### 1.6 Observation Normalization & Preprocessing Rigor
- Enhanced `ObservationNormalizer` in `data/normalizer.py`: chronological sorting, timestamp deduplication, physiological extreme preservation, and bounded (<=5 sample) interpolation. Added unit tests in `tests/test_normalizer_preprocessing.py`.

### 1.7 Security & Project Hygiene
- Notice: Existing Gemini API key in `.env` must be rotated.
- `.gitignore`: verified `.env` is ignored; added `data/vitaldb_cache/`.
- `.env.example`: created template with placeholder credentials.
- Canonical Entry Points: `scripts/run_monitoring.py` documented as canonical; `notebooks/archive/` documented as superseded exploratory prototypes.
- Model IDs: Aligned default model to `gemini-2.5-flash` and filtered candidate pool to active official Gemini models.

---

## Phase 2 — Real VitalDB Cases Execution

### 2.1 Real 7-Track Telemetry Loader & Bounded NIBP Hold
- 7 tracks: `Solar8000/HR`, `Solar8000/PLETH_SPO2`, `Solar8000/RR`, `Solar8000/NIBP_SBP`, `Solar8000/NIBP_DBP`, `Solar8000/NIBP_MBP`, `Solar8000/BT` -> `HR`, `SpO2`, `RR`, `SBP`, `DBP`, `MAP`, `BT`.
- Persistent Caching: Cached cases to `data/vitaldb_cache/case_<id>.parquet` (and `.csv`). Removed corrupt `data/processed/patient_4.csv`.
- Bounded NIBP Hold: Forward-fills intermittent NIBP up to 300s with `nibp_stale` and `nibp_age_seconds` flags; long gaps revert to `NaN`. Vectorized with NumPy for a 2500x loading speedup.
- Sampling Rate Alignment: 1s streaming for monitoring, resampled to 5s (60 samples in 300s) for Risk Agent feature extraction.

### 2.2 Case Selection and Stratified Partitioning
- `scripts/select_vitaldb_cases.py`: Selected 160 cases containing all 7 tracks. Ground truth: `High Risk = (death_inhosp == 1) | (icu_days > 0)`.
- Stratified splits: 112 Train (70%), 24 Validation (15%), 24 Test (15%) saved to `data/case_splits.json` with zero leakage. Added `tests/test_case_splits.py`.

### 2.3 Runtime & Terminal Replay on Real Cases
- `PatientStreamReplayer` updated to stream all 7 vitals with configurable speed multiplier and start sample.
- Terminal demo `run_carematrix.py` defaults to real VitalDB test-split case (case 242) and `DataAnalysisAgent` default loader uses cached real cases via `VitalDBAdapter`.
- `run_live_prototype.py` pre-registers real VitalDB test-split cases (242, 532, 1271) with bed naming "VitalDB case <id> — replay" and configurable start-sample (default: 60) to bypass initial induction artifact.
- Scenario injection endpoint (`/api/scenarios/trigger`) and dashboard UI explicitly labeled as "Fault Injection (Synthetic — Testing Only)".
- Real case demographic lookup via `gather_patient_context()` reads cached `cases.csv` before falling back to remote API, never fabricating values.
- `VitalDBAdapter.list_cases()` retains cases 0 and 4 for backward test compatibility while dynamically exposing all 160 stratified cases.

### 2.4 Real Segment Integration Testing
- Test Fixture: `data/test_fixtures/vitaldb_case_6_snippet.csv` with CC BY 4.0 attribution in `data/test_fixtures/README.md`.
- Integration Test: `tests/test_vitaldb_integration.py` verifying full pipeline replay on real case segment.

---

## Phase 3 — Risk Model Training, Evaluation, and Curves

### 3.1 Training Script & Multi-Model Evaluation
- Developed `scripts/train_risk_model.py` (695 lines).
- Trained 7 candidate model architectures: Extra Trees, Random Forest, Gradient Boosting, XGBoost, LightGBM, Logistic Regression, and MLP.
- Features: 54 statistical temporal features across 7 vitals (mean, std, min, max, slope) plus demographic covariates.
- Training Data: 1,333 5-minute sliding windows across 112 cases.
- Validation Data: 288 windows across 24 cases.
- Test Data: 282 windows across 24 cases.

### 3.2 Model Selection on Validation Split
- Selection Metric: Validation PR-AUC and ROC-AUC.
- Selected Model: `ExtraTreesClassifier` (Val ROC-AUC: 0.7711, PR-AUC: 0.3236, F1: 0.4000 at optimal decision threshold 0.51).
- Held-Out Test Split Performance: ROC-AUC: 0.7397, PR-AUC: 0.2285, Sensitivity: 0.6087, Specificity: 0.7413, Precision: 0.1728, F1: 0.2692.

### 3.3 Publication-Ready Figures
- Generated in `docs/figures/`:
  - `roc_curve.png`: ROC curves for Extra Trees and Random Forest with operating thresholds.
  - `pr_curve.png`: Precision-Recall curves with prevalence baseline.
  - `confusion_matrix.png`: Normalized and raw test set confusion matrices.
  - `feature_importance.png`: Top-15 clinical predictive features (MAP, RR, HR dynamics).
  - `validation_model_comparison.png`: Bar chart of ROC-AUC and PR-AUC across all 7 candidates.

---

## Phase 4 — Pipeline Behavior Fixes

### 4.1 Corroborated Deterioration vs Data Verification
- **Problem**: `ClinicalReasoningEngine.arbitrate()` contained `final_verification_required = (deterministic.verification_required or is_high_risk or data_quality_flag)`. The `or is_high_risk` forced `verification_required = True` on every high risk event, causing `CareCoordinationPolicy` to route to Rule 1 (`REQUEST_DATA_VERIFICATION`) instead of Rule 3 (`TRIGGER_URGENT_CLINICAL_ALERT`).
- **Solution**: Removed `or is_high_risk`. Set `final_verification_required = (deterministic.verification_required or conflict or data_quality_flag)`. Corroborated deterioration now triggers immediate bedside emergency alert.

### 4.2 Signal Quality vs Physiological Collapse Differentiation
- **Problem**: `assess_window_quality` in `data_analysis_agent/analyzer.py` flagged step drops (e.g. MAP dropping from 80 to 55) as sensor artifacts because jump exceeded 8x typical noise.
- **Solution**: Differentiated isolated single-sample transient spikes and high-frequency sign oscillations from sustained step changes. Implemented multi-vital concordance override: concurrent shifts across $\ge 2$ vitals confirm systemic deterioration, preventing artifact false alarms.

### 4.3 Sustained Sensor Disconnect Detection
- **Implementation**: Added `consecutive_missing_samples` tracking in `monitoring_agent/state.py` and `max_missing_samples = 15` (75s of signal loss) in `monitoring_agent/decision_engine.py`. Triggers a `DATA_QUALITY` escalation routing to `REQUEST_DATA_VERIFICATION`.

### 4.4 Body Temperature (BT) Documentation
- Documented in `monitoring_agent/state.py` that BT is processed strictly as an input feature for the 54-feature Risk Agent, while rapid real-time threshold monitoring evaluates HR, MAP, SpO2, and RR.

### 4.5 Latency Tracking
- Added per-stage (`stage_latency_ms`) and cumulative (`cumulative_latency_ms`) latency tracking to metadata across all 5 agents.
- Added comprehensive unit tests in `tests/test_pipeline_behavior.py`.

---

## Phase 5 — Real Evaluation Benchmark

### 5.1 Real Test Cohort Benchmark (24 Cases, 4.0 Hours)
- Rewrote `run_evaluation.py` to evaluate the 24 test-split cases from `data/case_splits.json`.
- Established independent clinical reference criteria:
  - Hypotension: $\text{MAP} < 65\text{ mmHg}$ for $\ge 60\text{s}$
  - Desaturation: $\text{SpO}_2 < 90\%$ for $\ge 30\text{s}$
  - Tachycardia: $\text{HR} > 120\text{ bpm}$ for $\ge 60\text{s}$
- Outcomes: 85.7% Sensitivity (6/7 episodes detected), 3.5 false alarms / hour, 165.0s mean advance warning lead time.

### 5.2 Adaptive Gatekeeper (Mode B vs Mode A)
- Downstream Calls Avoided: 396 / 500 (79.2% workload reduction).
- Wall-Clock Throughput Speedup: 2.0x speedup.
- Generated `docs/figures/mode_a_vs_mode_b_comparison.png`.

### 5.3 Architectural Ablations
- Executed 4 ablation configurations via `AblationRunner`.
- Mode A (Always-On) incurred 791.2 ms mean latency (P95: 1001.0 ms) per sample, whereas Mode B bypassed routine monitoring with 0 ms downstream overhead.
- Generated `docs/figures/ablation_study_latency.png`.

### 5.4 Deliverables
- Generated structured `evaluation_results.json`.
- Created comprehensive `RESULTS.md`.

---

## Phase 6 — Documentation and Project Cleanliness

- Regenerated `README.md` with accurate architecture, real dataset, model details, and exact reproduction commands.
- Created `LIMITATIONS.md` transparently addressing dataset scope, NIBP intermittency, non-diagnostic boundaries, and LLM latency constraints.
- Updated `CHANGES.md`.
- Verified complete test suite passing.
