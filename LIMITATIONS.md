# CareMatrix Limitations and Scope

This document provides a transparent, rigorous assessment of the clinical, algorithmic, and operational limitations of the **CareMatrix** multi-agent decision-support architecture.

---

## 1. Clinical Dataset and Population Scope

1. **Single-Center Surgical Population**: The risk prediction model and empirical validation are based on the **VitalDB** open dataset, collected from surgical patients undergoing anesthesia at Seoul National University Hospital. Physiological dynamics during intraoperative anesthesia (drug-induced vasodilation, mechanical ventilation, surgical stimulus) differ substantially from medical ICU pathology (sepsis, acute respiratory distress syndrome, cardiogenic shock) and general inpatient floor populations.
2. **Outcome Definition and Imbalance**: The high-risk target label was defined as in-hospital mortality or ICU admission > 0 days, exhibiting an empirical positive prevalence of approximately 10.0% (16 / 160 cases). While case-stratified splitting preserved distribution across splits, class imbalance limits positive predictive value (PPV: 0.1728 on test split).
3. **Multi-Center Generalizability**: The system has not yet been validated on multi-center ICU cohorts (e.g., MIMIC-IV Waveform Database or eICU Collaborative Research Database). Performance on external patient cohorts may degrade due to differing monitoring hardware, transducer zeroing practices, and clinical workflows.

---

## 2. Sensor and Signal Fidelity Constraints

1. **Intermittent Non-Invasive Blood Pressure (NIBP)**: While ECG-derived heart rate, plethysmographic SpO2, and respiratory rate provide second-by-second telemetry, NIBP cuff measurements are obtained intermittently (typically every 3 to 5 minutes). CareMatrix applies a bounded forward-fill hold capped at 300 seconds (5 minutes) with explicit staleness flags (`nibp_stale`, `nibp_age_seconds`). Abrupt hypotensive events occurring between cuff measurement cycles cannot be detected until the subsequent cuff inflation or until compensatory tachycardia/tachypnea manifests.
2. **Sensor Disconnection vs True Collapse**: Although CareMatrix differentiates transient single-sample spikes from sustained physiological shifts, a patient simultaneously experiencing sensor displacement and acute instability may produce ambiguous signals. In such scenarios, the system fail-safes to `REQUEST_DATA_VERIFICATION`, prompting immediate bedside clinical attendance.
3. **Body Temperature Latency**: Core body temperature (BT) changes over hours rather than seconds. Consequently, temperature is processed as a context feature by the 54-feature Risk Agent, but is omitted from rapid 5-second real-time alert loops.

---

## 3. Machine Learning and Statistical Boundaries

1. **Test Split Sample Size**: Due to strict case-level splitting without data leakage, the held-out test split comprises 24 cases (282 sliding windows, 2 positive outcome cases). While this guarantees zero train-to-test window contamination, larger prospective cohorts are needed to narrow confidence intervals for sensitivity and specificity.
2. **Demographic Imputation**: In real-time streams where patient demographic records (age, sex, BMI, ASA score) are not yet integrated or available from the EHR, the Risk Agent transparently records `context_source: unavailable` and applies training-cohort median values. This may slightly underestimate risk for extreme demographic outliers.
3. **Feature Approximation Fallback**: If an upstream failure prevents window delivery, the Risk Agent approximates features from the monitoring summary. While verified to run safely, feature approximation yields slightly different probabilities than exact 5-minute statistical windows.

---

## 4. Large Language Model (LLM) and RAG Reasoning

1. **Non-Diagnostic Decision Support**: CareMatrix is strictly a **decision-support tool** intended to augment clinical workflow, not an autonomous diagnostic medical device. It does not prescribe medications, adjust infusion pumps, or initiate invasive procedures. All proposed actions and order sets require human clinical evaluation.
2. **Variable Network Latency**: Cloud-hosted LLM inference (Google Gemini) introduces non-deterministic network latency (500 ms to 2,500 ms). While acceptable for surveillance triage, real-time alerting depends exclusively on the deterministic safety engine (`< 10 ms` latency) to guarantee bounded response times.
3. **Deterministic Safety Invariants**: Generative language models are susceptible to prompt sensitivity and potential narrative drift. CareMatrix enforces a strict deterministic safety barrier:
   - The LLM can never downgrade a deterministic escalation.
   - Any LLM attempt to propose `ROUTINE` priority for a model `HIGH RISK` prediction is automatically rejected by the safety arbiter and flagged as an LLM reasoning conflict.
4. **NCBI PubMed Retrieval Bounds**: Real-time PubMed searches rely on dynamic keyword queries over NCBI's E-utilities API. Keyword extraction may occasionally retrieve broad biomedical abstracts rather than specific bedside protocols. To mitigate this, CareMatrix enforces an automated fallback to verified, curated clinical guidelines (Surviving Sepsis Campaign, NEWS2, ASA Standards).

---

## 5. Multi-Agent Negotiation Bounds

1. **One-Round Challenge Cap**: Negotiation between the Data Analysis Agent and Risk Agent is capped at exactly **one round** per event chain. While this prevents unbounded recursion and compute exhaustion during patient deterioration, deeply discordant clinical scenarios are routed directly to human clinician arbitration (`SCHEDULE_CLINICIAN_REVIEW`).
2. **Communication Overhead**: Mode A (Always-On Full Pipeline) incurs significant computational and latency overhead (~791 ms per sample) when executed continuously. The Adaptive Gatekeeper (Mode B) resolves this during routine stability, but during prolonged multi-vital deterioration, downstream agent throughput requires adequate compute resources.
