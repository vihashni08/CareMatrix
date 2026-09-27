"""Curated medical knowledge base loader for CareMatrix clinical reasoning.

Provides authoritative, versioned clinical guidance on vital sign deviations,
deterioration patterns, sensor artifact troubleshooting, and escalation protocols.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clinical_reasoning_agent.rag.schemas import MedicalDocument


# Curated, authoritative medical guidance corpus relevant to intensive care monitoring
CURATED_MEDICAL_KNOWLEDGE: list[dict[str, Any]] = [
    {
        "document_id": "GUIDELINE_HEMODYNAMIC_001",
        "title": "Hemodynamic Deterioration, Acute Hypotension, and Shock Monitoring",
        "source": "Intensive Care Society & Surviving Sepsis Campaign Guidelines",
        "source_type": "clinical_guideline",
        "publication_date": "2023-10",
        "section": "Mean Arterial Pressure (MAP) and Tachycardia Thresholds",
        "text": (
            "Mean Arterial Pressure (MAP) below 65 mmHg is a critical clinical threshold associated with impaired "
            "autoregulation and inadequate tissue perfusion in intensive care patients. Concurrent tachycardia (heart rate "
            "> 100 bpm) with declining MAP represents a classic physiological compensatory mechanism attempting to preserve "
            "cardiac output in the presence of hypovolemia, vasodilation, or early septic shock. The shock index (heart rate "
            "divided by systolic blood pressure > 0.9) serves as an early indicator of occult hypoperfusion prior to severe "
            "systolic pressure collapse. Recommended bedside actions include immediate clinical assessment of end-organ "
            "perfusion (capillary refill, conscious level, urine output), confirmation of non-invasive blood pressure readings "
            "with manual verification, fluid responsiveness evaluation, and urgent escalation to attending intensive care team."
        ),
        "metadata": {
            "keywords": ["hypotension", "MAP", "tachycardia", "shock", "deterioration", "sepsis", "hemodynamic"],
            "target_vitals": ["MAP", "HR", "SBP"],
        },
    },
    {
        "document_id": "GUIDELINE_TACHYCARDIA_002",
        "title": "Acute Tachycardia and Cardiovascular Instability in Monitored Patients",
        "source": "American College of Cardiology / AHA Bedside Monitoring Guidance",
        "source_type": "clinical_guideline",
        "publication_date": "2024-01",
        "section": "Sinus Tachycardia Etiology and Hemodynamic Assessment",
        "text": (
            "Sustained or rapidly rising heart rate (> 100 bpm) in monitored bedded patients is most commonly a secondary "
            "physiological response to acute systemic stress, pain, systemic inflammatory response, fever, hypovolemia, "
            "or myocardial ischemia. When tachycardia occurs simultaneously with downward trends in blood pressure, it "
            "indicates decompensating circulatory shock requiring rapid fluid assessment and targeted stabilization. "
            "Isolated tachycardia without pressure collapse warrants close surveillance, 12-lead ECG review to exclude "
            "supraventricular or ventricular arrhythmias, and assessment of volume status before considering rate-control agents."
        ),
        "metadata": {
            "keywords": ["tachycardia", "heart rate", "HR", "arrhythmia", "ischemia", "cardiovascular"],
            "target_vitals": ["HR", "SBP", "DBP"],
        },
    },
    {
        "document_id": "GUIDELINE_RESPIRATORY_003",
        "title": "Respiratory Instability and Oxygen Desaturation Monitoring",
        "source": "British Thoracic Society & ATS Clinical Practice Guideline",
        "source_type": "clinical_guideline",
        "publication_date": "2023-08",
        "section": "Tachypnea, Hypoxemia, and Early Respiratory Distress",
        "text": (
            "Tachypnea (respiratory rate > 20 breaths per minute) is one of the most sensitive early indicators of general "
            "physiological deterioration, metabolic acidosis, or worsening gas exchange. Oxygen saturation (SpO2) falling "
            "below 92% (or below 88% in hypercapnic respiratory failure) signifies acute hypoxemia requiring immediate "
            "supplemental oxygen titration and airway patency verification. The combination of tachypnea and decreasing SpO2 "
            "signals impending respiratory failure or pulmonary edema. Bedside protocols mandate checking probe placement, "
            "auscultating breath sounds, verifying supplemental FiO2 delivery, and preparing high-flow or non-invasive ventilation."
        ),
        "metadata": {
            "keywords": ["respiratory", "tachypnea", "RR", "SpO2", "hypoxemia", "oxygenation"],
            "target_vitals": ["RR", "SpO2"],
        },
    },
    {
        "document_id": "GUIDELINE_SENSOR_QUALITY_004",
        "title": "Bedside Sensor Artifact Recognition and Signal Integrity Verification",
        "source": "American Association of Critical-Care Nurses (AACN) Practice Standards",
        "source_type": "physiological_protocol",
        "publication_date": "2023-11",
        "section": "Sensor Disconnect, Motion Artifact, and Transducer Calibration",
        "text": (
            "Physiological monitoring signals are prone to mechanical artifacts that mimic clinical deterioration. Common "
            "artifacts include optical photoplethysmograph sensor displacement causing falsely depressed SpO2 readings, "
            "electrocardiographic lead motion artifact causing erratic heart rate counters, and NIBP cuff slippage or patient "
            "movement causing spurious hypotension or missing data. Whenever an automated monitoring system flags data quality "
            "abnormalities (high missingness, excessive signal noise, or abrupt unphysiological step changes), clinicians must "
            "manually verify peripheral pulse, confirm transducer calibration, check electrode adhesion, and obtain a manual "
            "blood pressure reading before administering invasive therapy or vasoactive medications."
        ),
        "metadata": {
            "keywords": ["sensor", "artifact", "quality", "probe", "calibration", "data_quality_flag", "missingness"],
            "target_vitals": ["SpO2", "HR", "MAP", "NIBP"],
        },
    },
    {
        "document_id": "GUIDELINE_ESCALATION_005",
        "title": "National Early Warning Score (NEWS2) and Critical Care Escalation Protocols",
        "source": "Royal College of Physicians NEWS2 Standardized Guidance",
        "source_type": "consensus_statement",
        "publication_date": "2024-03",
        "section": "Physiological Escalation Triggers and Response Times",
        "text": (
            "The National Early Warning Score 2 (NEWS2) aggregates heart rate, blood pressure, respiratory rate, oxygen "
            "saturation, and temperature into a standardized risk tier. Patients exhibiting multi-vital physiological "
            "derangements (e.g. aggregate NEWS score >= 5 or any single extreme parameter score of 3) qualify for urgent "
            "bedside review within 30 minutes by a clinician with critical care competencies. Immediate continuous "
            "physiological monitoring is mandated. For scores >= 7, immediate emergency transfer or Medical Emergency Team "
            "(MET) bedside attendance is required, with preparation for advanced airway and circulatory stabilization."
        ),
        "metadata": {
            "keywords": ["escalation", "NEWS2", "priority", "urgent", "critical", "deterioration"],
            "target_vitals": ["HR", "MAP", "RR", "SpO2", "BT"],
        },
    },
]


class DocumentLoader:
    """Loads and validates authoritative medical knowledge documents."""

    def __init__(self, custom_documents: list[dict[str, Any]] | None = None):
        self.raw_data = custom_documents if custom_documents is not None else CURATED_MEDICAL_KNOWLEDGE

    def load_documents(self) -> list[MedicalDocument]:
        """Parse raw document dictionaries into typed MedicalDocument objects."""
        documents: list[MedicalDocument] = []
        for d in self.raw_data:
            doc = MedicalDocument(
                document_id=d.get("document_id", "DOC_UNKNOWN"),
                title=d.get("title", "Untitled Clinical Document"),
                source=d.get("source", "Authoritative Source"),
                source_type=d.get("source_type", "clinical_guideline"),
                publication_date=d.get("publication_date", "2024"),
                section=d.get("section", "General"),
                text=d.get("text", "").strip(),
                metadata=d.get("metadata", {}),
            )
            if doc.text:
                documents.append(doc)
        return documents

    load_curated_guidelines = load_documents


__all__ = [
    "CURATED_MEDICAL_KNOWLEDGE",
    "DocumentLoader",
    "MedicalDocument",
]
