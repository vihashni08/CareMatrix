"""LLM reasoning module for the CareMatrix Monitoring Agent.

Given the current deviation, trend, and signal-quality context computed by the
deterministic monitoring engine, the LLM proposes whether the physiological
changes represent genuine deterioration, sensor artifact, or transient noise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from typing import Any

from communication.llm_client import (
    BaseGeminiReasoner,
    LLMReasonerError,
    LLMSchemaValidationError,
    LLMTimeoutError,
    LLMUnavailableError,
    clean_json_text,
    find_api_key,
)

MONITORING_SYSTEM_INSTRUCTION = """You are a senior clinical physiological monitoring specialist for CareMatrix.
Your task is to analyze real-time vital sign telemetry including relative deviations from baseline,
temporal trends, and sensor signal quality indicators.

Determine whether the observed physiological state represents:
1. GENUINE_DETERIORATION: Consistent, clinically plausible multi-vital or severe single-vital physiological decline.
2. SENSOR_ARTIFACT: Unrealistic abrupt spikes, poor sensor signal quality flags, or physically implausible readings.
3. PHYSIOLOGICAL_NOISE: Benign transient fluctuation, patient movement, or minor sub-clinical drift.
4. STABLE: Parameters normal or within expected safe baseline variation.

You must respond ONLY with a valid JSON object matching this schema:
{
  "classification": "GENUINE_DETERIORATION" | "SENSOR_ARTIFACT" | "PHYSIOLOGICAL_NOISE" | "STABLE",
  "rationale": "Clear, concise 1-2 sentence clinical explanation of the classification.",
  "confidence": 0.0 to 1.0,
  "recommended_action": "Suggested immediate technical or monitoring action"
}
"""


class MonitoringClassification(str, Enum):
    GENUINE_DETERIORATION = "GENUINE_DETERIORATION"
    SENSOR_ARTIFACT = "SENSOR_ARTIFACT"
    PHYSIOLOGICAL_NOISE = "PHYSIOLOGICAL_NOISE"
    STABLE = "STABLE"


@dataclass
class MonitoringLLMProposal:
    """Structured proposal from Gemini for vital deviation analysis."""

    classification: str
    rationale: str
    confidence: float
    recommended_action: str = ""
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_monitoring_prompt(context: dict[str, Any]) -> str:
    """Format monitoring context into a concise structured prompt for Gemini."""
    vital_details = context.get("vital_details", {})
    alert_vitals = context.get("alert_eligible_vitals", [])
    overall_severity = context.get("overall_severity", "normal")
    is_candidate = context.get("is_candidate", False)
    is_persistent = context.get("is_persistent", False)

    vitals_summary = []
    for vital, info in vital_details.items():
        vitals_summary.append(
            f"- {vital}: current={info.get('current')}, baseline={info.get('baseline')}, "
            f"dev={info.get('relative_deviation'):.2f}, trend={info.get('trend')}, "
            f"signal_quality={info.get('signal_quality')}, severity={info.get('severity')}"
        )

    prompt = f"""EVALUATE PHYSIOLOGICAL TELEMETRY:
Overall Severity: {overall_severity}
Candidate Alert: {is_candidate}
Persistent Alert Condition Met: {is_persistent}
Alert Eligible Vitals: {alert_vitals}

Vital Sign Trajectories:
{chr(10).join(vitals_summary)}

Evaluate whether this observation represents GENUINE_DETERIORATION, SENSOR_ARTIFACT, PHYSIOLOGICAL_NOISE, or STABLE.
Return your evaluation strictly as the specified JSON object."""
    return prompt


class GeminiMonitoringReasoner(BaseGeminiReasoner):
    """LLM reasoner proposing deterioration vs artifact classification for MonitoringAgent."""

    def propose(self, context: dict[str, Any]) -> MonitoringLLMProposal:
        """Invoke Gemini on vital context and return validated structured proposal."""
        prompt = build_monitoring_prompt(context)
        raw_text = self.call_gemini(
            prompt=prompt,
            system_instruction=MONITORING_SYSTEM_INSTRUCTION,
        )
        return self.validate_and_parse(raw_text)

    def validate_and_parse(self, raw_text: str) -> MonitoringLLMProposal:
        """Validate and parse Gemini response into MonitoringLLMProposal."""
        cleaned = clean_json_text(raw_text)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMSchemaValidationError(f"Malformed JSON in Gemini monitoring response: {exc}") from exc

        if not isinstance(payload, dict):
            raise LLMSchemaValidationError("Expected JSON object at root of Gemini response.")

        required_keys = ["classification", "rationale", "confidence"]
        missing = [k for k in required_keys if k not in payload]
        if missing:
            raise LLMSchemaValidationError(f"Missing required fields in Gemini response: {missing}")

        classification_raw = str(payload["classification"]).strip().upper()
        valid_classes = {c.value for c in MonitoringClassification}
        if classification_raw not in valid_classes:
            raise LLMSchemaValidationError(
                f"Invalid classification '{classification_raw}'. Must be one of: {sorted(valid_classes)}"
            )

        rationale = str(payload["rationale"]).strip()
        if not rationale:
            raise LLMSchemaValidationError("rationale cannot be empty.")

        try:
            confidence = float(payload["confidence"])
            if not (0.0 <= confidence <= 1.0):
                raise ValueError()
        except (TypeError, ValueError):
            raise LLMSchemaValidationError("confidence must be a float between 0.0 and 1.0.")

        recommended_action = str(payload.get("recommended_action", "")).strip()

        return MonitoringLLMProposal(
            classification=classification_raw,
            rationale=rationale,
            confidence=confidence,
            recommended_action=recommended_action,
            raw_response=raw_text,
        )


__all__ = [
    "GeminiMonitoringReasoner",
    "MonitoringClassification",
    "MonitoringLLMProposal",
    "build_monitoring_prompt",
]
