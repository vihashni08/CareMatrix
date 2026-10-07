"""LLM reasoning and tool selection module for the CareMatrix Monitoring Agent.

Supports:
1. Genuine, bounded LLM tool-selection loops where Gemini decides which tool to call next
   based on progressive observations and analytical results.
2. Direct proposal generation for safety arbitration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import logging
from typing import Any, Dict, List, Optional

from communication.llm_client import (
    BaseGeminiReasoner,
    LLMReasonerError,
    LLMSchemaValidationError,
    LLMTimeoutError,
    LLMUnavailableError,
    clean_json_text,
    find_api_key,
)

logger = logging.getLogger("CareMatrix.MonitoringLLM")

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

TOOL_SELECTION_SYSTEM_INSTRUCTION = """You are an autonomous clinical physiological monitoring agent for CareMatrix.
Your goal is to inspect streaming patient vital sign observations and decide which analytical tools to invoke.

Available Tools:
- assess_signal_quality: Assess sensor fidelity (good, noisy, artifact, missing).
- invalid_measurement_mask: Flag implausible or detached sensor readings.
- preprocess: Clean raw telemetry and handle missing values.
- calculate_baseline: Calculate rolling median baseline for vitals.
- calculate_deviation: Calculate relative deviation percentage against baseline.
- calculate_trends: Classifies directional vital trajectories (increasing, decreasing, stable).

You must select EXACTLY ONE tool to run next, OR decide that analysis is complete ("finish").
DO NOT repeatedly call the same tool unless new data requires it.

You must respond ONLY with a valid JSON object matching this schema:
{
  "thought": "Clinical reasoning explaining what information is needed next and why",
  "action": "tool" | "finish",
  "tool_name": "assess_signal_quality" | "invalid_measurement_mask" | "preprocess" | "calculate_baseline" | "calculate_deviation" | "calculate_trends" | null,
  "tool_args": {},
  "final_decision": "CONTINUE_MONITORING" | "ESCALATE_TO_RISK" | null,
  "classification": "GENUINE_DETERIORATION" | "SENSOR_ARTIFACT" | "PHYSIOLOGICAL_NOISE" | "STABLE" | null,
  "rationale": "Clinical rationale for the current decision",
  "confidence": 0.0 to 1.0
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


@dataclass
class ToolSelectionStep:
    """Record of a single step in the agentic tool-selection loop."""

    iteration: int
    thought: str
    action: str  # "tool" or "finish"
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    tool_result_summary: Optional[str] = None
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolSelectionDecision:
    """Structured decision returned by Gemini in the tool selection loop."""

    thought: str
    action: str  # "tool" or "finish"
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    final_decision: Optional[str] = None
    classification: Optional[str] = None
    rationale: str = ""
    confidence: float = 1.0
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
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


def build_tool_selection_prompt(
    patient_id: int,
    observations_count: int,
    latest_observation: Dict[str, Any],
    executed_tools: List[str],
    state_summary: Dict[str, Any],
    iteration: int,
    max_iterations: int,
) -> str:
    """Build prompt providing agent state and history for LLM tool selection."""
    obs_lines = [f"  - {k}: {v}" for k, v in latest_observation.items()]

    executed_desc = ", ".join(executed_tools) if executed_tools else "None"

    return f"""PATIENT MONITORING AGENT CYCLE (Iteration {iteration + 1} of max {max_iterations}):
Patient ID: {patient_id}
Total Observations: {observations_count}

Latest Observation:
{chr(10).join(obs_lines)}

Current State Summary:
- Tools Already Executed: {executed_desc}
- Clean Values Available: {state_summary.get('has_clean_values', False)}
- Baseline Established: {state_summary.get('baseline_established', False)}
- Deviations Calculated: {state_summary.get('deviations_calculated', False)}
- Trends Calculated: {state_summary.get('trends_calculated', False)}
- Signal Quality Assessed: {state_summary.get('signal_quality_assessed', False)}
- Current Computed Context: {json.dumps(state_summary.get('vitals_glance', {}))}

TASK:
Determine what analytical tool should be executed next to evaluate the patient's state,
OR choose "finish" if sufficient information has been gathered.
Respond strictly in the specified JSON format.
"""


class GeminiMonitoringReasoner(BaseGeminiReasoner):
    """LLM reasoner providing tool selection and classification for MonitoringAgent."""

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

    def select_tool(
        self,
        patient_id: int,
        observations_count: int,
        latest_observation: Dict[str, Any],
        executed_tools: List[str],
        state_summary: Dict[str, Any],
        iteration: int,
        max_iterations: int = 5,
    ) -> ToolSelectionDecision:
        """Ask Gemini which analytical tool to run next or whether analysis is complete."""
        prompt = build_tool_selection_prompt(
            patient_id=patient_id,
            observations_count=observations_count,
            latest_observation=latest_observation,
            executed_tools=executed_tools,
            state_summary=state_summary,
            iteration=iteration,
            max_iterations=max_iterations,
        )

        raw_text = self.call_gemini(
            prompt=prompt,
            system_instruction=TOOL_SELECTION_SYSTEM_INSTRUCTION,
        )
        return self.validate_and_parse_tool_selection(raw_text)

    def validate_and_parse_tool_selection(self, raw_text: str) -> ToolSelectionDecision:
        """Validate and parse Gemini response into ToolSelectionDecision."""
        cleaned = clean_json_text(raw_text)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMSchemaValidationError(f"Malformed JSON in Gemini tool selection response: {exc}") from exc

        if not isinstance(payload, dict):
            raise LLMSchemaValidationError("Expected JSON object at root of Gemini tool selection response.")

        thought = str(payload.get("thought", "")).strip()
        action = str(payload.get("action", "finish")).strip().lower()
        if action not in ("tool", "finish"):
            action = "finish"

        tool_name = payload.get("tool_name")
        if tool_name:
            tool_name = str(tool_name).strip()
        tool_args = payload.get("tool_args", {})
        if not isinstance(tool_args, dict):
            tool_args = {}

        final_decision = payload.get("final_decision")
        if final_decision:
            final_decision = str(final_decision).strip().upper()

        classification = payload.get("classification")
        if classification:
            classification = str(classification).strip().upper()
            if classification not in {c.value for c in MonitoringClassification}:
                classification = None

        rationale = str(payload.get("rationale", "")).strip()
        try:
            confidence = float(payload.get("confidence", 1.0))
            if not (0.0 <= confidence <= 1.0):
                confidence = 1.0
        except (TypeError, ValueError):
            confidence = 1.0

        return ToolSelectionDecision(
            thought=thought,
            action=action,
            tool_name=tool_name,
            tool_args=tool_args,
            final_decision=final_decision,
            classification=classification,
            rationale=rationale,
            confidence=confidence,
            raw_response=raw_text,
        )


__all__ = [
    "GeminiMonitoringReasoner",
    "MonitoringClassification",
    "MonitoringLLMProposal",
    "ToolSelectionDecision",
    "ToolSelectionStep",
    "build_monitoring_prompt",
    "build_tool_selection_prompt",
]
