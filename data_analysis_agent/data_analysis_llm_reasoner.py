"""LLM reasoning and tool selection module for CareMatrix Data Analysis Agent.

Supports:
1. Genuine, bounded LLM tool-selection loops where Gemini decides which tool to call next
   based on the risk decision event, observation window, and previously computed analytical results.
2. Structured parsing, strict schema validation, and defensive fallback.
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

logger = logging.getLogger("CareMatrix.DataAnalysisLLM")

DATA_ANALYSIS_TOOL_SYSTEM_INSTRUCTION = """You are an autonomous clinical data analysis agent for CareMatrix.
Your goal is to inspect incoming patient risk decisions, patient observation windows, and determine which analytical tools to invoke to produce structured temporal evidence.

Available Tools:
- assess_window_quality: Assess data completeness, missingness ratio, sample sufficiency, and sensor artifacts in window.
- extract_metrics_and_patterns: Extract statistical metrics (mean, min, max, std), window historical changes, and multi-vital patterns (variability, concordance, discordance).
- calculate_trend: Calculate linear slope and classify robust descriptive trend for an individual vital sign series.
- verify_cross_agent_consistency: Verify consistency between Risk Agent predictions and Data Analysis temporal findings, detecting alignment or conflict.

You must select EXACTLY ONE tool to run next, OR decide that analysis is complete ("finish").
DO NOT repeatedly call the same tool unless necessary.

You must respond ONLY with a valid JSON object matching this schema:
{
  "thought": "Clinical reasoning explaining what information is needed next and why",
  "action": "tool" | "finish",
  "tool_name": "assess_window_quality" | "extract_metrics_and_patterns" | "calculate_trend" | "verify_cross_agent_consistency" | null,
  "tool_args": {},
  "rationale": "Summary rationale explaining analytical findings or readiness to finish",
  "confidence": 0.0 to 1.0
}
"""


@dataclass
class DataAnalysisToolDecision:
    """Structured decision returned by Gemini in the data analysis tool-selection loop."""

    thought: str
    action: str  # "tool" | "finish"
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    confidence: float = 1.0
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_data_analysis_tool_prompt(
    case_id: int,
    event_id: str,
    risk_level: str,
    affected_vitals: List[str],
    window_summary: Dict[str, Any],
    executed_tools: List[str],
    state_summary: Dict[str, Any],
    iteration: int,
    max_iterations: int,
) -> str:
    """Build prompt providing risk event context and history for LLM tool selection."""
    executed_desc = ", ".join(executed_tools) if executed_tools else "None"

    return f"""PATIENT DATA ANALYSIS AGENT CYCLE (Iteration {iteration + 1} of max {max_iterations}):
Patient / Case ID: {case_id}
Target Event ID: {event_id}
Upstream Risk Assessment: {risk_level}
Reported Affected Vitals: {affected_vitals}

Observation Window Summary:
- Sample Count: {window_summary.get('sample_count', 0)}
- Duration Seconds: {window_summary.get('duration_seconds', 0)}
- Vitals Available: {window_summary.get('vitals', [])}

Current Analysis State:
- Tools Already Executed: {executed_desc}
- Window Quality Assessed: {state_summary.get('quality_assessed', False)}
- Data Quality Flagged: {state_summary.get('quality_flagged', None)}
- Metrics & Patterns Extracted: {state_summary.get('metrics_extracted', False)}
- Consistency Verified: {state_summary.get('consistency_verified', False)}
- Active Discovered Patterns: {json.dumps(state_summary.get('patterns_glance', []))}
- Active Discovered Trends: {json.dumps(state_summary.get('trends_glance', {}))}

TASK:
Determine what analytical tool should be executed next to evaluate the patient's data,
OR choose "finish" if sufficient analytical evidence has been gathered to produce the DataAnalysisEvent.
Respond strictly in the specified JSON format.
"""


class GeminiDataAnalysisReasoner(BaseGeminiReasoner):
    """LLM reasoner providing dynamic tool selection for DataAnalysisAgent."""

    def select_tool(
        self,
        case_id: int,
        event_id: str,
        risk_level: str,
        affected_vitals: List[str],
        window_summary: Dict[str, Any],
        executed_tools: List[str],
        state_summary: Dict[str, Any],
        iteration: int,
        max_iterations: int = 5,
    ) -> DataAnalysisToolDecision:
        """Ask Gemini which analytical tool to execute next or whether analysis is complete."""
        prompt = build_data_analysis_tool_prompt(
            case_id=case_id,
            event_id=event_id,
            risk_level=risk_level,
            affected_vitals=affected_vitals,
            window_summary=window_summary,
            executed_tools=executed_tools,
            state_summary=state_summary,
            iteration=iteration,
            max_iterations=max_iterations,
        )

        raw_text = self.call_gemini(
            prompt=prompt,
            system_instruction=DATA_ANALYSIS_TOOL_SYSTEM_INSTRUCTION,
        )
        return self.validate_and_parse_tool_selection(raw_text)

    def validate_and_parse_tool_selection(self, raw_text: str) -> DataAnalysisToolDecision:
        """Validate and parse Gemini response into DataAnalysisToolDecision."""
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

        rationale = str(payload.get("rationale", "")).strip()
        try:
            confidence = float(payload.get("confidence", 1.0))
            if not (0.0 <= confidence <= 1.0):
                confidence = 1.0
        except (TypeError, ValueError):
            confidence = 1.0

        return DataAnalysisToolDecision(
            thought=thought,
            action=action,
            tool_name=tool_name,
            tool_args=tool_args,
            rationale=rationale,
            confidence=confidence,
            raw_response=raw_text,
        )


__all__ = [
    "DATA_ANALYSIS_TOOL_SYSTEM_INSTRUCTION",
    "DataAnalysisToolDecision",
    "GeminiDataAnalysisReasoner",
    "build_data_analysis_tool_prompt",
]

