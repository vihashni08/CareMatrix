"""LLM reasoning and cross-examination module for CareMatrix Risk Agent.

Enables Gemini-powered physiological risk evaluation and cross-agent challenge resolution.
When conflicting evidence is raised by other agents (such as DataAnalysisAgent detecting
stable trends despite high model risk), this reasoner re-evaluates the assessment
in light of the challenge to confirm (AGREE) or revise (PROPOSE).
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

RISK_SYSTEM_INSTRUCTION = """You are a senior clinical risk modeling specialist and safety arbiter for CareMatrix.
Your task is to analyze patient vital sign features, baseline deviations, and machine learning risk predictions.
You must also adjudicate cross-agent challenges when temporal trend findings conflict with model risk scores.

For challenge resolution:
- If model HIGH RISK is challenged by flat/stable trends, evaluate whether baseline severity or non-linear multi-vital combinations still justify HIGH RISK (AGREE), or whether the model is overcalling due to transient noise (PROPOSE revision to LOW RISK).
- If model LOW RISK is challenged by rapid deterioration or simultaneous worsening vitals, evaluate whether urgent safety requires conceding to the temporal evidence (PROPOSE revision to HIGH RISK) or if the vitals remain within compensated safe boundaries (AGREE).

You must respond strictly with a valid JSON object matching the requested schema.
"""


class RiskChallengeDecision(str, Enum):
    AGREE = "AGREE"
    PROPOSE = "PROPOSE"


@dataclass
class RiskChallengeResponse:
    """Structured proposal for resolving cross-agent challenges."""

    decision: str  # "AGREE" or "PROPOSE"
    revised_risk_level: str  # "HIGH RISK" or "LOW RISK"
    rationale: str
    confidence: float
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RiskLLMProposal:
    """Structured proposal for independent risk assessment."""

    risk_level: str  # "HIGH RISK" or "LOW RISK"
    contributing_vitals: list[str]
    trend_consistency: str  # "CONSISTENT", "CONFLICTING", "UNCERTAIN"
    rationale: str
    confidence: float
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_challenge_prompt(
    features: dict[str, float],
    model_prob: float,
    original_decision: str,
    challenge_evidence: list[str],
    conflict_flags: list[str],
) -> str:
    """Format cross-agent challenge prompt for Gemini arbitration."""
    feat_summary = "\n".join(f"- {k}: {v}" for k, v in sorted(features.items()) if not k.startswith("_"))
    ev_summary = "\n".join(f"- {ev}" for ev in challenge_evidence)
    flags = ", ".join(conflict_flags) if conflict_flags else "NONE"

    return f"""PATIENT RISK ASSESSMENT CROSS-EXAMINATION:
Original Risk Assessment: {original_decision}
Model Output Probability: {model_prob:.3f}

Patient Physiological Features:
{feat_summary or 'No raw features recorded'}

CHALLENGE RAISED BY DATA ANALYSIS AGENT:
Conflict Flags: {flags}
Conflicting Evidence:
{ev_summary or 'No specific narrative evidence provided'}

CLINICAL TASK:
Evaluate whether the challenging evidence from temporal trends warrants revising the risk assessment or if the original risk assessment should be confirmed (AGREE).
- If confirming (AGREE), provide clinical justification for why the baseline risk holds despite the conflicting temporal trends.
- If revising (PROPOSE), specify the revised risk level ("HIGH RISK" or "LOW RISK") and explain what changed your assessment.

Return strictly a JSON object matching this schema:
{{
  "decision": "AGREE" | "PROPOSE",
  "revised_risk_level": "HIGH RISK" | "LOW RISK",
  "rationale": "Clear, concise 1-2 sentence clinical justification.",
  "confidence": 0.0 to 1.0
}}"""


class GeminiRiskReasoner(BaseGeminiReasoner):
    """Gemini-powered reasoner for clinical risk evaluation and challenge resolution."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        timeout: float = 8.0,
    ):
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            timeout=timeout,
        )

    def re_evaluate_challenge(
        self,
        features: dict[str, float],
        model_prob: float,
        original_decision: str,
        challenge_evidence: list[str],
        conflict_flags: list[str],
    ) -> RiskChallengeResponse:
        """Re-evaluate risk in light of challenging evidence and return structured response."""
        prompt = build_challenge_prompt(
            features=features,
            model_prob=model_prob,
            original_decision=original_decision,
            challenge_evidence=challenge_evidence,
            conflict_flags=conflict_flags,
        )

        raw_text = self.call_gemini(prompt, system_instruction=RISK_SYSTEM_INSTRUCTION)
        cleaned = clean_json_text(raw_text)

        try:
            data = json.loads(cleaned)
        except Exception as exc:
            raise LLMSchemaValidationError(f"Invalid JSON from Gemini: {exc}") from exc

        if not isinstance(data, dict):
            raise LLMSchemaValidationError(f"Expected JSON object, got {type(data)}")

        decision = str(data.get("decision", "")).strip().upper()
        if decision not in ("AGREE", "PROPOSE"):
            raise LLMSchemaValidationError(f"Invalid decision '{decision}', expected AGREE or PROPOSE")

        revised_level = str(data.get("revised_risk_level", "")).strip().upper()
        if revised_level not in ("HIGH RISK", "LOW RISK", "HIGH_RISK", "LOW_RISK"):
            revised_level = "HIGH RISK" if "HIGH" in revised_level else "LOW RISK"
        revised_level = "HIGH RISK" if "HIGH" in revised_level else "LOW RISK"

        rationale = str(data.get("rationale", "")).strip()
        if not rationale:
            raise LLMSchemaValidationError("Missing required 'rationale' field")

        try:
            confidence = float(data.get("confidence", 0.8))
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.8

        return RiskChallengeResponse(
            decision=decision,
            revised_risk_level=revised_level,
            rationale=rationale,
            confidence=confidence,
            raw_response=raw_text,
        )

    def propose(
        self,
        features: dict[str, float],
        model_prob: float,
        threshold: float,
    ) -> RiskLLMProposal:
        """Independently reason about feature contributions and overall risk."""
        feat_str = "\n".join(f"- {k}: {v}" for k, v in sorted(features.items()) if not k.startswith("_"))
        prompt = f"""EVALUATE PHYSIOLOGICAL RISK FEATURES:
Model Probability: {model_prob:.3f} (Threshold: {threshold:.3f})

Feature Values:
{feat_str}

Evaluate whether this profile warrants HIGH RISK or LOW RISK.
Return strictly a JSON object:
{{
  "risk_level": "HIGH RISK" | "LOW RISK",
  "contributing_vitals": ["HR", ...],
  "trend_consistency": "CONSISTENT" | "CONFLICTING" | "UNCERTAIN",
  "rationale": "1-2 sentence clinical explanation.",
  "confidence": 0.0 to 1.0
}}"""
        raw_text = self.call_gemini(prompt, system_instruction=RISK_SYSTEM_INSTRUCTION)
        cleaned = clean_json_text(raw_text)

        try:
            data = json.loads(cleaned)
        except Exception as exc:
            raise LLMSchemaValidationError(f"Invalid JSON: {exc}") from exc

        level = "HIGH RISK" if "HIGH" in str(data.get("risk_level", "")).upper() else "LOW RISK"
        return RiskLLMProposal(
            risk_level=level,
            contributing_vitals=list(data.get("contributing_vitals", [])),
            trend_consistency=str(data.get("trend_consistency", "CONSISTENT")),
            rationale=str(data.get("rationale", "")),
            confidence=float(data.get("confidence", 0.8)),
            raw_response=raw_text,
        )


__all__ = [
    "GeminiRiskReasoner",
    "RiskChallengeDecision",
    "RiskChallengeResponse",
    "RiskLLMProposal",
    "build_challenge_prompt",
]
