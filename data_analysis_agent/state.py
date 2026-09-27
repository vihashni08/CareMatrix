"""State tracking for the Data Analysis Agent."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PatientAnalysisState:
    """Maintains historical trend and analysis context across patient assessments."""

    last_analysed_timestamp: float | None = None
    previous_trends: dict[str, str] = field(default_factory=dict)
    previous_data_quality_flag: bool | None = None
    previous_patterns: list[str] = field(default_factory=list)
    analysed_events: int = 0


__all__ = ["PatientAnalysisState"]

