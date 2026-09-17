"""Central prototype configuration for the CareMatrix Monitoring Agent.

These engineering parameters are configurable research defaults. They are not
clinically validated thresholds or diagnostic criteria.
"""

BASELINE_WINDOW_SECONDS = 60
MIN_BASELINE_SAMPLES = 10

DEVIATION_THRESHOLDS = {
    "HR": 0.20,
    "MAP": 0.20,
    "SpO2": 0.05,
    "RR": 0.40,
}
PERSISTENCE_SECONDS = 10

# ---------------------------------------------------------------------------
# Severity classification (configurable engineering defaults)
# ---------------------------------------------------------------------------
# Multipliers applied to the per-vital DEVIATION_THRESHOLDS to form severity
# tier boundaries.  For example HR base = 0.20: mild >= 0.20, moderate >= 0.30,
# severe >= 0.40, critical >= 0.60.
SEVERITY_MULTIPLIERS = {
    "mild": 1.0,
    "moderate": 1.5,
    "severe": 2.0,
    "critical": 3.0,
}

# Persistence duration (seconds) required before an alert is raised at each
# severity level.  Higher severity requires less persistence to trigger quickly.
SEVERITY_PERSISTENCE = {
    "mild": 10,
    "moderate": 8,
    "severe": 5,
    "critical": 5,
}

# Minimum number of simultaneously deviating vitals required per severity.
# Severe and critical single-vital deviations can trigger independently;
# mild and moderate still require corroboration from a second vital.
SEVERITY_MIN_VITALS = {
    "mild": 2,
    "moderate": 2,
    "severe": 1,
    "critical": 1,
}

# Trend classification uses robust medians from the first and final thirds of a
# recent window, rather than reacting to one or two noisy readings.
# A short recent window makes the descriptive trend reflect the current portion
# of a physiological event (including recovery), rather than pre-event values.
TREND_WINDOW_SECONDS = 10
TREND_MIN_POINTS = 5
TREND_MIN_RELATIVE_CHANGE = 0.03

# Signal-quality settings identify isolated, abrupt spikes without removing the
# original readings. They are deliberately conservative prototype settings.
SIGNAL_QUALITY_WINDOW_SECONDS = 10
SIGNAL_QUALITY_MIN_VALID_POINTS = 5
ARTIFACT_SPIKE_THRESHOLD = 0.50

# Event-based lifecycle deduplication prevents duplicate alerts while one event
# is active. No global refractory period is needed: a recovered new event must
# only satisfy the normal persistence requirement again.
ALERT_COOLDOWN_SECONDS = 0

# Recovery is confirmed only after the multi-vital deviation condition remains
# absent for this many one-second samples. This avoids closing an event on one
# briefly improving observation.
RECOVERY_DURATION_SECONDS = 5
