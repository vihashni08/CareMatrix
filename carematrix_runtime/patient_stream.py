"""Continuous Patient Data Stream Simulator for CareMatrix.

Simulates realistic, timestamped physiological observations for multiple patients
across clinically relevant scenarios (Stable, Gradual Deterioration, Sudden Shock,
Recovery, Noisy Sensor, Missing Data, and Persistent Abnormality).
"""

from __future__ import annotations

import math
import random
import time
from enum import Enum
from typing import Any, Generator


class PatientScenario(str, Enum):
    STABLE = "STABLE"
    GRADUAL_DETERIORATION = "GRADUAL_DETERIORATION"
    SUDDEN_ABNORMALITY = "SUDDEN_ABNORMALITY"
    RECOVERY = "RECOVERY"
    NOISY_SENSOR = "NOISY_SENSOR"
    MISSING_DATA = "MISSING_DATA"
    PERSISTENT_ABNORMALITY = "PERSISTENT_ABNORMALITY"


class PatientProfile:
    """Configurable baseline physiology and scenario state for a single patient."""

    def __init__(
        self,
        patient_id: int,
        name: str | None = None,
        scenario: PatientScenario = PatientScenario.STABLE,
        base_hr: float = 72.0,
        base_map: float = 85.0,
        base_spo2: float = 98.0,
        base_rr: float = 14.0,
        base_bt: float = 36.8,
    ):
        self.patient_id = patient_id
        self.name = name or f"Patient {patient_id}"
        self.scenario = scenario
        self.base_hr = base_hr
        self.base_map = base_map
        self.base_spo2 = base_spo2
        self.base_rr = base_rr
        self.base_bt = base_bt

        self.current_t: float = 0.0
        self.scenario_start_t: float = 0.0
        self.scenario_step: int = 0

    def set_scenario(self, scenario: PatientScenario | str) -> None:
        """Switch current clinical scenario dynamically."""
        if isinstance(scenario, str):
            scenario = PatientScenario(scenario.upper())
        self.scenario = scenario
        self.scenario_start_t = self.current_t
        self.scenario_step = 0


class PatientStreamSimulator:
    """Simulates real-time physiological telemetry for one or more inpatient ICU/floor beds."""

    def __init__(
        self,
        profiles: list[PatientProfile] | None = None,
        seed: int | None = 42,
    ):
        self._seed = seed
        self._rng = random.Random(seed) if seed is not None else random.Random()
        if profiles is None:
            self.patients: dict[int, PatientProfile] = {
                101: PatientProfile(101, name="Bed 101 (Post-Op General)", scenario=PatientScenario.STABLE),
                102: PatientProfile(102, name="Bed 102 (ICU Sepsis Surveillance)", scenario=PatientScenario.GRADUAL_DETERIORATION),
                103: PatientProfile(103, name="Bed 103 (Telemetry Floor Cardiac)", scenario=PatientScenario.STABLE),
            }
        else:
            self.patients = {p.patient_id: p for p in profiles}

    def get_patient(self, patient_id: int) -> PatientProfile | None:
        return self.patients.get(patient_id)

    def set_scenario(self, patient_id: int, scenario: PatientScenario | str) -> bool:
        p = self.get_patient(patient_id)
        if p is not None:
            p.set_scenario(scenario)
            return True
        return False

    def generate_observation(self, patient_id: int, dt: float = 1.0) -> dict[str, Any]:
        """Generate one timestamped observation dictionary for the given patient."""
        p = self.patients[patient_id]
        p.current_t += dt
        p.scenario_step += 1
        t_in_scenario = p.scenario_step * dt

        # Random micro-variations using local deterministic RNG
        hr_noise = self._rng.gauss(0, 0.8)
        map_noise = self._rng.gauss(0, 1.2)
        spo2_noise = self._rng.gauss(0, 0.3)
        rr_noise = self._rng.gauss(0, 0.4)
        bt_noise = self._rng.gauss(0, 0.02)

        hr = p.base_hr + hr_noise
        map_val = p.base_map + map_noise
        spo2 = min(100.0, p.base_spo2 + spo2_noise)
        rr = p.base_rr + rr_noise
        bt = p.base_bt + bt_noise

        # Apply Scenario Dynamics
        if p.scenario == PatientScenario.STABLE:
            # Vitals remain normal and stable
            pass

        elif p.scenario == PatientScenario.GRADUAL_DETERIORATION:
            # Progressive tachycardia + hypotension + tachypnea over 40 seconds
            prog = min(1.0, t_in_scenario / 40.0)
            hr += prog * 45.0          # HR climbs towards ~118
            map_val -= prog * 30.0     # MAP drops towards ~55
            spo2 -= prog * 6.0         # SpO2 drops towards ~92
            rr += prog * 12.0          # RR climbs towards ~26

        elif p.scenario == PatientScenario.SUDDEN_ABNORMALITY:
            # Sudden acute hypotension & severe desaturation within 5 seconds
            if t_in_scenario >= 5.0:
                hr += 40.0
                map_val -= 35.0
                spo2 -= 10.0
                rr += 10.0

        elif p.scenario == PatientScenario.RECOVERY:
            # Values smoothly return back to normal baseline
            prog = min(1.0, t_in_scenario / 25.0)
            # Interpolate from abnormal back to base
            hr = (p.base_hr + 40.0) * (1.0 - prog) + (p.base_hr) * prog + hr_noise
            map_val = (p.base_map - 25.0) * (1.0 - prog) + (p.base_map) * prog + map_noise
            spo2 = min(100.0, (p.base_spo2 - 6.0) * (1.0 - prog) + (p.base_spo2) * prog + spo2_noise)
            if prog >= 1.0:
                p.scenario = PatientScenario.STABLE

        elif p.scenario == PatientScenario.NOISY_SENSOR:
            # High variance sensor artifact spikes / intermittent motion noise
            if self._rng.random() < 0.4:
                hr += self._rng.choice([-35.0, 50.0])
                spo2 = max(70.0, spo2 - self._rng.uniform(15.0, 25.0))
            if self._rng.random() < 0.2:
                map_val += self._rng.choice([-40.0, 45.0])

        elif p.scenario == PatientScenario.MISSING_DATA:
            # Intermittent sensor disconnect or unacquired leads (None / NaN)
            if p.scenario_step % 2 == 0:
                map_val = None  # NIBP cuff cycle / disconnect
            if p.scenario_step % 3 == 0:
                spo2 = None     # Finger probe dislodged
            bt = None           # Temp probe unavailable

        elif p.scenario == PatientScenario.PERSISTENT_ABNORMALITY:
            # Sustained hemodynamic instability
            hr = p.base_hr + 48.0 + hr_noise
            map_val = p.base_map - 32.0 + map_noise
            spo2 = p.base_spo2 - 7.0 + spo2_noise
            rr = p.base_rr + 12.0 + rr_noise

        # Derived SBP and DBP approximations when MAP is numeric
        sbp = (map_val * 1.35) if map_val is not None else None
        dbp = (map_val * 0.75) if map_val is not None else None

        return {
            "Time": round(p.current_t, 2),
            "timestamp": round(p.current_t, 2),
            "patient_id": p.patient_id,
            "case_id": p.patient_id,
            "HR": round(hr, 1) if hr is not None else None,
            "MAP": round(map_val, 1) if map_val is not None else None,
            "SpO2": round(min(100.0, max(0.0, spo2)), 1) if spo2 is not None else None,
            "RR": round(max(0.0, rr), 1) if rr is not None else None,
            "BT": round(bt, 2) if bt is not None else None,
            "SBP": round(sbp, 1) if sbp is not None else None,
            "DBP": round(dbp, 1) if dbp is not None else None,
            "scenario": p.scenario.value,
        }

    def stream(
        self,
        patient_id: int,
        max_samples: int | None = None,
        delay_seconds: float = 0.0,
    ) -> Generator[dict[str, Any], None, None]:
        """Generate a continuous sequence of observations for a patient."""
        count = 0
        while max_samples is None or count < max_samples:
            obs = self.generate_observation(patient_id)
            yield obs
            count += 1
            if delay_seconds > 0:
                time.sleep(delay_seconds)
