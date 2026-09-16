"""Quick connection and track-availability check for the VitalDB API."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitoring_agent.monitoring_agent import find_suitable_cases


def main() -> None:
    try:
        case_ids = find_suitable_cases(limit=5)
        print("VitalDB connection successful.")
        print("First matching case IDs:", case_ids)
    except RuntimeError as exc:
        print(f"VitalDB test failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
