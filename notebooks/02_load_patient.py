"""Load and inspect one suitable VitalDB patient case."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitoring_agent.monitoring_agent import find_suitable_cases, load_case_data


def main() -> None:
    try:
        case_id = find_suitable_cases(limit=1)[0]
        df = load_case_data(case_id)
        print(f"Case ID: {case_id}")
        print(f"Shape: {df.shape}")
        print(df.head())
    except (RuntimeError, ValueError) as exc:
        print(f"Patient loading failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
