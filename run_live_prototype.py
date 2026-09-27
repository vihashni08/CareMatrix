#!/usr/bin/env python3
"""CareMatrix Live Autonomous System & Clinician Dashboard Prototype.

Launches the continuous 5-agent runtime with real-time patient telemetry simulator
and the interactive web dashboard on http://localhost:5050.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser

from carematrix_runtime.runtime import CareMatrixRuntime
from carematrix_runtime.server import run_server


def main():
    parser = argparse.ArgumentParser(
        description="Launch CareMatrix Continuous 5-Agent Runtime & Clinician Dashboard."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5050,
        help="Port to run the web server on (default: 5050)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host interface to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Telemetry simulation stream tick interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--enable-llm",
        action="store_true",
        help="Enable Gemini LLM for live clinical reasoning synthesis (default: deterministic mode)",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Automatically open the web dashboard in your default browser",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed agent terminal logs",
    )
    args = parser.parse_args()

    url = f"http://localhost:{args.port}/"

    print("=" * 72)
    print("CAREMATRIX — INTELLIGENT HEALTHCARE DECISION SUPPORT SYSTEM")
    print("Continuous 5-Agent Autonomous Runtime & Clinician Dashboard")
    print("=" * 72)
    print(f"• Active Pipeline    : Monitoring → Risk → Data Analysis → Clinical Reasoning → Care Coordination")
    print(f"• Supervisor Status  : Active (Continuous heartbeat tracking & recovery)")
    print(f"• Simulation Stream  : Active (Beds 101, 102, 103, interval={args.interval}s)")
    print(f"• Reasoning Engine   : {'Gemini 2.5 Flash + Medical RAG' if args.enable_llm else 'Deterministic Clinical Rules + RAG'}")
    print(f"• Clinician Dashboard: {url}")
    print("=" * 72)
    print("\nPress Ctrl+C to safely terminate the runtime and agents.\n")

    if args.open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    runtime = CareMatrixRuntime(
        stream_interval_seconds=args.interval,
        enable_llm_reasoning=args.enable_llm,
        verbose=not args.quiet,
    )

    run_server(
        host=args.host,
        port=args.port,
        runtime=runtime,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
