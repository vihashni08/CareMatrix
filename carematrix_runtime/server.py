"""CareMatrix Backend API & Real-Time SSE Server.

Provides RESTful endpoints for system health, multi-patient states,
telemetry history, clinical alert lifecycle management, and dynamic scenario triggering,
plus real-time Server-Sent Events (SSE) streaming to the Clinician Dashboard.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from typing import Any

from flask import Flask, Response, jsonify, render_template_string, request, send_from_directory, stream_with_context
from flask_cors import CORS

from carematrix_runtime.patient_stream import PatientScenario
from carematrix_runtime.runtime import CareMatrixRuntime
from data.adapters.mimic_adapter import MIMICIVAdapter
from data.adapters.vitaldb_adapter import VitalDBAdapter

# Suppress noisy werkzeug logs in production/demo mode
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)


def create_app(runtime: CareMatrixRuntime | None = None) -> Flask:
    """Create and configure the Flask web and API application."""
    static_folder = os.path.join(os.path.dirname(__file__), "static")
    os.makedirs(static_folder, exist_ok=True)

    app = Flask(__name__, static_folder=static_folder)
    CORS(app)

    # Attach or instantiate runtime
    if runtime is None:
        runtime = CareMatrixRuntime(stream_interval_seconds=1.0, verbose=False)
    app.config["RUNTIME"] = runtime

    # Thread-safe client SSE queues
    sse_clients: list[queue.Queue] = []
    sse_lock = threading.RLock()

    def _broadcast_sse(event_type: str, data: Any) -> None:
        """Forward runtime events to all connected SSE clients."""
        payload = json.dumps({"type": event_type, "data": data, "timestamp": time.time()})
        msg = f"event: message\ndata: {payload}\n\n"
        with sse_lock:
            for q in list(sse_clients):
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    pass

    def _runtime_event_listener(topic: str, event: Any) -> None:
        """Listener passed to runtime to broadcast events."""
        event_dict = event if isinstance(event, dict) else (event.to_dict() if hasattr(event, "to_dict") else str(event))
        if topic == "observation":
            _broadcast_sse("vital_tick", event_dict)
        elif topic == "monitoring_alerts":
            _broadcast_sse("monitoring_alert", event_dict)
        elif topic == "risk_predictions":
            _broadcast_sse("risk_prediction", event_dict)
        elif topic == "data_analysis_inputs" or topic == "analytical_evidence":
            _broadcast_sse("data_analysis", event_dict)
        elif topic == "clinical_decisions":
            _broadcast_sse("clinical_reasoning", event_dict)
        elif topic == "care_coordination_events":
            _broadcast_sse("care_coordination", event_dict)
        elif topic == "heartbeats":
            _broadcast_sse("heartbeat", event_dict)

    runtime.add_live_listener(_runtime_event_listener)

    # ------------------------------------------------------------------------
    # Frontend Routes
    # ------------------------------------------------------------------------
    @app.route("/")
    @app.route("/dashboard")
    def dashboard():
        """Serve the Clinician Dashboard Single-Page Application."""
        index_path = os.path.join(static_folder, "index.html")
        if os.path.exists(index_path):
            return send_from_directory(static_folder, "index.html")
        return jsonify({"message": "CareMatrix Dashboard HTML not found in static folder."}), 404

    # ------------------------------------------------------------------------
    # REST API Routes
    # ------------------------------------------------------------------------
    @app.route("/api/health", methods=["GET"])
    def get_health():
        """System health and supervisor status across all 5 agents."""
        rt: CareMatrixRuntime = app.config["RUNTIME"]
        supervisor_snap = rt.supervisor.get_system_health_snapshot()
        active_alerts = rt.alert_manager.get_active_alerts()

        return jsonify({
            "status": "healthy" if supervisor_snap["all_healthy"] else "degraded",
            "all_healthy": supervisor_snap["all_healthy"],
            "total_agents": supervisor_snap["total_agents"],
            "supervisor": supervisor_snap,
            "runtime_metrics": rt.metrics,
            "active_alert_count": len(active_alerts),
            "timestamp": time.time(),
        })

    @app.route("/api/patients", methods=["GET"])
    def list_patients():
        """List summary of all monitored inpatient beds."""
        rt: CareMatrixRuntime = app.config["RUNTIME"]
        patients = rt.state_manager.get_patient_summary_list()
        # Inject data_source for each patient so the dashboard bed strip can badge them
        for p in patients:
            p["data_source"] = rt.get_patient_data_source(p["patient_id"])
        return jsonify({
            "total": len(patients),
            "patients": patients,
            "timestamp": time.time(),
        })

    @app.route("/api/patients/<int:patient_id>", methods=["GET"])
    def get_patient_detail(patient_id: int):
        """Detailed record and 5-agent state for a single bed."""
        rt: CareMatrixRuntime = app.config["RUNTIME"]
        detail = rt.state_manager.get_patient_detail(patient_id)
        if detail is None:
            return jsonify({"error": f"Patient {patient_id} not found"}), 404
        detail["data_source"] = rt.get_patient_data_source(patient_id)
        return jsonify(detail)

    @app.route("/api/patients/<int:patient_id>/vitals", methods=["GET"])
    def get_patient_vitals(patient_id: int):
        """Recent vital sign observations time series for charting."""
        rt: CareMatrixRuntime = app.config["RUNTIME"]
        limit = int(request.args.get("limit", 60))
        history = rt.state_manager.get_patient_vital_history(patient_id)
        return jsonify({
            "patient_id": patient_id,
            "count": len(history[-limit:]),
            "vitals": history[-limit:],
        })

    @app.route("/api/alerts", methods=["GET"])
    def get_alerts():
        """Retrieve active and historical clinical alerts."""
        rt: CareMatrixRuntime = app.config["RUNTIME"]
        pid_raw = request.args.get("patient_id")
        patient_id = int(pid_raw) if pid_raw is not None else None

        active = rt.alert_manager.get_active_alerts(patient_id=patient_id)
        history = rt.alert_manager.get_alert_history(patient_id=patient_id)

        return jsonify({
            "active_count": len(active),
            "history_count": len(history),
            "active": active,
            "history": history,
        })

    @app.route("/api/alerts/<alert_id>/acknowledge", methods=["POST"])
    def acknowledge_alert(alert_id: str):
        """Clinician acknowledgment of an active alert."""
        rt: CareMatrixRuntime = app.config["RUNTIME"]
        data = request.get_json(silent=True) or {}
        clinician_id = data.get("clinician_id", "clinician_on_duty")

        alert = rt.alert_manager.acknowledge_alert(alert_id, clinician_id=clinician_id)
        if alert is None:
            return jsonify({"error": f"Alert '{alert_id}' not found or already resolved"}), 404

        # Record explicit feedback into episodic PatientMemory
        if hasattr(rt, "patient_memory") and rt.patient_memory is not None:
            rt.patient_memory.record_clinician_feedback(
                patient_id=alert.patient_id,
                event_id=alert.event_id,
                action="ACKNOWLEDGE",
                clinician_id=clinician_id,
                care_action_id=alert.care_action_id,
                priority=alert.priority,
            )

        _broadcast_sse("alert_updated", alert.to_dict())
        return jsonify({
            "success": True,
            "alert": alert.to_dict(),
        })

    @app.route("/api/alerts/<alert_id>/resolve", methods=["POST"])
    def resolve_alert(alert_id: str):
        """Manual clinician resolution of an alert."""
        rt: CareMatrixRuntime = app.config["RUNTIME"]
        data = request.get_json(silent=True) or {}
        reason = data.get("reason", "Manually resolved by clinician")
        clinician_id = data.get("clinician_id", "clinician_on_duty")
        is_false_positive = bool(data.get("is_false_positive", False))
        is_override = bool(data.get("is_override", False))

        alert = rt.alert_manager.resolve_alert(alert_id, reason=reason)
        if alert is None:
            return jsonify({"error": f"Alert '{alert_id}' not found or already resolved"}), 404

        # Record explicit feedback into episodic PatientMemory
        if hasattr(rt, "patient_memory") and rt.patient_memory is not None:
            rt.patient_memory.record_clinician_feedback(
                patient_id=alert.patient_id,
                event_id=alert.event_id,
                action="RESOLVE",
                clinician_id=clinician_id,
                reason=reason,
                care_action_id=alert.care_action_id,
                priority=alert.priority,
                is_false_positive=is_false_positive,
                is_override=is_override,
            )

        _broadcast_sse("alert_updated", alert.to_dict())
        return jsonify({
            "success": True,
            "alert": alert.to_dict(),
        })

    @app.route("/api/patients/<int:patient_id>/memory", methods=["GET"])
    def get_patient_memory(patient_id: int):
        """Retrieve episodic memory for a monitored bed."""
        rt: CareMatrixRuntime = app.config["RUNTIME"]
        if not hasattr(rt, "patient_memory") or rt.patient_memory is None:
            return jsonify({"patient_id": patient_id, "count": 0, "episodes": []})
        episodes = rt.patient_memory.get_episodes(patient_id)
        return jsonify({
            "patient_id": patient_id,
            "count": len(episodes),
            "episodes": [ep.to_dict() for ep in episodes],
            "context_prompt": rt.patient_memory.format_memory_for_prompt(patient_id),
        })

    @app.route("/api/scenarios/trigger", methods=["POST"])
    def trigger_scenario():
        """Dynamically switch simulation scenario for a patient."""
        rt: CareMatrixRuntime = app.config["RUNTIME"]
        data = request.get_json(silent=True) or {}

        pid_raw = data.get("patient_id")
        scenario_raw = data.get("scenario")

        if pid_raw is None or scenario_raw is None:
            return jsonify({"error": "patient_id and scenario are required"}), 400

        try:
            patient_id = int(pid_raw)
            # Match enum
            scenario_name = str(scenario_raw).upper()
            scenario_enum = PatientScenario[scenario_name]
        except KeyError:
            valid = [s.name for s in PatientScenario]
            return jsonify({"error": f"Invalid scenario. Valid options: {valid}"}), 400
        except ValueError:
            return jsonify({"error": "patient_id must be an integer"}), 400

        success = rt.trigger_scenario(patient_id, scenario_enum)
        if not success:
            return jsonify({"error": f"Patient {patient_id} not registered in simulator"}), 404

        _broadcast_sse("scenario_changed", {
            "patient_id": patient_id,
            "scenario": scenario_enum.value,
        })

        return jsonify({
            "success": True,
            "patient_id": patient_id,
            "scenario": scenario_enum.value,
        })

    # ------------------------------------------------------------------------
    # Dataset Discovery & Real Patient Loading
    # ------------------------------------------------------------------------
    @app.route("/api/datasets", methods=["GET"])
    def list_datasets():
        """List available case IDs from VitalDB and MIMIC adapters.

        Returns:
            { "vitaldb": [0, 4, ...], "mimic": [1001, 1002, ...] }
        """
        vitaldb_cases = VitalDBAdapter().list_cases()
        mimic_cases = MIMICIVAdapter().list_cases()
        return jsonify({
            "vitaldb": vitaldb_cases,
            "mimic": mimic_cases,
        })

    @app.route("/api/patients/dataset", methods=["POST"])
    def load_dataset_patient():
        """Load a real dataset case into the live monitoring pipeline.

        Request body (JSON):
            { "case_id": <int|str>, "adapter": "vitaldb"|"mimic" }

        Returns:
            Same shape as GET /api/patients/<id> (with data_source field).
        """
        rt: CareMatrixRuntime = app.config["RUNTIME"]
        data = request.get_json(silent=True) or {}

        case_id_raw = data.get("case_id")
        adapter_name = data.get("adapter", "vitaldb")

        if case_id_raw is None:
            return jsonify({"error": "case_id is required"}), 400

        try:
            case_id = int(case_id_raw)
        except (TypeError, ValueError):
            case_id = str(case_id_raw)

        info = rt.add_dataset_patient(case_id=case_id, adapter_name=adapter_name)
        patient_id = info["patient_id"]

        # Return full patient detail (same shape as GET /api/patients/<id>)
        detail = rt.state_manager.get_patient_detail(patient_id) or {}
        detail["data_source"] = info["data_source"]
        detail["case_id"] = info["case_id"]
        detail["adapter"] = info["adapter"]

        _broadcast_sse("patient_added", {
            "patient_id": patient_id,
            "data_source": info["data_source"],
        })

        return jsonify(detail), 201

    # ------------------------------------------------------------------------
    # Real-Time Server-Sent Events (SSE) Stream
    # ------------------------------------------------------------------------
    @app.route("/api/events/stream", methods=["GET"])
    def sse_stream():
        """Push real-time telemetry, alerts, and agent actions to web clients."""
        client_q: queue.Queue = queue.Queue(maxsize=100)
        with sse_lock:
            sse_clients.append(client_q)

        def event_generator():
            # Send initial greeting and state
            rt: CareMatrixRuntime = app.config["RUNTIME"]
            initial_payload = json.dumps({
                "type": "connection_established",
                "data": {
                    "patients": rt.state_manager.get_patient_summary_list(),
                    "active_alerts": rt.alert_manager.get_active_alerts(),
                    "health": rt.supervisor.get_system_health_snapshot(),
                },
                "timestamp": time.time(),
            })
            yield f"event: message\ndata: {initial_payload}\n\n"

            try:
                while True:
                    try:
                        msg = client_q.get(timeout=10.0)
                        yield msg
                    except queue.Empty:
                        # Keep-alive ping comment
                        yield ": keepalive\n\n"
            except GeneratorExit:
                pass
            finally:
                with sse_lock:
                    if client_q in sse_clients:
                        sse_clients.remove(client_q)

        response = Response(stream_with_context(event_generator()), mimetype="text/event-stream")
        response.headers["Cache-Control"] = "no-cache"
        response.headers["X-Accel-Buffering"] = "no"
        response.headers["Connection"] = "keep-alive"
        return response

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 5050,
    runtime: CareMatrixRuntime | None = None,
    stream_interval: float = 1.0,
    enable_llm: bool = False,
    verbose: bool = True,
) -> None:
    """Launch the CareMatrix Runtime and Flask Web Server."""
    if runtime is None:
        runtime = CareMatrixRuntime(
            stream_interval_seconds=stream_interval,
            enable_llm_reasoning=enable_llm,
            verbose=verbose,
        )

    # Start multi-agent background workers and simulation stream
    runtime.start()

    app = create_app(runtime=runtime)
    print(f"\n======================================================================")
    print(f"CAREMATRIX CLINICIAN DASHBOARD ONLINE")
    print(f"Web Dashboard: http://localhost:{port}/")
    print(f"REST Health:   http://localhost:{port}/api/health")
    print(f"SSE Stream:    http://localhost:{port}/api/events/stream")
    print(f"======================================================================\n")

    try:
        app.run(host=host, port=port, threaded=True)
    finally:
        runtime.stop()


if __name__ == "__main__":
    run_server(port=5050, stream_interval=1.0)
