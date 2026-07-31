"""
agents/run_cascade.py

The autonomous background orchestrator.
Watches state/incident_state.json and fires the Diagnostic and Calibration
agents for any incident that doesn't have staged drafts yet, so a clinician
never has to remember to re-run this by hand after monitor.py raises a new
incident.
"""
import json
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
STATE_DIR = REPO_ROOT / "state"
INCIDENT_STATE_FILE = STATE_DIR / "incident_state.json"
POLL_INTERVAL_SECONDS = 5


def already_staged(patient_id: str) -> bool:
    diag_path = STATE_DIR / f"diagnostic_draft_{patient_id}.txt"
    calib_path = STATE_DIR / f"calibration_draft_{patient_id}.json"
    return diag_path.exists() and calib_path.exists()


def load_incident_state() -> dict:
    if not INCIDENT_STATE_FILE.exists():
        return {}
    try:
        return json.loads(INCIDENT_STATE_FILE.read_text())
    except json.JSONDecodeError:
        print("Error reading incident_state.json. File might be empty or mid-write.")
        return {}


def stage_patient(patient_id: str) -> bool:
    diag_path = STATE_DIR / f"diagnostic_draft_{patient_id}.txt"
    calib_path = STATE_DIR / f"calibration_draft_{patient_id}.json"

    print(f"--- Processing {patient_id} ---")

    if not diag_path.exists():
        print("[*] Waking Diagnostic Agent...")
        diag_res = subprocess.run(
            ["python3", "agents/diagnostic_agent.py", patient_id], capture_output=True, text=True
        )
        if diag_res.returncode != 0:
            print(f"[!] FAILED for {patient_id} during Diagnosis. Skipping to next patient.")
            print(diag_res.stdout)
            print(diag_res.stderr)
            print()
            return False
        print(f"[\u2713] Diagnosis staged for {patient_id}.")

    if not calib_path.exists():
        print("[*] Waking Calibration Agent...")
        cal_res = subprocess.run(
            ["python3", "agents/calibration_agent.py", patient_id], capture_output=True, text=True
        )
        if cal_res.returncode != 0:
            print(f"[!] FAILED for {patient_id} during Calibration. Skipping to next patient.")
            print(cal_res.stdout)
            print(cal_res.stderr)
            print()
            return False
        print(f"[\u2713] Calibration staged for {patient_id}.")

    print(f"[\u2713] Drafts staged for {patient_id}.\n")
    return True


def run_once() -> None:
    incident_state = load_incident_state()
    if not incident_state:
        return

    pending = [pid for pid in incident_state if not already_staged(pid)]
    if not pending:
        return

    print(f"Found {len(pending)} incident(s) needing drafts. Staging...\n")
    for patient_id in pending:
        stage_patient(patient_id)


def main() -> None:
    print("=== SynaptoFlow Agent Cascade (watching) ===")
    print(f"Polling {INCIDENT_STATE_FILE} every {POLL_INTERVAL_SECONDS}s. Press Ctrl+C to stop.\n")
    while True:
        run_once()
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()