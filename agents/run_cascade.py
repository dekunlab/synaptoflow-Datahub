"""
agents/run_cascade.py

The autonomous background orchestrator. 
Reads open incidents and fires the Diagnostic and Calibration agents 
to pre-stage drafts before the clinician opens the Cockpit.
"""

import json
import subprocess
from pathlib import Path

# Set up paths relative to this script
REPO_ROOT = Path(__file__).parent.parent
INCIDENT_STATE_FILE = REPO_ROOT / "monitor" / "incident_state.json"

def main():
    print("=== SynaptoFlow Agent Cascade ===")
    
    # 1. Check if there are any incidents
    if not INCIDENT_STATE_FILE.exists():
        print("No incident_state.json found. No patients currently drifting.")
        return

    with open(INCIDENT_STATE_FILE, "r") as f:
        try:
            incident_state = json.load(f)
        except json.JSONDecodeError:
            print("Error reading incident_state.json. File might be empty.")
            return

    if not incident_state:
        print("No open incidents. System nominal.")
        return

    print(f"Found {len(incident_state)} open incident(s). Starting background cascade...\n")

    # 2. Loop through each patient and run the agents
    for patient_id, incident_urn in incident_state.items():
        print(f"--- Processing {patient_id} ---")

        # Run Diagnostic Agent
        print(f"[*] Waking Diagnostic Agent...")
        diag_res = subprocess.run(["python3", "agents/diagnostic_agent.py", patient_id])
        
        if diag_res.returncode != 0:
            print(f"[!] FAILED for {patient_id} during Diagnosis. Skipping to next patient.\n")
            continue

        # Run Calibration Agent
        print(f"[*] Waking Calibration Agent...")
        cal_res = subprocess.run(["python3", "agents/calibration_agent.py", patient_id])
        
        if cal_res.returncode != 0:
            print(f"[!] FAILED for {patient_id} during Calibration. Skipping to next patient.\n")
            continue
        
        print(f"[✓] Drafts successfully staged for {patient_id}.\n")

    print("=== Cascade complete. All data staged for Cockpit review. ===")

if __name__ == "__main__":
    main()