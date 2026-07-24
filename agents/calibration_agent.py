"""
agents/calibration_agent.py

Calibration Agent for SynaptoFlow.

Drafts a recalibration proposal for a patient with an open drift Incident. 
For each of the 8 channels, it runs a recursive least squares (RLS) fit -- 
processed trial-by-trial -- of the channel's raw `channel_N_feature` readings 
against the known `true_angle_deg` over a fresh calibration window. This recovers 
the channel's current (drifted) preferred direction. The result is compared 
against the currently deployed direction to calculate drift, and blended into 
a proposed new calibration at a default 100% recalibration strength.

Note: This script only computes and stages a draft proposal to 
`monitor/calibration_draft_<patient_id>.json`. It does not deploy or write 
to DataHub. Deployment is handled separately after clinical review.

--- The Regression Model ---
The generative model for one channel is:

    feature = baseline_rate + amplitude * cos(true_angle - preferred_dir) + noise

Using cos(a - b) = cos(a)cos(b) + sin(a)sin(b), that's linear in three unknowns:

    feature = c0 + c1*cos(true_angle) + c2*sin(true_angle) + noise
    where c1 = amplitude*cos(preferred_dir), c2 = amplitude*sin(preferred_dir)

Fitting [1, cos(true_angle), sin(true_angle)] -> feature recovers 
preferred_dir = atan2(c2, c1) directly. The fit recovers its own baseline (c0) 
and amplitude (hypot(c1, c2)) from the fresh window alone.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
TELEMETRY_PATH = REPO_ROOT / "sim" / "output" / "telemetry.csv"
CALIB_PARAMS_PATH = REPO_ROOT / "sim" / "output" / "calibration_params.json"
DRAFT_OUT_DIR = REPO_ROOT / "monitor"

N_CHANNELS = 8
CALIBRATION_WINDOW_TRIALS = 60  # matches simulator's own CALIBRATION_TRIALS
DEFAULT_RECAL_STRENGTH_PCT = 100.0
LOW_CONFIDENCE_R2_THRESHOLD = 0.5


def load_current_calibration(patient_id: str) -> np.ndarray:
    """Loads this patient's currently-deployed per-channel preferred directions (degrees)."""
    all_params = json.loads(CALIB_PARAMS_PATH.read_text())
    if patient_id not in all_params:
        raise KeyError(f"{patient_id} not found in {CALIB_PARAMS_PATH}")
    return np.array(all_params[patient_id], dtype=float)


def load_calibration_window(patient_id: str, n_trials: int = CALIBRATION_WINDOW_TRIALS) -> pd.DataFrame:
    """
    Reads the telemetry.csv and returns the most recent n_trials rows
    for this patient in ascending trial order.
    """
    df = pd.read_csv(TELEMETRY_PATH)
    pdf = df[df.patient_id == patient_id].sort_values("trial").reset_index(drop=True)
    if len(pdf) < n_trials:
        raise ValueError(
            f"Only {len(pdf)} trials available for {patient_id}, need {n_trials} for a calibration window."
        )
    return pdf.tail(n_trials).reset_index(drop=True)


def shortest_angle_diff_deg(a_deg, b_deg):
    """Signed shortest-path difference a - b, wrapped to (-180, 180]. Works elementwise on arrays."""
    return (a_deg - b_deg + 180) % 360 - 180


def rls_fit_channel(true_angle_deg: np.ndarray, channel_feature: np.ndarray, forgetting_factor: float = 1.0):
    """
    Recursive least squares, processed trial-by-trial in arrival order
    (not a single batch np.linalg.lstsq call), fitting:

        feature_t = c0 + c1*cos(true_angle_t) + c2*sin(true_angle_t) + noise_t

    forgetting_factor=1.0 means no exponential forgetting -- appropriate
    here since we already truncate to a recent window rather than fitting
    the whole session; pass <1.0 (e.g. 0.98) to also weight the most recent
    trials within the window more heavily, if that's ever wanted.

    Returns (preferred_direction_deg, amplitude, baseline, r_squared).
    """
    n = len(true_angle_deg)
    angle_rad = np.deg2rad(true_angle_deg)
    X = np.column_stack([np.ones(n), np.cos(angle_rad), np.sin(angle_rad)])
    y = channel_feature

    theta = np.zeros(3)
    P = np.eye(3) * 1000.0  # large initial uncertainty, standard RLS init

    for t in range(n):
        x_t = X[t]
        y_t = y[t]
        Px = P @ x_t
        denom = forgetting_factor + x_t @ Px
        k_t = Px / denom
        error_t = y_t - x_t @ theta
        theta = theta + k_t * error_t
        P = (P - np.outer(k_t, Px)) / forgetting_factor

    c0, c1, c2 = theta
    amplitude = float(np.hypot(c1, c2))
    preferred_direction_deg = float(np.rad2deg(np.arctan2(c2, c1)) % 360)

    y_pred = X @ theta
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return preferred_direction_deg, amplitude, float(c0), r_squared


def blend_directions(current_deg: np.ndarray, refit_deg: np.ndarray, strength_pct: float) -> np.ndarray:
    """
    Blends currently-deployed and freshly-refit per-channel directions by
    recalibration strength (0-100%). 0% = keep the current decoder untouched. 
    100% = fully adopt the refit. Blends along the shortest path around the circle.
    """
    strength = np.clip(strength_pct, 0.0, 100.0) / 100.0
    delta = shortest_angle_diff_deg(refit_deg, current_deg)
    return (current_deg + strength * delta) % 360


def draft_calibration_proposal(patient_id: str, window_trials: int = CALIBRATION_WINDOW_TRIALS) -> dict:
    """
    Builds the full staged proposal: per-channel refit directions, drift
    vs. the currently-deployed calibration, fit quality (R^2), and the
    default (100%-strength) proposed new calibration. Nothing is written to
    DataHub -- this is a draft dict only, for the cockpit to show/edit and
    for the Deployment Agent to eventually write.
    """
    current = load_current_calibration(patient_id)
    window = load_calibration_window(patient_id, window_trials)

    refit_directions = np.zeros(N_CHANNELS)
    amplitudes = np.zeros(N_CHANNELS)
    r_squared = np.zeros(N_CHANNELS)

    true_angles = window["true_angle_deg"].to_numpy()
    for ch in range(N_CHANNELS):
        col = f"channel_{ch}_feature"
        direction, amp, _baseline, r2 = rls_fit_channel(true_angles, window[col].to_numpy())
        refit_directions[ch] = direction
        amplitudes[ch] = amp
        r_squared[ch] = r2

    drift_vs_current = shortest_angle_diff_deg(refit_directions, current)
    proposed = blend_directions(current, refit_directions, DEFAULT_RECAL_STRENGTH_PCT)

    low_confidence_channels = [
        ch for ch in range(N_CHANNELS) if r_squared[ch] < LOW_CONFIDENCE_R2_THRESHOLD
    ]

    return {
        "patient_id": patient_id,
        "window_trials": window_trials,
        "window_trial_range": [int(window["trial"].min()), int(window["trial"].max())],
        "channels": [
            {
                "channel": ch,
                "current_deployed_direction_deg": round(float(current[ch]), 2),
                "refit_direction_deg": round(float(refit_directions[ch]), 2),
                "drift_deg": round(float(drift_vs_current[ch]), 2),
                "fit_amplitude": round(float(amplitudes[ch]), 3),
                "fit_r_squared": round(float(r_squared[ch]), 3),
                "proposed_direction_deg_default_strength": round(float(proposed[ch]), 2),
            }
            for ch in range(N_CHANNELS)
        ],
        "mean_abs_drift_deg": round(float(np.mean(np.abs(drift_vs_current))), 2),
        "max_abs_drift_deg": round(float(np.max(np.abs(drift_vs_current))), 2),
        "default_recal_strength_pct": DEFAULT_RECAL_STRENGTH_PCT,
        "low_confidence_channels": low_confidence_channels,
    }


def save_draft(proposal: dict) -> Path:
    """Stages the proposal to a local JSON file for the cockpit to read."""
    DRAFT_OUT_DIR.mkdir(exist_ok=True)
    out_path = DRAFT_OUT_DIR / f"calibration_draft_{proposal['patient_id']}.json"
    out_path.write_text(json.dumps(proposal, indent=2))
    return out_path


def main(patient_id: str):
    proposal = draft_calibration_proposal(patient_id)
    draft_path = save_draft(proposal)

    print(f"=== Staged calibration proposal for {patient_id} ===\n")
    print(
        f"Calibration window: trials {proposal['window_trial_range'][0]}-{proposal['window_trial_range'][1]} "
        f"({proposal['window_trials']} trials)\n"
    )
    print(f"{'ch':<4}{'current':<10}{'refit':<10}{'drift':<9}{'R2':<7}{'proposed@100%':<15}")
    for c in proposal["channels"]:
        print(
            f"{c['channel']:<4}{c['current_deployed_direction_deg']:<10}{c['refit_direction_deg']:<10}"
            f"{c['drift_deg']:<9}{c['fit_r_squared']:<7}{c['proposed_direction_deg_default_strength']:<15}"
        )
    print(
        f"\nMean abs drift: {proposal['mean_abs_drift_deg']} deg | "
        f"Max abs drift: {proposal['max_abs_drift_deg']} deg"
    )
    if proposal["low_confidence_channels"]:
        print(
            f"LOW CONFIDENCE (R^2 < {LOW_CONFIDENCE_R2_THRESHOLD}) on channels: "
            f"{proposal['low_confidence_channels']} -- flag this in the cockpit."
        )
    print(f"\n(Draft staged to {draft_path} -- nothing written to DataHub yet, pending cockpit review.)")


if __name__ == "__main__":
    pid = sys.argv[1] if len(sys.argv) > 1 else "patient_11"
    main(pid)