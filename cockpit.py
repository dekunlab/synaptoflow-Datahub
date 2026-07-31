"""
cockpit.py

Review interface for staged recalibration proposals. Reads the artifacts
that run_cascade.py's diagnostic and calibration agents write to monitor/
and lets a human accept, adjust, or reject each one. Writes the reviewed
outcome to monitor/approved_<patient_id>.json for the deployment agent to
pick up.

This process does not call Groq, does not run any regression, and does not
talk to DataHub. Every number shown for a given recalibration strength is
computed from the current_deployed_direction_deg and drift_deg already
present in the calibration agent's output: proposed = current + drift *
(strength / 100), wrapped to [0, 360).

Run with: streamlit run cockpit.py
"""

import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).parent
STATE_DIR = REPO_ROOT / "state"
INCIDENT_STATE_PATH = STATE_DIR / "incident_state.json"

N_CHANNELS = 8
LOW_CONFIDENCE_R2_THRESHOLD = 0.5


# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------

def diagnostic_draft_path(patient_id: str) -> Path:
    return STATE_DIR / f"diagnostic_draft_{patient_id}.txt"


def calibration_draft_path(patient_id: str) -> Path:
    return STATE_DIR / f"calibration_draft_{patient_id}.json"


def approved_path(patient_id: str) -> Path:
    return STATE_DIR / f"approved_{patient_id}.json"


def load_incident_queue() -> dict:
    if not INCIDENT_STATE_PATH.exists():
        return {}
    return json.loads(INCIDENT_STATE_PATH.read_text())


def load_diagnostic_draft(patient_id: str) -> str | None:
    path = diagnostic_draft_path(patient_id)
    if not path.exists():
        return None
    return path.read_text().strip()


def load_calibration_draft(patient_id: str) -> dict | None:
    path = calibration_draft_path(patient_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_approved(patient_id: str) -> dict | None:
    path = approved_path(patient_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_approved(payload: dict) -> Path:
    STATE_DIR.mkdir(exist_ok=True)
    path = approved_path(payload["patient_id"])
    path.write_text(json.dumps(payload, indent=2))
    return path


# --------------------------------------------------------------------------
# Proposal math (arithmetic only, against fields the calibration agent
# already computed and staged — no regression runs here)
# --------------------------------------------------------------------------

def proposed_direction(current_deg: float, drift_deg: float, strength_pct: float) -> float:
    return (current_deg + drift_deg * (strength_pct / 100.0)) % 360.0


def status_label(patient_id: str) -> str:
    if load_approved(patient_id) is not None:
        return "REVIEWED"
    if load_diagnostic_draft(patient_id) is None or load_calibration_draft(patient_id) is None:
        return "PENDING"
    return "READY"


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        :root {
            --bg: #ffffff;
            --bg-subtle: #fafafa;
            --text: #0a0a0a;
            --text-muted: #6b6b6b;
            --border: #e4e4e4;
            --border-strong: #0a0a0a;
            --accent-alert: #ac3b3b;
            --accent-ok: #2f6b4f;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, sans-serif;
            color: var(--text);
        }

        .stApp {
            background: var(--bg);
        }

        .mono { font-family: 'JetBrains Mono', monospace; }

        .eyebrow {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 500;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 0.4rem;
        }

        .panel-title {
            font-size: 1.6rem;
            font-weight: 700;
            letter-spacing: -0.01em;
            margin-bottom: 0.1rem;
        }

        .panel-meta {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            color: var(--text-muted);
            margin-bottom: 1.4rem;
        }

        .card {
            border: 1px solid var(--border);
            background: var(--bg-subtle);
            padding: 1.1rem 1.3rem;
            animation: fadeIn 0.3s ease-out;
        }

        .card-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 0.6rem;
        }

        .card-body {
            font-size: 0.92rem;
            line-height: 1.55;
            white-space: pre-wrap;
        }

        .alert-box {
            border: 1px solid var(--accent-alert);
            background: #fdf5f5;
            color: var(--accent-alert);
            padding: 0.7rem 1rem;
            font-size: 0.82rem;
            font-family: 'JetBrains Mono', monospace;
            animation: fadeIn 0.3s ease-out;
        }

        .empty-state {
            border: 1px dashed var(--border);
            padding: 1.4rem;
            color: var(--text-muted);
            font-size: 0.9rem;
            animation: fadeIn 0.3s ease-out;
        }

        .readout-row {
            display: flex;
            gap: 2.2rem;
            margin: 0.6rem 0 1.1rem 0;
        }

        .readout {
            font-family: 'JetBrains Mono', monospace;
        }

        .readout-value {
            font-size: 1.3rem;
            font-weight: 600;
        }

        .readout-label {
            font-size: 0.68rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--text-muted);
        }

        .dial-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.6rem;
            margin-bottom: 1.2rem;
        }

        .dial-card {
            border: 1px solid var(--border);
            text-align: center;
            padding: 0.4rem 0.2rem 0.6rem 0.2rem;
            transition: border-color 0.15s ease;
            animation: fadeIn 0.3s ease-out;
        }

        .dial-card:hover {
            border-color: var(--border-strong);
        }

        .dial-caption {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            color: var(--text-muted);
            margin-top: 0.1rem;
        }

        .dial-drift {
            font-weight: 600;
        }

        .legend {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-bottom: 0.9rem;
        }

        .legend-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 0.35rem;
        }

        div.stButton > button {
            border-radius: 0;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            font-size: 0.85rem;
            padding: 0.5rem 1rem;
            transition: all 0.15s ease;
        }

        div.stButton > button:hover {
            transform: translateY(-1px);
        }

        [data-testid="stSidebar"] {
            background: var(--bg-subtle);
            border-right: 1px solid var(--border);
        }

        hr {
            border: none;
            border-top: 1px solid var(--border);
            margin: 1.6rem 0;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(4px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def dial_svg(channel: int, current_deg: float, drift_deg: float, strength_pct: float, low_confidence: bool) -> str:
    size, cx, cy, radius = 96, 48, 48, 34

    def point(deg: float, length: float) -> tuple[float, float]:
        rad = math.radians(deg)
        return cx + length * math.cos(rad), cy - length * math.sin(rad)

    proposed_deg = proposed_direction(current_deg, drift_deg, strength_pct)
    cur_x, cur_y = point(current_deg, radius - 5)
    prop_x, prop_y = point(proposed_deg, radius - 5)

    tick_marks = ""
    for ref_deg in (0, 90, 180, 270):
        x1, y1 = point(ref_deg, radius)
        x2, y2 = point(ref_deg, radius - 4)
        tick_marks += f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#d0d0d0" stroke-width="1"/>'

    proposed_stroke = "#ac3b3b" if abs(drift_deg * strength_pct / 100.0) > 1.0 else "#0a0a0a"
    dash = ' stroke-dasharray="2,2"' if low_confidence else ""

    return f'''
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
        <circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="#e4e4e4" stroke-width="1"/>
        {tick_marks}
        <line x1="{cx}" y1="{cy}" x2="{prop_x:.1f}" y2="{prop_y:.1f}" stroke="{proposed_stroke}" stroke-width="2"{dash}/>
        <line x1="{cx}" y1="{cy}" x2="{cur_x:.1f}" y2="{cur_y:.1f}" stroke="#0a0a0a" stroke-width="1.5" opacity="0.35"/>
        <circle cx="{cx}" cy="{cy}" r="2" fill="#0a0a0a"/>
    </svg>
    '''


def render_channel_grid(channels: list[dict], strength_pct: float) -> None:
    st.markdown(
        '<div class="legend">'
        '<span class="legend-dot" style="background:#0a0a0a; opacity:0.35;"></span>currently deployed &nbsp;&nbsp;'
        '<span class="legend-dot" style="background:#ac3b3b;"></span>proposed at current strength'
        "</div>",
        unsafe_allow_html=True,
    )
    cards = []
    for ch in channels:
        proposed = proposed_direction(ch["current_deployed_direction_deg"], ch["drift_deg"], strength_pct)
        low_conf = ch["fit_r_squared"] < LOW_CONFIDENCE_R2_THRESHOLD
        svg = dial_svg(ch["channel"], ch["current_deployed_direction_deg"], ch["drift_deg"], strength_pct, low_conf)
        scaled_drift = ch["drift_deg"] * strength_pct / 100.0
        cards.append(
            f'<div class="dial-card">{svg}'
            f'<div class="dial-caption">CH {ch["channel"]}</div>'
            f'<div class="dial-caption mono dial-drift">{proposed:.1f}&deg;</div>'
            f'<div class="dial-caption">{scaled_drift:+.1f}&deg;</div>'
            f"</div>"
        )
    st.markdown(f'<div class="dial-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_patient(patient_id: str, incident_urn: str) -> None:
    diagnosis_text = load_diagnostic_draft(patient_id)
    calibration = load_calibration_draft(patient_id)
    already_reviewed = load_approved(patient_id)

    st.markdown('<div class="eyebrow">// Incident</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="panel-title">{patient_id}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="panel-meta mono">{incident_urn}</div>', unsafe_allow_html=True)

    if already_reviewed is not None:
        st.markdown(
            f'<div class="alert-box" style="border-color: var(--accent-ok); background: #f4f8f6; color: var(--accent-ok);">'
            f'Reviewed {already_reviewed["reviewed_at"]} &mdash; status: {already_reviewed["status"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">// Diagnosis</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="card-label">Agent draft</div>', unsafe_allow_html=True)
        if diagnosis_text is None:
            st.markdown(
                '<div class="empty-state">Diagnosis pending. The diagnostic agent has not written its output yet.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="card"><div class="card-body">{html.escape(diagnosis_text)}</div></div>',
                unsafe_allow_html=True,
            )

    with col_b:
        st.markdown('<div class="card-label">Clinical notes</div>', unsafe_allow_html=True)
        notes_key = f"notes_{patient_id}"
        if notes_key not in st.session_state:
            st.session_state[notes_key] = diagnosis_text or ""
        st.text_area(
            "Clinical notes",
            key=notes_key,
            height=160,
            label_visibility="collapsed",
            disabled=diagnosis_text is None or already_reviewed is not None,
        )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">// Calibration</div>', unsafe_allow_html=True)

    if calibration is None:
        st.markdown(
            '<div class="empty-state">Calibration pending. The calibration agent has not written its output yet.</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<div class="panel-meta mono">window: trials {calibration["window_trial_range"][0]}-'
        f'{calibration["window_trial_range"][1]} ({calibration["window_trials"]} trials)</div>',
        unsafe_allow_html=True,
    )

    strength_key = f"strength_{patient_id}"
    default_strength = calibration.get("default_recal_strength_pct", 100.0)

    if already_reviewed is not None:
        strength_pct = already_reviewed.get("recalibration_strength_pct")
        if strength_pct is None:
            strength_pct = already_reviewed.get("reviewed_proposal", {}).get("recalibration_strength_pct", 0)
        st.markdown(
            f'<div class="panel-meta mono">Recalibration strength: {strength_pct}% '
            f'(locked &mdash; already {already_reviewed["status"]})</div>',
            unsafe_allow_html=True,
        )
    else:
        strength_pct = st.slider(
            "Recalibration strength",
            min_value=0,
            max_value=100,
            value=int(default_strength),
            key=strength_key,
        )

    scaled_mean = calibration["mean_abs_drift_deg"] * strength_pct / 100.0
    scaled_max = calibration["max_abs_drift_deg"] * strength_pct / 100.0

    st.markdown(
        f'<div class="readout-row">'
        f'<div class="readout"><div class="readout-value">{scaled_mean:.1f}&deg;</div>'
        f'<div class="readout-label">mean channel shift</div></div>'
        f'<div class="readout"><div class="readout-value">{scaled_max:.1f}&deg;</div>'
        f'<div class="readout-label">max channel shift</div></div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    render_channel_grid(calibration["channels"], strength_pct)

    if calibration["low_confidence_channels"]:
        chans = ", ".join(str(c) for c in calibration["low_confidence_channels"])
        st.markdown(
            f'<div class="alert-box">Low confidence fit (R&sup2; &lt; {LOW_CONFIDENCE_R2_THRESHOLD}) '
            f"on channel(s) {chans}. Shown with a dashed proposed line above.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">// Review</div>', unsafe_allow_html=True)

    def channels_at(strength: float) -> list[dict]:
        return [
            {
                "channel": ch["channel"],
                "current_deployed_direction_deg": ch["current_deployed_direction_deg"],
                "refit_direction_deg": ch["refit_direction_deg"],
                "proposed_direction_deg": round(
                    proposed_direction(ch["current_deployed_direction_deg"], ch["drift_deg"], strength), 2
                ),
            }
            for ch in calibration["channels"]
        ]

    def build_payload(status: str) -> dict:
        payload = {
            "patient_id": patient_id,
            "incident_urn": incident_urn,
            "status": status,
            "diagnosis_text": diagnosis_text,
            "clinical_notes": st.session_state.get(notes_key, ""),
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }

        if status == "rejected":
            # Kept as a snapshot of what was on screen at the moment of
            # rejection, for the audit trail -- not an instruction. Nested
            # separately from the top-level recalibration_strength_pct and
            # channels fields the approved statuses use, so nothing here
            # can be mistaken for a value to deploy.
            payload["reviewed_proposal"] = {
                "recalibration_strength_pct": strength_pct,
                "channels": channels_at(strength_pct),
            }
            return payload

        applied_strength = default_strength if status == "approved_as_is" else strength_pct
        payload["recalibration_strength_pct"] = applied_strength
        payload["channels"] = channels_at(applied_strength)
        return payload

    if already_reviewed is None:
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Approve As-Is", key=f"approve_as_is_{patient_id}", use_container_width=True):
                save_approved(build_payload("approved_as_is"))
                st.rerun()
        with col2:
            if st.button("Approve With Edits", key=f"approve_edits_{patient_id}", use_container_width=True):
                save_approved(build_payload("approved_with_edits"))
                st.rerun()
        with col3:
            if st.button("Reject", key=f"reject_{patient_id}", use_container_width=True):
                save_approved(build_payload("rejected"))
                st.rerun()
    else:
        st.markdown(
            '<div class="panel-meta mono">Decision recorded &mdash; no further action available for this incident.</div>',
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(page_title="SynaptoFlow Cockpit", layout="wide", initial_sidebar_state="expanded")
    inject_styles()

    with st.sidebar:
        st.markdown('<div class="eyebrow">// SynaptoFlow</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-title" style="font-size:1.15rem;">Calibration Review</div>', unsafe_allow_html=True)
        st.markdown("<hr/>", unsafe_allow_html=True)

        queue = load_incident_queue()
        if not queue:
            st.markdown(
                '<div class="empty-state">No open incidents.</div>',
                unsafe_allow_html=True,
            )
            return

        patient_ids = sorted(queue.keys())
        labels = [f"{pid}  ·  {status_label(pid)}" for pid in patient_ids]
        selected_label = st.radio("Open incidents", labels, label_visibility="collapsed")
        selected_patient = patient_ids[labels.index(selected_label)]

    render_patient(selected_patient, queue[selected_patient])


if __name__ == "__main__":
    main()