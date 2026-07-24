"""
agents/diagnostic_agent.py

Diagnostic Agent for SynaptoFlow.

Triggered by an open DataHub Incident. Gathers patient decoder context 
(entity details, lineage) via the MCP server and recent telemetry trends. 
Uses Groq to generate a plain-English clinical explanation of the signal drift.

Note: This script only stages the diagnosis locally to 
`monitor/diagnostic_draft_<patient_id>.txt`. It does not execute DataHub mutations. 
Write-backs are handled by the Deployment Agent after clinical approval in the Cockpit.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import pandas as pd
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

PLATFORM = "synaptoflow"
MONITOR_DIR = Path(__file__).parent.parent / "monitor"
STATE_FILE = MONITOR_DIR / "incident_state.json"


def model_urn(patient_id: str) -> str:
    return f"urn:li:mlModel:(urn:li:dataPlatform:{PLATFORM},decoder_{patient_id},PROD)"


def deployment_urn(patient_id: str) -> str:
    return f"urn:li:mlModelDeployment:(urn:li:dataPlatform:{PLATFORM},live_session_{patient_id},PROD)"


def get_recent_telemetry_summary(patient_id: str) -> str:
    """Reads the actual telemetry.csv and summarizes the recent trend -- real numbers, not invented."""
    df = pd.read_csv("sim/output/telemetry.csv")
    pdf = df[df.patient_id == patient_id].reset_index(drop=True)
    recent = pdf.tail(50)
    return (
        f"Angle error over the last 50 trials: min={recent.angle_error_deg.min():.1f}, "
        f"max={recent.angle_error_deg.max():.1f}, mean={recent.angle_error_deg.mean():.1f} degrees.\n"
        f"KL divergence over the last 50 trials: min={recent.kl_divergence.min():.2f}, "
        f"max={recent.kl_divergence.max():.2f}, mean={recent.kl_divergence.mean():.2f}."
    )


async def gather_context(session: ClientSession, patient_id: str) -> dict:
    """
    Pulls real context from DataHub via MCP tools.
    """
    m_urn = model_urn(patient_id)

    entities_result = await session.call_tool("get_entities", {"urns": [m_urn]})
    lineage_result = await session.call_tool(
        "get_lineage", {"urn": m_urn, "upstream": True, "max_hops": 3}
    )

    return {
        "entities": entities_result,
        "lineage": lineage_result,
    }


def draft_diagnosis(patient_id: str, context: dict, telemetry_summary: str) -> str:
    """Calls a real LLM (Groq) with real gathered context to draft a plain-English diagnosis."""
    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )

    prompt = f"""You are a clinical BCI engineer reviewing a signal drift incident for {patient_id}.

Real telemetry summary:
{telemetry_summary}

Real DataHub entity/lineage context:
{json.dumps(str(context))[:2000]}

Write a short (3-4 sentence), plain-English clinical explanation of what's
likely happening and why recalibration is being proposed. Do not invent
numbers -- only reference the telemetry summary above."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def save_draft(patient_id: str, diagnosis: str) -> Path:
    """Saves the generated diagnosis to a staged text file for the Cockpit to read."""
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    draft_path = MONITOR_DIR / f"diagnostic_draft_{patient_id}.txt"
    draft_path.write_text(diagnosis, encoding="utf-8")
    return draft_path


async def main(patient_id: str):
    if not STATE_FILE.exists():
        print(f"State file {STATE_FILE} does not exist. Nothing to diagnose.")
        return

    incident_state = json.loads(STATE_FILE.read_text())
    if patient_id not in incident_state:
        print(f"No open incident found for {patient_id} in {STATE_FILE}. Nothing to diagnose.")
        return

    telemetry_summary = get_recent_telemetry_summary(patient_id)

    server_env = {
        **os.environ,
        "DATAHUB_GMS_URL": os.environ.get("DATAHUB_GMS_URL", "http://localhost:8080"),
        "DATAHUB_GMS_TOKEN": os.environ["DATAHUB_GMS_TOKEN"],
        "TOOLS_IS_MUTATION_ENABLED": "true",
    }
    server_params = StdioServerParameters(command="uvx", args=["mcp-server-datahub@latest"], env=server_env)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            context = await gather_context(session, patient_id)

    diagnosis = draft_diagnosis(patient_id, context, telemetry_summary)
    draft_file = save_draft(patient_id, diagnosis)

    print(f"=== Staged diagnosis for {patient_id} (incident {incident_state[patient_id]}) ===\n")
    print(diagnosis)
    print(f"\n[✓] Draft successfully saved to {draft_file}")
    print("(Nothing written to DataHub yet -- this is a draft, pending cockpit review.)")


if __name__ == "__main__":
    pid = sys.argv[1] if len(sys.argv) > 1 else "patient_11"
    asyncio.run(main(pid))