"""
agents/deployment_agent.py

Reads every state/approved_<patient_id>.json not yet deployed and writes
the clinician's decision back to DataHub: resolving the incident, flipping
the drift tag, tagging provenance, saving a Decision document, and updating
the local calibration baseline. Rejected proposals are documented but never
deployed.

Run standalone once a batch of proposals is ready for processing:

    python3 agents/deployment_agent.py

Idempotent by design: each processed file is marked "deployed": true, so
re-running only picks up files that haven't been handled yet.
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from datahub.emitter.mcp import MetadataChangeProposalWrapper
from datahub.emitter.rest_emitter import DatahubRestEmitter
from datahub.metadata.schema_classes import TagPropertiesClass

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

PLATFORM = "synaptoflow"
REPO_ROOT = Path(__file__).parent.parent
STATE_DIR = REPO_ROOT / "state"
INCIDENT_STATE_FILE = STATE_DIR / "incident_state.json"
CALIB_PARAMS_PATH = REPO_ROOT / "sim" / "output" / "calibration_params.json"

RESOLUTION_TAGS = [
    (
        "resolution-ai-proposed",
        "Resolution: AI-Proposed",
        "SynaptoFlow: this deployment's decoder was recalibrated using the AI-proposed values unmodified.",
    ),
    (
        "resolution-human-modified",
        "Resolution: Human-Modified",
        "SynaptoFlow: this deployment's decoder was recalibrated using values a clinician edited before approving.",
    ),
]


def deployment_urn(patient_id: str) -> str:
    return f"urn:li:mlModelDeployment:(urn:li:dataPlatform:{PLATFORM},live_session_{patient_id},PROD)"


def dataset_urn(patient_id: str) -> str:
    return f"urn:li:dataset:(urn:li:dataPlatform:{PLATFORM},raw_neural_stream_{patient_id},PROD)"


def model_urn(patient_id: str) -> str:
    return f"urn:li:mlModel:(urn:li:dataPlatform:{PLATFORM},decoder_{patient_id},PROD)"


# --------------------------------------------------------------------------
# One-time setup
# --------------------------------------------------------------------------

def ensure_resolution_tags_exist(gms_url: str, token: str) -> None:
    """Creates the provenance tags before the deployment loop runs. Tags never auto-create."""
    emitter = DatahubRestEmitter(gms_server=gms_url, token=token)
    for tag_id, name, description in RESOLUTION_TAGS:
        emitter.emit_mcp(
            MetadataChangeProposalWrapper(
                entityUrn=f"urn:li:tag:{tag_id}",
                aspect=TagPropertiesClass(name=name, description=description),
            )
        )
    print("Confirmed resolution-ai-proposed and resolution-human-modified tags exist.\n")


# --------------------------------------------------------------------------
# MCP tool result handling
# --------------------------------------------------------------------------

def _check_tool_result(result, tool_name: str) -> None:
    if getattr(result, "isError", False):
        detail = result.content[0].text if result.content else "no error detail returned"
        raise RuntimeError(f"{tool_name} failed: {detail}")


def _extract_tool_payload(result) -> Optional[dict]:
    """
    save_document returns a structured dict (success, urn, message, author).
    FastMCP tools populate structuredContent when available; older clients
    fall back to the JSON-encoded text block, so both are checked here.
    """
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    if result.content:
        try:
            return json.loads(result.content[0].text)
        except (json.JSONDecodeError, AttributeError):
            return None
    return None


# --------------------------------------------------------------------------
# GraphQL: incident resolution
# --------------------------------------------------------------------------

def resolve_incident(gms_url: str, token: str, incident_urn: str, message: str) -> bool:
    """Invokes the DataHub updateIncidentStatus GraphQL mutation."""
    query = """
    mutation updateIncidentStatus($urn: String!, $input: IncidentStatusInput!) {
      updateIncidentStatus(urn: $urn, input: $input)
    }
    """
    variables = {
        "urn": incident_urn,
        "input": {"state": "RESOLVED", "stage": "FIXED", "message": message},
    }
    resp = requests.post(
        f"{gms_url}/api/graphql",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"query": query, "variables": variables},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL error resolving incident: {data['errors']}")
    return data["data"]["updateIncidentStatus"]


# --------------------------------------------------------------------------
# MCP tool calls
# --------------------------------------------------------------------------

async def flip_drift_tag(session: ClientSession, patient_id: str) -> None:
    dep_urn = deployment_urn(patient_id)
    result = await session.call_tool("add_tags", {"tag_urns": ["urn:li:tag:drift-baseline"], "entity_urns": [dep_urn]})
    _check_tool_result(result, "add_tags")
    result = await session.call_tool("remove_tags", {"tag_urns": ["urn:li:tag:drift-drifted"], "entity_urns": [dep_urn]})
    _check_tool_result(result, "remove_tags")


async def tag_resolution_provenance(session: ClientSession, patient_id: str, status: str) -> None:
    tag = "urn:li:tag:resolution-ai-proposed" if status == "approved_as_is" else "urn:li:tag:resolution-human-modified"
    result = await session.call_tool("add_tags", {"tag_urns": [tag], "entity_urns": [deployment_urn(patient_id)]})
    _check_tool_result(result, "add_tags")


def resolution_message(payload: dict) -> str:
    return (
        f"Recalibration applied at {payload['recalibration_strength_pct']}% strength "
        f"({payload['status']}). See linked Decision document for full detail."
    )


def build_decision_content(payload: dict) -> str:
    lines = [
        "## Diagnosis",
        payload.get("diagnosis_text") or "(none)",
        "",
        "## Clinical Notes",
        payload.get("clinical_notes") or "(none)",
    ]

    if payload["status"] == "rejected":
        reviewed = payload.get("reviewed_proposal")
        if reviewed:
            lines += [
                "",
                f"## Proposal Reviewed and Rejected (was showing "
                f"{reviewed['recalibration_strength_pct']}% recalibration strength)",
                "",
                "| Channel | Currently Deployed (deg) | Was Proposing (deg) |",
                "|---|---|---|",
            ]
            for ch in reviewed["channels"]:
                lines.append(
                    f"| {ch['channel']} | {ch['current_deployed_direction_deg']} | {ch['proposed_direction_deg']} |"
                )
    else:
        lines += [
            "",
            f"## Recalibration Applied ({payload['recalibration_strength_pct']}% strength)",
            "",
            "| Channel | Previous (deg) | New (deg) |",
            "|---|---|---|",
        ]
        for ch in payload["channels"]:
            lines.append(
                f"| {ch['channel']} | {ch['current_deployed_direction_deg']} | {ch['proposed_direction_deg']} |"
            )

    return "\n".join(lines)


async def save_decision_document(session: ClientSession, patient_id: str, payload: dict) -> Optional[dict]:
    title = f"Recalibration Decision - {patient_id} - {payload['status']}"
    result = await session.call_tool(
        "save_document",
        {
            "document_type": "Decision",
            "title": title,
            "content": build_decision_content(payload),
            "related_assets": [dataset_urn(patient_id), model_urn(patient_id)],
        },
    )
    _check_tool_result(result, "save_document")
    doc = _extract_tool_payload(result)
    if doc is None:
        print(f"  warning: could not parse save_document response for {patient_id}; document was likely still saved.")
    elif not doc.get("success", True):
        raise RuntimeError(f"save_document reported failure: {doc.get('message')}")
    return doc


# --------------------------------------------------------------------------
# Incident queue / calibration baseline updates
# --------------------------------------------------------------------------

def remove_from_incident_queue(patient_id: str) -> None:
    """
    Clears this patient's local incident pointer once their approval file
    has been fully processed, regardless of outcome (deployed OR
    documented-and-rejected). Without this, monitor.py's own
    "if patient_id not in incident_state" check would permanently refuse
    to ever raise a new incident for this patient again, since the stale
    key from this episode would still be sitting there. Safe to call
    repeatedly -- removing an already-absent key is a no-op.
    """
    if not INCIDENT_STATE_FILE.exists():
        return
    incident_state = json.loads(INCIDENT_STATE_FILE.read_text())
    if patient_id in incident_state:
        del incident_state[patient_id]
        INCIDENT_STATE_FILE.write_text(json.dumps(incident_state, indent=2))


def update_calibration_params(patient_id: str, channels: list) -> None:
    """Replaces this patient's calibration baseline with the deployed values, indexed by channel."""
    n = len(channels)
    by_channel = {ch["channel"]: ch["proposed_direction_deg"] for ch in channels}
    if set(by_channel.keys()) != set(range(n)):
        raise ValueError(f"Channels for {patient_id} are not a complete 0-{n - 1} set: {sorted(by_channel.keys())}")

    all_params = json.loads(CALIB_PARAMS_PATH.read_text())
    all_params[patient_id] = [by_channel[ch] for ch in range(n)]
    CALIB_PARAMS_PATH.write_text(json.dumps(all_params, indent=2))


# --------------------------------------------------------------------------
# Per-file processing
# --------------------------------------------------------------------------

async def process_approved_file(session: ClientSession, gms_url: str, token: str, path: Path) -> tuple:
    """
    Each side-effecting step is checkpointed to disk individually, not just
    at the end. If a step fails partway through (e.g. a network error on
    save_document after the incident has already been resolved), a re-run
    skips the steps already marked done instead of repeating them -- so an
    incident never gets resolved twice and a Decision document never gets
    created twice for the same approval.
    """
    payload = json.loads(path.read_text())
    patient_id = payload["patient_id"]

    if payload.get("deployed"):
        return patient_id, "already processed, skipped"

    status = payload["status"]

    def checkpoint() -> None:
        path.write_text(json.dumps(payload, indent=2))

    if status == "rejected":
        if not payload.get("decision_document_urn"):
            doc = await save_decision_document(session, patient_id, payload)
            if doc and doc.get("urn"):
                payload["decision_document_urn"] = doc["urn"]
            checkpoint()
        outcome = "rejected -- decision documented, no deployment"

    elif status in ("approved_as_is", "approved_with_edits"):
        if not payload.get("incident_resolved"):
            resolve_incident(gms_url, token, payload["incident_urn"], resolution_message(payload))
            payload["incident_resolved"] = True
            checkpoint()

        if not payload.get("tags_updated"):
            await flip_drift_tag(session, patient_id)
            await tag_resolution_provenance(session, patient_id, status)
            payload["tags_updated"] = True
            checkpoint()

        if not payload.get("decision_document_urn"):
            doc = await save_decision_document(session, patient_id, payload)
            if doc and doc.get("urn"):
                payload["decision_document_urn"] = doc["urn"]
            checkpoint()

        if not payload.get("calibration_updated"):
            update_calibration_params(patient_id, payload["channels"])
            payload["calibration_updated"] = True
            checkpoint()

        outcome = f"deployed at {payload['recalibration_strength_pct']}% strength ({status})"

    else:
        raise ValueError(f"Unrecognized status '{status}'")

    if payload.get("decision_document_urn"):
        outcome += f" | doc: {payload['decision_document_urn']}"

    remove_from_incident_queue(patient_id)
    payload["deployed"] = True
    payload["deployed_at"] = datetime.now(timezone.utc).isoformat()
    checkpoint()

    return patient_id, outcome


POLL_INTERVAL_SECONDS = 5


async def run_deployment(session: ClientSession, gms_url: str, token: str) -> None:
    approved_files = sorted(STATE_DIR.glob("approved_*.json"))
    pending = [p for p in approved_files if not json.loads(p.read_text()).get("deployed")]
    if not pending:
        return

    print(f"Found {len(pending)} approved file(s) to process...\n")
    for path in pending:
        try:
            patient_id, outcome = await process_approved_file(session, gms_url, token, path)
            print(f"[OK]     {patient_id}: {outcome}")
        except Exception as exc:
            print(f"[FAILED] {path.name}: {exc}")


async def main() -> None:
    gms_url = os.environ.get("DATAHUB_GMS_URL", "http://localhost:8080")
    token = os.environ["DATAHUB_GMS_TOKEN"]

    server_env = {
        **os.environ,
        "DATAHUB_GMS_URL": gms_url,
        "DATAHUB_GMS_TOKEN": token,
        "TOOLS_IS_MUTATION_ENABLED": "true",
    }
    server_params = StdioServerParameters(command="uvx", args=["mcp-server-datahub@latest"], env=server_env)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            ensure_resolution_tags_exist(gms_url, token)

            print(f"Deployment watcher started. Polling every {POLL_INTERVAL_SECONDS}s. Press Ctrl+C to stop.\n")
            while True:
                await run_deployment(session, gms_url, token)
                await asyncio.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    asyncio.run(main())