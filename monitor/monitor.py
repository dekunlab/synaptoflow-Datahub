import asyncio
import json
import os
from pathlib import Path

import pandas as pd
import requests
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from datahub.emitter.mcp import MetadataChangeProposalWrapper
from datahub.emitter.rest_emitter import DatahubRestEmitter
from datahub.metadata.schema_classes import TagPropertiesClass

PLATFORM = "synaptoflow"
KL_THRESHOLD = 0.5
ANGLE_ERROR_THRESHOLD = 11.0
CHECKPOINT_EVERY = 10   # trials
ROLLING_ERR_WINDOW = 20

ANGLE_PROP_URN = "urn:li:structuredProperty:io.synaptoflow.angleErrorDegrees"
KL_PROP_URN = "urn:li:structuredProperty:io.synaptoflow.klDivergenceScore"

STATE_FILE = Path(__file__).parent / "incident_state.json"


def deployment_urn(patient_id: str) -> str:
    return f"urn:li:mlModelDeployment:(urn:li:dataPlatform:{PLATFORM},live_session_{patient_id},PROD)"


def dataset_urn(patient_id: str) -> str:
    return f"urn:li:dataset:(urn:li:dataPlatform:{PLATFORM},raw_neural_stream_{patient_id},PROD)"


def ensure_tags_exist(gms_url: str, token: str) -> None:
    """
    Creates the two drift-state tags as real Tag entities, once, before the
    monitor loop runs. Tags do NOT auto-create on first use in this DataHub
    version (confirmed by a real error) -- same proven low-level MCP
    pattern already used successfully elsewhere in this build (MLFeature,
    MLPrimaryKey, MLModelDeployment all use this exact mechanism).
    """
    emitter = DatahubRestEmitter(gms_server=gms_url, token=token)
    tags = [
        ("drift-baseline", "Drift: Baseline", "SynaptoFlow: this deployment's decoder is currently within normal drift guardrails."),
        ("drift-drifted", "Drift: Drifted", "SynaptoFlow: this deployment's decoder has crossed a drift guardrail and needs review."),
    ]
    for tag_id, name, description in tags:
        emitter.emit_mcp(MetadataChangeProposalWrapper(
            entityUrn=f"urn:li:tag:{tag_id}",
            aspect=TagPropertiesClass(name=name, description=description),
        ))
    print("Confirmed drift-baseline and drift-drifted tags exist.\n")


def raise_incident(gms_url: str, token: str, resource_urn: str, title: str, description: str) -> str:
    """Calls DataHub's raiseIncident GraphQL mutation directly (no MCP tool exists for this yet)."""
    query = """
    mutation raiseIncident($input: RaiseIncidentInput!) {
      raiseIncident(input: $input)
    }
    """
    variables = {
        "input": {
            "type": "OPERATIONAL",
            "title": title,
            "description": description,
            "resourceUrn": resource_urn,
        }
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
        raise RuntimeError(f"GraphQL error raising incident: {data['errors']}")
    return data["data"]["raiseIncident"]


async def run_monitor():
    gms_url = os.environ.get("DATAHUB_GMS_URL", "http://localhost:8080")
    token = os.environ["DATAHUB_GMS_TOKEN"]

    telemetry = pd.read_csv("sim/output/telemetry.csv")
    patient_ids = sorted(telemetry.patient_id.unique())

    incident_state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}

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

            ensure_tags_exist(gms_url, token)

            for patient_id in patient_ids:
                pdf = telemetry[telemetry.patient_id == patient_id].reset_index(drop=True)
                dep_urn = deployment_urn(patient_id)
                ds_urn = dataset_urn(patient_id)
                print(f"\n=== {patient_id} ===")

                # Every patient starts at baseline right after calibration -- set that explicitly
                await session.call_tool("add_tags", {
                    "tag_urns": ["urn:li:tag:drift-baseline"],
                    "entity_urns": [dep_urn],
                })
                current_state = "baseline"

                for checkpoint_trial in range(0, len(pdf), CHECKPOINT_EVERY):
                    kl = pdf["kl_divergence"].iloc[checkpoint_trial]
                    if pd.isna(kl):
                        continue  # not enough history yet for a KL estimate

                    window_start = max(0, checkpoint_trial - ROLLING_ERR_WINDOW + 1)
                    rolling_err = pdf["angle_error_deg"].iloc[window_start:checkpoint_trial + 1].mean()

                    # Always refresh the live numbers, every checkpoint -- on the DATASET,
                    # since Structured Properties don't support MLModelDeployment (see docstring)
                    await session.call_tool("add_structured_properties", {
                        "property_values": {
                            ANGLE_PROP_URN: [round(float(rolling_err), 2)],
                            KL_PROP_URN: [round(float(kl), 3)],
                        },
                        "entity_urns": [ds_urn],
                    })

                    is_drifted = (kl > KL_THRESHOLD) or (rolling_err > ANGLE_ERROR_THRESHOLD)
                    new_state = "drifted" if is_drifted else "baseline"

                    if new_state != current_state:
                        if new_state == "drifted":
                            await session.call_tool("add_tags", {
                                "tag_urns": ["urn:li:tag:drift-drifted"], "entity_urns": [dep_urn],
                            })
                            await session.call_tool("remove_tags", {
                                "tag_urns": ["urn:li:tag:drift-baseline"], "entity_urns": [dep_urn],
                            })
                            if patient_id not in incident_state:
                                incident_urn = raise_incident(
                                    gms_url, token, ds_urn,
                                    title=f"Signal Drift Detected - {patient_id}",
                                    description=(
                                        f"Angle error rolling average reached {rolling_err:.1f} deg and "
                                        f"KL divergence reached {kl:.2f} at trial {checkpoint_trial}, "
                                        f"crossing SynaptoFlow's drift guardrails "
                                        f"(KL > {KL_THRESHOLD} or angle error > {ANGLE_ERROR_THRESHOLD} deg). "
                                        f"See deployment {dep_urn} for current session state."
                                    ),
                                )
                                incident_state[patient_id] = incident_urn
                                STATE_FILE.write_text(json.dumps(incident_state, indent=2))  # save immediately, not just at the end
                                print(f"  trial {checkpoint_trial}: DRIFT DETECTED -> incident {incident_urn}")
                        else:
                            await session.call_tool("add_tags", {
                                "tag_urns": ["urn:li:tag:drift-baseline"], "entity_urns": [dep_urn],
                            })
                            await session.call_tool("remove_tags", {
                                "tag_urns": ["urn:li:tag:drift-drifted"], "entity_urns": [dep_urn],
                            })
                            print(f"  trial {checkpoint_trial}: back under threshold (incident, if any, stays open)")

                        current_state = new_state
                    else:
                        print(f"  trial {checkpoint_trial}: err={rolling_err:.1f} kl={kl:.2f} state={current_state}")

    STATE_FILE.write_text(json.dumps(incident_state, indent=2))
    print(f"\nDone. Open incidents tracked in {STATE_FILE}:")
    print(json.dumps(incident_state, indent=2))


if __name__ == "__main__":
    asyncio.run(run_monitor())