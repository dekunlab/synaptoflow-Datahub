import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import datahub.emitter.mce_builder as builder
import datahub.metadata.schema_classes as models
from datahub.emitter.mcp import MetadataChangeProposalWrapper
from datahub.emitter.rest_emitter import DatahubRestEmitter
from datahub.sdk import DataHubClient
from datahub.sdk.dataset import Dataset
from datahub.sdk.mlmodelgroup import MLModelGroup
from datahub.sdk.mlmodel import MLModel
from datahub.metadata.urns import MlModelGroupUrn

PLATFORM = "synaptoflow"
N_CHANNELS = 8  
MODEL_GROUP_ID = "synaptoflow-population-vector-decoder"


DRIFT_SEVERITIES = {
    "patient_01": 0.0, "patient_02": 0.0, "patient_03": 0.05, "patient_04": 0.05,
    "patient_05": 0.3, "patient_06": 0.5, "patient_07": 0.7, "patient_08": 0.9,
    "patient_09": 1.1, "patient_10": 1.4, "patient_11": 1.8, "patient_12": 2.3,
}


def main():
    gms_server = os.environ.get("DATAHUB_GMS_URL", "http://localhost:8080")
    token = os.environ.get("DATAHUB_GMS_TOKEN")

    
    client = DataHubClient.from_env()

    
    emitter = DatahubRestEmitter(gms_server=gms_server, token=token)

    telemetry = pd.read_csv("sim/output/telemetry.csv")
    patient_ids = sorted(telemetry.patient_id.unique())

    
    model_group = MLModelGroup(
        id=MODEL_GROUP_ID,
        platform=PLATFORM,
        name="SynaptoFlow Population-Vector Decoder",
        description=(
            "Family of per-patient cursor-control decoders. Each patient gets their "
            "own calibrated MLModel; this group ties them together as one decoder family."
        ),
        custom_properties={"decoder_type": "population_vector", "channels": str(N_CHANNELS)},
    )
    client.entities.upsert(model_group)
    print(f"Created model group: {model_group.urn}")

    for patient_id in patient_ids:
        severity = DRIFT_SEVERITIES[patient_id]
        patient_telemetry = telemetry[telemetry.patient_id == patient_id]
        kl_values = patient_telemetry.kl_divergence.dropna()
        final_kl = kl_values.iloc[-1] if len(kl_values) else None
        final_err = patient_telemetry.angle_error_deg.iloc[-10:].mean()

        
        dataset = Dataset(
            platform=PLATFORM,
            name=f"raw_neural_stream_{patient_id}",
            description=f"Synthetic raw multi-channel neural telemetry stream for {patient_id}.",
            custom_properties={
                "channels": str(N_CHANNELS),
                "drift_severity_deg": str(severity),
            },
        )
        client.entities.upsert(dataset)
        dataset_urn = str(dataset.urn)

        
        feature_table_name = f"neural_features_{patient_id}"

        primary_key_urn = builder.make_ml_primary_key_urn(
            feature_table_name=feature_table_name, primary_key_name="patient_id"
        )
        emitter.emit_mcp(MetadataChangeProposalWrapper(
            entityUrn=primary_key_urn,
            aspect=models.MLPrimaryKeyPropertiesClass(
                description="Identifies which synthetic patient this feature set belongs to.",
                sources=[dataset_urn],
                dataType="TEXT",
            ),
        ))

        feature_urns = []
        for ch in range(N_CHANNELS):
            feature_urn = builder.make_ml_feature_urn(
                feature_table_name=feature_table_name,
                feature_name=f"channel_{ch}_band_power",
            )
            emitter.emit_mcp(MetadataChangeProposalWrapper(
                entityUrn=feature_urn,
                aspect=models.MLFeaturePropertiesClass(
                    description=f"Band-power feature for synthetic channel {ch}.",
                    sources=[dataset_urn],
                    dataType="CONTINUOUS",
                ),
            ))
            feature_urns.append(feature_urn)

        feature_table_urn = builder.make_ml_feature_table_urn(
            feature_table_name=feature_table_name, platform=PLATFORM
        )
        emitter.emit_mcp(MetadataChangeProposalWrapper(
            entityUrn=feature_table_urn,
            aspect=models.MLFeatureTablePropertiesClass(
                description=f"Extracted band-power features for {patient_id}'s decoder.",
                mlFeatures=feature_urns,
                mlPrimaryKeys=[primary_key_urn],
            ),
        ))

        
        mlmodel = MLModel(
            id=f"decoder_{patient_id}",
            platform=PLATFORM,
            name=f"Decoder ({patient_id})",
            description=f"Population-vector decoder calibrated for {patient_id}.",
            model_group=MlModelGroupUrn(platform=PLATFORM, name=MODEL_GROUP_ID),
            custom_properties={
                "drift_severity_deg": str(severity),
                "channels": str(N_CHANNELS),
            },
            extra_aspects=[
                models.MLModelPropertiesClass(mlFeatures=feature_urns),
            ],
        )
        client.entities.upsert(mlmodel)
        model_urn = mlmodel.urn

    
        deployment_urn = builder.make_ml_model_deployment_urn(
            platform=PLATFORM, deployment_name=f"live_session_{patient_id}", env="PROD"
        )
        emitter.emit_mcp(MetadataChangeProposalWrapper(
            entityUrn=deployment_urn,
            aspect=models.MLModelDeploymentPropertiesClass(
                description=f"Live BCI session for {patient_id}.",
                customProperties={
                    "drift_severity_deg": str(severity),
                    "final_angle_error_deg": f"{final_err:.1f}",
                    "final_kl_divergence": f"{final_kl:.2f}" if final_kl is not None else "n/a",
                },
            ),
        ))

    
        mlmodel_entity = client.entities.get(model_urn)
        mlmodel_entity.add_deployment(deployment_urn)
        client.entities.update(mlmodel_entity)

        print(f"Ingested {patient_id}: dataset, {N_CHANNELS} features, model, deployment.")

    print("\nDone. Open the DataHub UI and search 'synaptoflow' to browse the graph.")


if __name__ == "__main__":
    main()