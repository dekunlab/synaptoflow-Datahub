import os
import time
import warnings
warnings.filterwarnings("ignore")

import datahub.emitter.mce_builder as builder
from datahub.api.entities.dataprocess.dataprocess_instance import (
    DataProcessInstance,
    InstanceRunResult,
)
from datahub.emitter.mcp_builder import ContainerKey
from datahub.emitter.rest_emitter import DatahubRestEmitter
from datahub.metadata.schema_classes import DataProcessTypeClass
from datahub.utilities.urns.dataset_urn import DatasetUrn
from datahub.sdk import DataHubClient
from datahub.metadata.urns import MlModelUrn

PLATFORM = "synaptoflow"


DRIFT_SEVERITIES = {
    "patient_01": 0.0, "patient_02": 0.0, "patient_03": 0.05, "patient_04": 0.05,
    "patient_05": 0.3, "patient_06": 0.5, "patient_07": 0.7, "patient_08": 0.9,
    "patient_09": 1.1, "patient_10": 1.4, "patient_11": 1.8, "patient_12": 2.3,
}


def main():
    gms_server = os.environ.get("DATAHUB_GMS_URL", "http://localhost:8080")
    token = os.environ.get("DATAHUB_GMS_TOKEN")

    emitter = DatahubRestEmitter(gms_server=gms_server, token=token)
    client = DataHubClient.from_env()

    
    experiment_container = ContainerKey(
        platform=f"urn:li:dataPlatform:{PLATFORM}",
        name="synaptoflow-calibration-experiments",
        env="PROD",
    )

    for patient_id in sorted(DRIFT_SEVERITIES):
        dataset_urn = builder.make_dataset_urn(
            name=f"raw_neural_stream_{patient_id}", platform=PLATFORM, env="PROD"
        )
        model_urn = builder.make_ml_model_urn(
            model_name=f"decoder_{patient_id}", platform=PLATFORM, env="PROD"
        )

        training_run = DataProcessInstance.from_container(
            container_key=experiment_container,
            id=f"calibration_{patient_id}",
        )
        training_run.type = DataProcessTypeClass.BATCH_AD_HOC
        training_run.properties = {
            "patient_id": patient_id,
            "drift_severity_deg": str(DRIFT_SEVERITIES[patient_id]),
            "decoder_type": "population_vector",
        }
        training_run.inlets = [DatasetUrn.create_from_string(dataset_urn)]
    
        training_run.outlets = [MlModelUrn.from_string(model_urn)]

        now = int(time.time() * 1000)
        training_run.emit_process_start(
            emitter=emitter,
            start_timestamp_millis=now,
            attempt=1,
            emit_template=False,
            materialize_iolets=True,
        )
        training_run.emit_process_end(
            emitter=emitter,
            end_timestamp_millis=now,
            result=InstanceRunResult.SUCCESS,
            result_type="synaptoflow",
            attempt=1,
            start_timestamp_millis=now,
        )

        # Link the run into the model's own lineage (trainingJobs field)
        mlmodel_entity = client.entities.get(
            MlModelUrn(platform=PLATFORM, name=f"decoder_{patient_id}")
        )
        mlmodel_entity.add_training_job(str(training_run.urn))
        client.entities.update(mlmodel_entity)

        print(f"Created calibration training run for {patient_id}: {training_run.urn}")

    print("\nDone. Each decoder's lineage now traces back through its calibration run to its raw dataset.")


if __name__ == "__main__":
    main()