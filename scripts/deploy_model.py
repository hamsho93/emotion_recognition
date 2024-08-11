# deploy_model.py

import os
import mlflow
from azureml.core import Workspace, Model
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core.authentication import AzureCliAuthentication
from azureml.exceptions import WebserviceException

def main():
    # Set up Azure ML workspace
    cli_auth = AzureCliAuthentication()
    ws = Workspace.from_config(auth=cli_auth)

    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

    # Set the experiment name
    experiment_name = 'emotion_recognition_experiment'
    mlflow.set_experiment(experiment_name)

    # Get the latest run
    latest_run = mlflow.search_runs(experiment_names=[experiment_name], order_by=["start_time desc"], max_results=1).iloc[0]

    # Get the model path
    model_path = latest_run.artifact_uri + "/model"

    # Register the model
    model = Model.register(workspace=ws,
                           model_path=model_path,
                           model_name="emotion_recognition_model",
                           tags={"run_id": latest_run.run_id},
                           description="Emotion recognition model trained on RAVDESS dataset")

    print(f"Model registered: {model.name}, version {model.version}")

    # Define inference configuration
    inference_config = InferenceConfig(
        entry_script="score.py",
        source_directory="./src",
        conda_file="./conda_env.yml"
    )

    # Define deployment configuration
    deployment_config = AciWebservice.deploy_configuration(
        cpu_cores=1,
        memory_gb=1,
        tags={"data": "RAVDESS", "method": "pytorch"},
        description="Emotion recognition model deployment"
    )

    # Deploy the model
    service_name = 'emotion-recognition-service'
    try:
        service = Model.deploy(workspace=ws,
                               name=service_name,
                               models=[model],
                               inference_config=inference_config,
                               deployment_config=deployment_config,
                               overwrite=True)
        service.wait_for_deployment(show_output=True)
    except WebserviceException as e:
        print(f"Deployment failed: {e}")
        return

    print(f"Model deployed successfully. Service URL: {service.scoring_uri}")

    # Test the deployed service
    test_sample = {"audio_data": "base64_encoded_audio_data_here"}
    print("Testing the deployed model with a sample input...")
    print(f"Test result: {service.run(input_data=test_sample)}")

if __name__ == "__main__":
    main()