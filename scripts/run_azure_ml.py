import os
import mlflow
from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.authentication import InteractiveLoginAuthentication

# Get the path to the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Set up Azure ML workspace
auth = InteractiveLoginAuthentication()
config_path = os.path.join(project_root, "config.json")
ws = Workspace.from_config(path=config_path, auth=auth)

# Set up compute target
compute_name = "emotion-recognition-cluster"

try:
    compute_target = ComputeTarget(workspace=ws, name=compute_name)
    print(f'Found existing compute target. Current status: {compute_target.get_status()}')
except:
    print('Creating a new compute target...')
    provisioning_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS3_v2', 
                                                                min_nodes=0,
                                                                max_nodes=4)
    compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)
    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

print(f'Compute target status: {compute_target.get_status()}')

# Create an environment and add the connection string
env = Environment.from_conda_specification(
    name="audio_env", 
    file_path=os.path.join(project_root, "conda_env.yml")
)

env.environment_variables["AZURE_STORAGE_CONNECTION_STRING"] = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

# Print the environment details
print("Environment details:")
print(f"Name: {env.name}")
print("Conda dependencies:")
for dep in env.python.conda_dependencies.conda_packages:
    print(f"  - {dep}")
print("Pip dependencies:")
for dep in env.python.conda_dependencies.pip_packages:
    print(f"  - {dep}")

# Set up ScriptRunConfig
src = ScriptRunConfig(
    source_directory=project_root,
    script='src/train.py',
    compute_target=compute_target,
    environment=env
)

# Add script arguments
src.arguments = ["--data_path", "https://emotiondevblob.blob.core.windows.net/dev",
                 "--num_epochs", "20",
                 "--batch_size", "64"]

# Create an experiment
experiment = Experiment(workspace=ws, name='emotion_recognition_experiment')

# Submit the experiment
run = experiment.submit(src)
print("Submitted to compute target")
print(f"Run ID: {run.id}")

# Wait for the run to complete
run.wait_for_completion(show_output=True)

print("Training job completed")

# Optionally, you can retrieve and print some metrics
metrics = run.get_metrics()
for metric_name in metrics:
    print(f"{metric_name}: {metrics[metric_name]}")

# If you want to download the output files
run.download_files(prefix="outputs")
print("Output files downloaded")