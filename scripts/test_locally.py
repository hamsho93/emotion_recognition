import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from azureml.core import Workspace
from src.train import train
from src.utils import load_audio_from_blob

def print_versions():
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.executable}")
    print(f"NumPy version: {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    
    import azureml.core
    print(f"Azure ML version: {azureml.core.VERSION}")

def test_azure_connection():
    try:
        ws = Workspace.from_config()
        print(f"Successfully connected to workspace: {ws.name}")
    except Exception as e:
        print(f"Error connecting to Azure workspace: {e}")

def test_blob_storage():
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        print("AZURE_STORAGE_CONNECTION_STRING is not set")
        return
 
    container_name = "dev"  # Update this to match your actual container name
    try:
        from azure.storage.blob import BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        blobs = list(container_client.list_blobs())
        print(f"Successfully connected to blob storage. Found {len(blobs)} blobs in container '{container_name}'.")
    except Exception as e:
        print(f"Error accessing blob storage: {e}")

def test_train_function():
    try:
        train(
            data_path="https://emotiondevblob.blob.core.windows.net/dev",
            model_type="test_model",
            num_epochs=1,
            batch_size=32,
            learning_rate=0.001
        )
        print("Training function completed successfully")
    except Exception as e:
        print(f"Error in training function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting local tests...")
    print_versions()
    print("\nTesting Azure connection:")
    test_azure_connection()
    print("\nTesting blob storage access:")
    test_blob_storage()
    print("\nTesting train function:")
    test_train_function()
    print("\nLocal tests completed.")