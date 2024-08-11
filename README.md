Certainly! Here's a README.md file that describes how to execute the code locally and in Azure:

```markdown
# Emotion Recognition System

This project implements an AI-powered emotion recognition system that combines audio and text analysis to classify emotions in speech.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Local Execution](#local-execution)
4. [Azure Deployment](#azure-deployment)
5. [Directory Structure](#directory-structure)
6. [Usage](#usage)
7. [MLflow Tracking](#mlflow-tracking)
8. [Contributing](#contributing)
9. [License](#license)

## Project Overview

This emotion recognition system uses a multi-modal approach, combining Wav2Vec2 for audio processing, MFCC for additional audio features, and BERT for text analysis. The system is trained on the RAVDESS dataset and can classify speech into eight emotion categories.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/emotion-recognition-system.git
   cd emotion-recognition-system
   ```

2. Create a virtual environment:
   ```
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Local Execution

To run the project locally:

1. Ensure you have the RAVDESS dataset downloaded and placed in the `data/` directory.

2. Run the data preparation script:
   ```
   python scripts/prepare_data.py
   ```

3. Train the model:
   ```
   python scripts/train_model.py
   ```

4. Make predictions:
   ```
   python scripts/predict.py path/to/audio/file.wav
   ```

## Azure Deployment

To deploy and run the project on Azure:

1. Set up an Azure account and install the Azure CLI.

2. Login to Azure:
   ```
   az login
   ```

3. Create an Azure Machine Learning workspace:
   ```
   az ml workspace create -n myworkspace -g myresourcegroup
   ```

4. Set up your compute target:
   ```
   az ml computetarget create amlcompute -n cpu-cluster --min-nodes 0 --max-nodes 4
   ```

5. Upload your data to Azure Blob Storage:
   ```
   az storage blob upload-batch -d mycontainer -s data/ --account-name mystorageaccount
   ```

6. Modify the `azure/train.py` script to use your Azure datasets and compute target.

7. Submit the training job:
   ```
   az ml run submit-script -e myexperiment -d azure/train.py
   ```

8. Deploy the model as a web service:
   ```
   az ml model deploy -n myservice -m mymodel:1 --ic azure/inference_config.yml --dc azure/deployment_config.yml
   ```

## Directory Structure

```
emotion-recognition-system/
│
├── data/                  # Dataset files
├── models/                # Saved models
├── notebooks/             # Jupyter notebooks for analysis
├── scripts/               # Python scripts for data preparation, training, and prediction
├── utils/                 # Utility functions and classes
├── azure/                 # Azure-specific configuration and scripts
├── tests/                 # Unit tests
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Usage

To use the trained model for prediction:

```python
from scripts.predict import predict_emotion

emotion = predict_emotion('path/to/audio/file.wav')
print(f"Predicted emotion: {emotion}")
```

## MLflow Tracking

This project uses MLflow for experiment tracking. To view the MLflow UI locally:

1. Run an MLflow server:
   ```
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
   ```

2. Open a web browser and navigate to `http://localhost:5000`

For Azure deployments, MLflow is integrated with Azure ML. You can view experiments in the Azure ML studio.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
```

This README provides a comprehensive guide for both local execution and Azure deployment. It covers installation, usage, directory structure, and MLflow tracking. You may need to adjust some details (like GitHub repository URL, Azure resource names, etc.) to match your specific project setup.