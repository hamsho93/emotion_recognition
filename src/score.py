# score.py

import os
import json
import torch
from model import EmotionRecognitionModel  # Import your model class

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pth')
    model = EmotionRecognitionModel()  # Initialize your model
    model.load_state_dict(torch.load(model_path))
    model.eval()

def run(raw_data):
    try:
        data = json.loads(raw_data)['audio_data']
        # Preprocess your audio data here
        # Make prediction
        with torch.no_grad():
            output = model(input_data)
        return json.dumps({"prediction": output.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})