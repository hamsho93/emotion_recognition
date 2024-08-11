import numpy as np
import pandas as pd
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model, BertTokenizer, BertModel
import joblib
import azure.cognitiveservices.speech as speechsdk
import os
import logging
from datetime import datetime


# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
# Set up logging
log_dir = current_dir + '/log/'
log_file = os.path.join(log_dir, f'emotion_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Azure Speech SDK configuration
speech_key = "6d5d6cb1ed08442c80b45fd7f9a80734"
service_region = "eastus"
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

# Load the Wav2Vec2 processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# Load BERT tokenizer and model
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Load PCA model
pca_path = os.path.join(current_dir, 'utils', 'pca_model.joblib')
loaded_pca = joblib.load(pca_path)

# Load the final model (assuming you're using the RandomForest model)
model_path = os.path.join(current_dir, 'models', 'NN_3Layers_model.joblib')
final_model = joblib.load(model_path)

# Load the LabelEncoder
le_path = os.path.join(current_dir, 'utils', 'target_encoder.joblib')
le = joblib.load(le_path)

def transcribe_audio_with_diarization(audio_file):
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
    transcriber = speechsdk.transcription.ConversationTranscriber(speech_config=speech_config, audio_config=audio_config)
    done = False
    transcription = []
    def stop_cb(evt):
        nonlocal done
        done = True
    def transcribed_cb(evt):
        nonlocal transcription
        transcription.append(evt)
    transcriber.transcribed.connect(transcribed_cb)
    transcriber.session_stopped.connect(stop_cb)
    transcriber.canceled.connect(stop_cb)
    transcriber.start_transcribing_async()
    while not done:
        pass
    transcriber.stop_transcribing_async()
    return transcription

def extract_wav2vec_features(audio, max_length=250):
    input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        outputs = wav2vec_model(input_values)
    features = outputs.last_hidden_state.squeeze().numpy()
    return pad_or_truncate(features, max_length)

def extract_mfcc(audio, sr=16000, n_mfcc=13):
    return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

def extract_bert_features(text, max_length=128):
    inputs = bert_tokenizer(text, return_tensors="pt", max_length=max_length, padding="max_length", truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.squeeze(0).numpy()

def pad_or_truncate(array, target_length, axis=0):
    if array.shape[axis] > target_length:
        slices = [slice(None)] * array.ndim
        slices[axis] = slice(0, target_length)
        return array[tuple(slices)]
    elif array.shape[axis] < target_length:
        pad_width = [(0, 0)] * array.ndim
        pad_width[axis] = (0, target_length - array.shape[axis])
        return np.pad(array, pad_width, mode='constant')
    return array

def engineer_features(audio, transcription):
    max_wav2vec_length = 250
    max_mfcc_length = 200
    max_bert_length = 128
    
    wav2vec_features = extract_wav2vec_features(audio, max_length=max_wav2vec_length)
    mfcc_features = pad_or_truncate(extract_mfcc(audio), max_mfcc_length, axis=1)
    bert_features = extract_bert_features(transcription, max_length=max_bert_length)
    
    combined_features = np.concatenate([
        wav2vec_features.flatten(),
        mfcc_features.flatten(),
        bert_features.flatten(),
    ])
    
    return combined_features

def predict_emotion(audio_file):
    logging.info(f"Processing audio file: {audio_file}")
    # Load audio file
    audio, sr = librosa.load(audio_file, sr=16000)
    
    # Transcribe audio
    transcription = transcribe_audio_with_diarization(audio_file)
    
    emotions = []
    for event in transcription:
        text = event.result.text
        # Engineer features for this line
        features = engineer_features(audio, text)
        
        # Reshape features to match PCA input shape
        features = features.reshape(1, -1)
        
        # Apply PCA transformation
        reduced_features = loaded_pca.transform(features)
        
        # Make prediction
        prediction = final_model.predict(reduced_features)
        
        # Convert prediction to emotion label
        emotion = le.inverse_transform(prediction)[0]
        
        emotions.append((text, emotion))
        logging.info(f"Line: {text} | Emotion: {emotion}")
        print(f"Line: {text} | Emotion: {emotion}")
    
    # Overall emotion (you can modify this logic if needed)
    overall_emotion = max(set(emotions), key=emotions.count)
    logging.info(f"Overall predicted emotion: {overall_emotion}")
    return overall_emotion

if __name__ == "__main__":
    audio_file_path = os.path.join(current_dir, 'oot_audio', 'sample_customer_service_cal.wav')
    if not os.path.exists(audio_file_path):
        print("Error: The specified audio file does not exist.")
        logging.error(f"Audio file not found: {audio_file_path}")
    else:
        predicted_emotion = predict_emotion(audio_file_path)
        print(f"Overall predicted emotion: {predicted_emotion}")
        logging.info(f"Emotion prediction completed for {audio_file_path}")