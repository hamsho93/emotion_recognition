import os
import io
import numpy as np
import torch
import torchaudio
import librosa
from azure.storage.blob import BlobServiceClient
from transformers import Wav2Vec2Processor, Wav2Vec2Model, BertTokenizer, BertModel
import logging
logging.basicConfig(level=logging.INFO)

# Load the Wav2Vec2 processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# Load BERT tokenizer and model
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def load_and_resample_audio(audio_data, target_sr=16000):
    waveform, sample_rate = torchaudio.load(io.BytesIO(audio_data))
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    return waveform.squeeze().numpy()

def extract_wav2vec_features(audio, max_length=250):
    input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        outputs = wav2vec_model(input_values)
    features = outputs.last_hidden_state.squeeze().numpy()
    if features.shape[0] > max_length:
        return features[:max_length, :]
    elif features.shape[0] < max_length:
        pad_width = ((0, max_length - features.shape[0]), (0, 0))
        return np.pad(features, pad_width, mode='constant')
    return features

def extract_bert_features(text, max_length=128):
    inputs = bert_tokenizer(text, return_tensors="pt", max_length=max_length, padding="max_length", truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.squeeze(0).numpy()

def load_audio_from_blob(connection_string, container_name):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    data = []
    labels = []
    total_blobs = 0
    processed_blobs = 0
    skipped_blobs = 0

    max_wav2vec_length = 250
    max_mfcc_length = 200
    max_bert_length = 128

    for blob in container_client.list_blobs():
        total_blobs += 1
        parts = blob.name.split('/')
        if len(parts) < 2:
            skipped_blobs += 1
            logging.warning(f"Skipping blob: {blob.name} - Invalid format")
            continue
        
        try:
            file_name = parts[-1]
            emotion = extract_emotion_from_filename(file_name)
            
            blob_client = container_client.get_blob_client(blob)
            audio_data = blob_client.download_blob().readall()
            
            # Process audio
            audio = load_and_resample_audio(audio_data)
            
            # Extract features
            wav2vec_features = extract_wav2vec_features(audio, max_length=max_wav2vec_length)
            mfcc_features = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)
            mfcc_features = pad_or_truncate(mfcc_features, max_mfcc_length, axis=1)
            
            # For BERT features, we'll use a placeholder. In a real scenario, you'd need the actual transcription.
            bert_features = extract_bert_features("placeholder text", max_length=max_bert_length)
            
            # Flatten and combine features
            wav2vec_flat = wav2vec_features.flatten()
            mfcc_flat = mfcc_features.flatten()
            bert_flat = bert_features.flatten()
            
            combined_features = np.concatenate([wav2vec_flat, mfcc_flat, bert_flat])
            
            data.append(combined_features)
            labels.append(emotion)
            processed_blobs += 1
        except Exception as e:
            skipped_blobs += 1
            logging.error(f"Error processing blob: {blob.name} - {str(e)}")

    logging.info(f"Total blobs: {total_blobs}")
    logging.info(f"Processed blobs: {processed_blobs}")
    logging.info(f"Skipped blobs: {skipped_blobs}")

    return np.array(data), np.array(labels)

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

def extract_emotion_from_filename(filename):
    emotion_code = filename.split('-')[2]
    emotion_map = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
    return emotion_map.get(emotion_code, 'unknown')
