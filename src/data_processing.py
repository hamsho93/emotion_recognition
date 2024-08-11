import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
from transformers import Wav2Vec2Processor, BertTokenizer

wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def load_and_preprocess_audio(file_path, target_sr=16000):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    return waveform.squeeze().numpy()

def extract_wav2vec_features(audio):
    input_values = wav2vec_processor(audio, sampling_rate=16000, return_tensors="pt").input_values
    return input_values.squeeze()

def extract_bert_features(text):
    inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze()

class EmotionDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        audio = load_and_preprocess_audio(item['file_path'])
        audio_features = extract_wav2vec_features(audio)
        text_ids, text_mask = extract_bert_features(item['transcription'])
        label = torch.tensor(item['emotion_label'], dtype=torch.long)
        return {
            'audio_features': audio_features,
            'text_ids': text_ids,
            'text_mask': text_mask,
            'label': label
        }

def preprocess_data(df, data_path):
    df['file_path'] = df['file_name'].apply(lambda x: f"{data_path}/{x}")
    df['emotion_label'] = df['emotion'].map({
        'neutral': 0, 'calm': 1, 'happy': 2, 'sad': 3,
        'angry': 4, 'fearful': 5, 'disgust': 6, 'surprised': 7
    })
    return df