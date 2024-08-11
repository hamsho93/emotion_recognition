import pytest
import torch
from src.model import EmotionClassifier

def test_emotion_classifier():
    model = EmotionClassifier()
    
    batch_size = 2
    audio_features = torch.randn(batch_size, 100, 768)  # Assuming 100 time steps and 768 features
    text_ids = torch.randint(0, 1000, (batch_size, 50))  # Assuming max 50 tokens
    text_mask = torch.ones(batch_size, 50)
    
    output = model(audio_features, text_ids, text_mask)
    
    assert output.shape == (batch_size, 8)  # 8 emotion classes