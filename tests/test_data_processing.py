import pytest
from src.data_processing import preprocess_data, EmotionDataset

def test_preprocess_data():
    # Create a sample dataframe
    df = pd.DataFrame({
        'file_name': ['audio1.wav', 'audio2.wav'],
        'emotion': ['happy', 'sad']
    })
    
    processed_df = preprocess_data(df, 'data_path')
    
    assert 'file_path' in processed_df.columns
    assert 'emotion_label' in processed_df.columns
    assert processed_df.loc[0, 'emotion_label'] == 2  # 'happy' should be mapped to 2
    assert processed_df.loc[1, 'emotion_label'] == 3  # 'sad' should be mapped to 3

def test_emotion_dataset():
    # Create a sample dataframe
    df = pd.DataFrame({
        'file_path': ['audio1.wav', 'audio2.wav'],
        'transcription': ['Hello', 'World'],
        'emotion_label': [0, 1]
    })
    
    dataset = EmotionDataset(df)
    
    assert len(dataset) == 2
    
    item = dataset[0]
    assert 'audio_features' in item
    assert 'text_ids' in item
    assert 'text_mask' in item
    assert 'label' in item