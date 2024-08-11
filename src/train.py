import os
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, balanced_accuracy_score
from azureml.core import Run
from src.utils import load_audio_from_blob
from src.model import EmotionClassifier
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)

def train(data_path, model_type, num_epochs, batch_size, learning_rate):
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is not set")
    
    container_name = data_path.split("/")[-1]
    X, y = load_audio_from_blob(connection_string, container_name)
    
    logging.info(f"Loaded data shape: X: {X.shape}, y: {y.shape}")
    
    # Convert string labels to numerical values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    # Create dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize the model
    input_size = X.shape[1]
    num_classes = len(label_encoder.classes_)
    model = EmotionClassifier(input_size, num_classes)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Get the run context
    run = Run.get_context()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        run.log("train_loss", avg_train_loss)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                all_preds.extend(predicted.numpy())
                all_labels.extend(batch_y.numpy())
        
        avg_val_loss = val_loss / len(test_loader)
        accuracy = correct / total
        run.log("val_loss", avg_val_loss)
        run.log("val_accuracy", accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.4f}")

    # Final evaluation
    print("\nFinal Evaluation:")
    print(classification_report(all_labels, all_preds))
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    run.log("balanced_accuracy", balanced_acc)

    print("Training completed.")

if __name__ == "__main__":
    train(
        data_path="https://emotiondevblob.blob.core.windows.net/dev",
        model_type="emotion_classifier",
        num_epochs=20,
        batch_size=32,
        learning_rate=0.001
    )