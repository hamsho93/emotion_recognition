import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import mlflow
import mlflow.sklearn
import logging
import os
import joblib

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)


logging.basicConfig(level=logging.INFO)

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model, model_name):
    """
    Train and evaluate a machine learning model, logging results with MLflow.
    """
    with mlflow.start_run(run_name=model_name):
        # Log model parameters
        mlflow.log_params(model.get_params())
        
        # Train the model
        if isinstance(model, MLPClassifier):
            model = OneVsRestClassifier(model)
        model.fit(X_train, y_train)

        # Save the trained model
        models_path = os.path.join(current_dir, 'models', f'{model_name}_model.joblib')
        joblib.dump(model, models_path)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate AUC
        n_classes = len(np.unique(y_train))
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
        else:
            # Some models like SVM don't have predict_proba
            y_score = model.decision_function(X_test)

        # Ensure y_score has the correct shape
        if y_score.shape[1] > n_classes:
            y_score = y_score[:, :n_classes]

        # Try to calculate AUC, but handle the case where it's not possible
        try:
            auc = roc_auc_score(y_test_bin, y_score, average='weighted', multi_class='ovr')
        except ValueError:
            print(f"Warning: Unable to calculate AUC for {model_name}. This may be due to having only one class in the test set.")
            auc = None

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        if auc is not None:
            mlflow.log_metric("auc", auc)
        
        # Generate classification report
        clf_report = classification_report(y_test, y_pred, output_dict=True)
        for label, metrics in clf_report.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric_name}", value)
        
        # Generate and save confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        cm_plot_path = os.path.join(current_dir, 'plots', f'confusion_matrix_{model_name}.png')
        plt.savefig(cm_plot_path)
        mlflow.log_artifact(cm_plot_path)
        plt.close()
        
        # Log the model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"\n{model_name} Results:")
        print(classification_report(y_test, y_pred))
        print(f"{model_name} Accuracy: {accuracy:.4f}")
        if auc is not None:
            print(f"{model_name} AUC: {auc:.4f}")
        else:
            print(f"{model_name} AUC: Not available")
        
        return accuracy, auc

def main():
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Set the experiment name
    mlflow.set_experiment("Emotion Recognition V2")

    # Load reduced features
    features_path = os.path.join(current_dir, 'data', 'reduced_features.npy')
    X_reduced = np.load(features_path)
    
    # Load metadata
    metadata_path = os.path.join(current_dir, 'data', 'metadata.csv')
    metadata_df = pd.read_csv(metadata_path)
    
    # Prepare data for modeling
    le = LabelEncoder()
    y = le.fit_transform(metadata_df['emotion'])
    n_classes = len(np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42, stratify=y)

    # Log dataset info
    with mlflow.start_run(run_name="Dataset Info"):
        mlflow.log_param("n_samples", len(y))
        mlflow.log_param("n_features", X_reduced.shape[1])
        mlflow.log_param("n_classes", n_classes)

    # Train and evaluate models
    models = [
        (RandomForestClassifier(n_estimators=100, random_state=42), "RandomForest"),
        (SVC(kernel='rbf', random_state=42, probability=True), "SVM"),
        (MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42), "NN_1Layer"),
        (MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42), "NN_2Layers"),
        (MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=1000, random_state=42), "NN_3Layers"),
        (MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=1000, random_state=42), "NN_2Layers_Wide"),
    ]

    results = []
    for model, name in models:
        accuracy, auc = train_and_evaluate_model(X_train, X_test, y_train, y_test, model, name)
        results.append((name, accuracy, auc))



    print("\nModel training and evaluation completed.")
    print("\nSummary of Results:")
    for name, accuracy, auc in results:
        if auc is not None:
            print(f"{name}: Accuracy = {accuracy:.4f}, AUC = {auc:.4f}")
        else:
            print(f"{name}: Accuracy = {accuracy:.4f}, AUC = Not available")

    print(f"\nMLflow runs can be viewed at {mlflow.get_tracking_uri()}")

if __name__ == "__main__":
    main()