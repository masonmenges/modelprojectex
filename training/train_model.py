"""
Basic Model Training Example

This script demonstrates a simple workflow for training a machine learning model.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification
import joblib
import os
from prefect import flow, task, get_run_logger


@task
def load_data(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, n_classes=2, random_state=42):
    """
    Load or generate training data.
    In a real scenario, this would load data from files or databases.

    Args:
        n_samples: Number of samples to generate
        n_features: Total number of features
        n_informative: Number of informative features
        n_redundant: Number of redundant features
        n_classes: Number of classes
        random_state: Random seed for reproducibility
    """
    logger = get_run_logger()
    logger.info("Starting data loading task")
    logger.info(f"Generating dataset with parameters: n_samples={n_samples}, n_features={n_features}, n_classes={n_classes}")

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        random_state=random_state
    )

    logger.info(f"Data loaded successfully: shape={X.shape}, classes={np.unique(y)}")
    logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


@task
def prepare_data(X, y, test_size=0.2):
    """
    Split data into training and testing sets.
    """
    logger = get_run_logger()
    logger.info(f"Starting data preparation task")
    logger.info(f"Split ratio: {100-test_size*100:.1f}% train, {test_size*100:.1f}% test")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Testing set: {X_test.shape[0]} samples")
    logger.info(f"Train class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    logger.info(f"Test class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    return X_train, X_test, y_train, y_test


@task
def train_model(X_train, y_train):
    """
    Train the machine learning model.
    """
    logger = get_run_logger()
    logger.info("Starting model training task")

    model_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42,
        'n_jobs': -1
    }
    logger.info(f"Training Random Forest with parameters: {model_params}")

    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)

    logger.info("Model training complete!")
    logger.info(f"Model type: {type(model).__name__}")
    logger.info(f"Number of features: {model.n_features_in_}")
    logger.info(f"Number of classes: {model.n_classes_}")

    return model


@task
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test data.
    """
    logger = get_run_logger()
    logger.info("Starting model evaluation task")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f"Model Accuracy: {accuracy:.4f}")
    logger.info(f"Predictions generated for {len(y_pred)} samples")

    # Log classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    logger.info("Classification Metrics:")
    for class_label, metrics in report.items():
        if isinstance(metrics, dict):
            logger.info(f"  Class {class_label}: precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1-score={metrics['f1-score']:.4f}")

    # Also print for console visibility
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return accuracy


@task
def save_model(model, filepath="trained_model.joblib"):
    """
    Save the trained model to disk.
    """
    logger = get_run_logger()
    logger.info("Starting model saving task")
    logger.info(f"Saving model to: {filepath}")

    joblib.dump(model, filepath)

    # Verify file was created and log file size
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath)
        logger.info(f"Model saved successfully! File size: {file_size / 1024:.2f} KB")
    else:
        logger.error(f"Failed to save model to {filepath}")

    return filepath


@flow
def model_training_qa(test_size=0.2, n_samples=1000, n_features=20, n_informative=15, n_redundant=5, n_classes=2, random_state=42):
    """
    Model training QA flow function.

    Args:
        test_size: Proportion of dataset to use for testing (default: 0.2)
        n_samples: Number of samples to generate (default: 1000)
        n_features: Total number of features (default: 20)
        n_informative: Number of informative features (default: 15)
        n_redundant: Number of redundant features (default: 5)
        n_classes: Number of classes (default: 2)
        random_state: Random seed for reproducibility (default: 42)
    """
    logger = get_run_logger()

    print("="*50)
    print("Starting Model Training Pipeline")
    print("="*50)

    logger.info("="*50)
    logger.info("Starting Model Training QA Flow")
    logger.info("="*50)
    logger.info(f"Flow parameters: test_size={test_size}, n_samples={n_samples}, n_features={n_features}")

    # Load data
    logger.info("Step 1: Loading data")
    X, y = load_data(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        random_state=random_state
    )

    # Prepare data
    logger.info("Step 2: Preparing data")
    X_train, X_test, y_train, y_test = prepare_data(X, y, test_size=test_size)

    # Train model
    logger.info("Step 3: Training model")
    model = train_model(X_train, y_train)

    # Evaluate model
    logger.info("Step 4: Evaluating model")
    accuracy = evaluate_model(model, X_test, y_test)

    # Save model
    logger.info("Step 5: Saving model")
    model_path = os.path.join(os.path.dirname(__file__), "trained_model.joblib")
    saved_path = save_model(model, model_path)

    logger.info("="*50)
    logger.info(f"Training Pipeline Complete! Final accuracy: {accuracy:.4f}")
    logger.info(f"Model saved to: {saved_path}")
    logger.info("="*50)

    print("\n" + "="*50)
    print("Training Pipeline Complete!")
    print("="*50)

    return {"accuracy": accuracy, "model_path": saved_path}


if __name__ == "__main__":
    model_training_qa()
