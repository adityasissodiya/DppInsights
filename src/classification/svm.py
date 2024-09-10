import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from data_processing import load_data, preprocess_data, split_data
from feature_engineering import scale_features

def train_svm(X_train, y_train):
    """
    Train an SVM classifier on the training data.
    
    :param X_train: Training features
    :param y_train: Target values for training
    :return: Trained SVM model
    """
    model = SVC(kernel='linear', random_state=42)  # Linear kernel as a default
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the SVM model on the test set.
    
    :param model: Trained model
    :param X_test: Testing features
    :param y_test: True target values
    :return: None (prints evaluation results)
    """
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    
    # Print detailed classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Main block to execute the functions
if __name__ == "__main__":
    # Load and preprocess the data
    df = load_data()

    if df is not None:
        # Preprocess and scale data for classification
        df_processed = preprocess_data(df)
        
        # Split the data, predicting 'EndOfLifeOption_Recyclable' (binary classification)
        X_train, X_test, y_train, y_test = split_data(df_processed, 'EndOfLifeOption_Recyclable')
        
        # List of numerical features (already scaled in feature_engineering)
        numerical_features = ['YearOfManufacture', 'MaterialPlastic', 'MaterialMetal', 'MaterialGlass', 
                              'RecyclabilityScore', 'CO2Emissions', 'RepairFrequency', 'RepairabilityScore']

        # Scale the features
        X_train_scaled, X_test_scaled = scale_features(X_train, X_test, numerical_features)

        # Train the SVM model
        model = train_svm(X_train_scaled, y_train)

        # Evaluate the model
        evaluate_model(model, X_test_scaled, y_test)
