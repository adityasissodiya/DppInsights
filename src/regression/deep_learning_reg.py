import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from data_processing import load_data, preprocess_data, split_data
from feature_engineering import scale_features

def build_neural_network(input_shape):
    """
    Build a simple feedforward neural network for regression using Keras.
    
    :param input_shape: The shape of the input features
    :return: Compiled neural network model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer for regression (single continuous value)
    ])
    
    # Compile the model with mean squared error as the loss function and Adam optimizer
    model.compile(optimizer='adam', 
                  loss='mean_squared_error', 
                  metrics=['mean_squared_error'])
    
    return model

def train_neural_network(model, X_train, y_train, epochs=20, batch_size=32):
    """
    Train the neural network on the training data.
    
    :param model: The compiled neural network model
    :param X_train: Training features
    :param y_train: Training labels (target values)
    :param epochs: Number of training epochs
    :param batch_size: Size of each training batch
    :return: Trained model
    """
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the neural network model on the test set.
    
    :param model: Trained neural network model
    :param X_test: Testing features
    :param y_test: True target values
    :return: None (prints evaluation results)
    """
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R2): {r2:.2f}")

# Main block to execute the functions
if __name__ == "__main__":
    # Load and preprocess the data
    df = load_data()

    if df is not None:
        # Preprocess and scale data for regression
        df_processed = preprocess_data(df)
        
        # Split the data, predicting 'RepairCost' (you can switch to 'CO2Emissions' if desired)
        X_train, X_test, y_train, y_test = split_data(df_processed, 'RepairCost')
        
        # List of numerical features (already scaled in feature_engineering)
        numerical_features = ['YearOfManufacture', 'MaterialPlastic', 'MaterialMetal', 'MaterialGlass', 
                              'RecyclabilityScore', 'CO2Emissions', 'RepairFrequency', 'RepairabilityScore']

        # Scale the features
        X_train_scaled, X_test_scaled = scale_features(X_train, X_test, numerical_features)

        # Build the neural network model
        model = build_neural_network(input_shape=X_train_scaled.shape[1])

        # Train the model
        model = train_neural_network(model, X_train_scaled, y_train)

        # Evaluate the model
        evaluate_model(model, X_test_scaled, y_test)
