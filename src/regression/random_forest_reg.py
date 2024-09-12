import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from data_processing import load_data, preprocess_data, split_data
from feature_engineering import scale_features

def train_random_forest_regressor(X_train, y_train):
    """
    Train a Random Forest regressor on the training data.
    
    :param X_train: Training features
    :param y_train: Target values for training
    :return: Trained Random Forest Regressor model
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the Random Forest Regressor on the test set.
    
    :param model: Trained Random Forest Regressor model
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

        # Train the Random Forest Regressor model
        model = train_random_forest_regressor(X_train_scaled, y_train)

        # Evaluate the model
        evaluate_model(model, X_test_scaled, y_test)
