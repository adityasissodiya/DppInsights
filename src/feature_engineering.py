import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_features(X_train, X_test, numerical_features):
    """
    Scale numerical features using StandardScaler.
    
    :param X_train: Training set
    :param X_test: Testing set
    :param numerical_features: List of numerical columns to scale
    :return: Scaled training and testing sets
    """
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform both training and testing sets
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

    return X_train_scaled, X_test_scaled

# Main block to execute the functions
if __name__ == "__main__":
    # Load the preprocessed data (use the same function as in data_processing.py)
    from data_processing import load_data, preprocess_data, split_data
    
    df = load_data()

    if df is not None:
        # Preprocess the data
        df_processed = preprocess_data(df)

        # Splitting data for predicting RepairCost (as an example)
        X_train, X_test, y_train, y_test = split_data(df_processed, 'RepairCost')

        # List of numerical features to scale
        numerical_features = ['YearOfManufacture', 'MaterialPlastic', 'MaterialMetal', 'MaterialGlass', 
                              'RecyclabilityScore', 'CO2Emissions', 'RepairFrequency', 'RepairabilityScore']

        # Scale the features
        X_train_scaled, X_test_scaled = scale_features(X_train, X_test, numerical_features)

        print("Feature scaling complete!")
        print(f"Scaled training data shape: {X_train_scaled.shape}")
        print(f"Scaled testing data shape: {X_test_scaled.shape}")
