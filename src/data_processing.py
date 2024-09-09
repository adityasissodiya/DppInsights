import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path='data/processed/pseudo_product_data.csv'):
    """
    Load the dataset from the CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully!")
    except FileNotFoundError:
        print(f"File not found at {file_path}. Please check the path.")
        return None
    return df

def preprocess_data(df):
    """
    Preprocess the dataset.
    - Handle missing values (if any)
    - Encode categorical variables
    - Drop non-numeric columns not useful for the model
    """
    # Drop unnecessary non-numeric columns
    df = df.drop(columns=['ProductID', 'ProductName'])

    # Example of encoding categorical variables using one-hot encoding
    df_encoded = pd.get_dummies(df, columns=['ProductType', 'EndOfLifeOption', 'SupplierLocation', 'Manufacturer'], drop_first=True)

    return df_encoded

def split_data(df, target_column):
    """
    Split the dataset into training and testing sets.
    :param df: DataFrame to be split
    :param target_column: The name of the target column for prediction
    :return: X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test

# Main block to execute the functions
if __name__ == "__main__":
    # Load the data
    df = load_data()

    if df is not None:
        # Preprocess the data
        df_processed = preprocess_data(df)

        # Example: Splitting data for predicting RepairCost
        X_train, X_test, y_train, y_test = split_data(df_processed, 'RepairCost')

        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")
