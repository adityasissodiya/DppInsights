import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from matplotlib import pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_processing import load_data, preprocess_data, split_data
from feature_engineering import scale_features
import seaborn as sns

# Define paths for results and visualizations
RESULTS_DIR = 'results/metrics/'
VISUALIZATIONS_DIR = 'results/visualizations/'

# Ensure the directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Helper function to save metrics as CSV
def save_metrics(metrics, model_name):
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(RESULTS_DIR, f'{model_name}_metrics.csv'), index=False)

# Helper function to save visualizations (e.g., confusion matrix)
def save_visualization(plot_name):
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, f'{plot_name}.png'))
    plt.close()

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
    report = classification_report(y_test, y_pred, output_dict=True)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the metrics
    metrics = {
        'Model': 'SVM',
        'Accuracy': accuracy
    }
    save_metrics(metrics, 'svm')

    # Confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.title('SVM Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    save_visualization('svm_confusion_matrix')

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
