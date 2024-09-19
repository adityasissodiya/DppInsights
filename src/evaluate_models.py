import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define directories
MODEL_DIR = 'models/saved_models/'
RESULTS_DIR = 'results/metrics/'
VISUALIZATIONS_DIR = 'results/visualizations/'

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Load test data
df_test = pd.read_csv('data/processed/pseudo_product_data.csv')
X_test = df_test.drop(columns=['EndOfLifeOption'])  # Adjust 'target_column' to actual target column
y_test = df_test['EndOfLifeOption']

# Helper function to save evaluation metrics
def save_metrics(metrics, model_name):
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(RESULTS_DIR, f'{model_name}_metrics.csv'), index=False)

# Helper function to plot and save confusion matrix
def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, f'{model_name}_confusion_matrix.png'))
    plt.close()

# Function to evaluate classification models
def evaluate_classification_model(model_name, model_path):
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    metrics = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    }

    # Save metrics
    save_metrics(metrics, model_name)

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, model_name)

    print(f"Classification evaluation for {model_name} completed.")

# Function to evaluate regression models
def evaluate_regression_model(model_name, model_path):
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'Model': model_name,
        'MSE': mse,
        'R2': r2
    }

    # Save metrics
    save_metrics(metrics, model_name)

    print(f"Regression evaluation for {model_name} completed.")

# Main function to evaluate all models
if __name__ == "__main__":
    # Evaluate classification models
    evaluate_classification_model('Decision Tree', os.path.join(MODEL_DIR, 'decision_tree.pkl'))
    evaluate_classification_model('Random Forest', os.path.join(MODEL_DIR, 'random_forest.pkl'))
    evaluate_classification_model('SVM', os.path.join(MODEL_DIR, 'svm.pkl'))
    evaluate_classification_model('Neural Network Classifier', os.path.join(MODEL_DIR, 'neural_network_classifier.pkl'))

    # Evaluate regression models
    evaluate_regression_model('Random Forest Regressor', os.path.join(MODEL_DIR, 'random_forest_reg.pkl'))
    evaluate_regression_model('Neural Network Regressor', os.path.join(MODEL_DIR, 'neural_network_reg.pkl'))

    # Add more models if needed, and extend to clustering and anomaly detection evaluations if necessary.
