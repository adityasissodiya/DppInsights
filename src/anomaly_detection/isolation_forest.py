import sys
import os
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing import load_data, preprocess_data
from feature_engineering import scale_features

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

# Helper function to save visualizations (e.g., PCA plot)
def save_visualization(plot_name):
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, f'{plot_name}.png'))
    plt.close()

def run_isolation_forest(X):
    """
    Apply Isolation Forest for anomaly detection.
    
    :param X: Feature matrix
    :return: Isolation Forest model and anomaly labels
    """
    iso_forest = IsolationForest(contamination=0.05, random_state=42)  # Assume 5% of the data is anomalous
    anomaly_labels = iso_forest.fit_predict(X)  # -1 means anomaly, 1 means normal
    return iso_forest, anomaly_labels

def plot_anomalies(X, anomaly_labels):
    """
    Plot anomalies in the dataset using PCA for dimensionality reduction.
    
    :param X: Feature matrix
    :param anomaly_labels: Labels indicating which points are anomalies
    :return: None (displays a plot)
    """
    from sklearn.decomposition import PCA
    
    # Reduce dimensions to 2D for visualization using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=anomaly_labels, cmap='coolwarm', s=50, alpha=0.8)
    plt.title("Isolation Forest - Anomaly Detection")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label='Anomaly Label')
    save_visualization('isolation_forest_anomalies')  # Save the visualization as PNG

# Main block to execute the functions
if __name__ == "__main__":
    # Load and preprocess the data
    df = load_data()

    if df is not None:
        # Preprocess and scale data
        df_processed = preprocess_data(df)
        
        # List of numerical features to consider for anomaly detection
        numerical_features = ['YearOfManufacture', 'MaterialPlastic', 'MaterialMetal', 'MaterialGlass', 
                              'RecyclabilityScore', 'CO2Emissions', 'RepairFrequency', 'RepairabilityScore']

        # Scale the features
        X_scaled, _ = scale_features(df_processed[numerical_features], df_processed[numerical_features], numerical_features)

        # Run Isolation Forest
        iso_forest, anomaly_labels = run_isolation_forest(X_scaled)

        # Plot and save the anomalies
        plot_anomalies(X_scaled, anomaly_labels)

        # Output and save the number of anomalies detected
        num_anomalies = sum(anomaly_labels == -1)
        print(f"Number of anomalies detected: {num_anomalies}")

        # Save metrics
        metrics = {
            'Model': 'Isolation Forest',
            'Number of Anomalies': num_anomalies
        }
        save_metrics(metrics, 'isolation_forest')
