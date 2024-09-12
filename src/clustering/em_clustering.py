import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from data_processing import load_data, preprocess_data
from feature_engineering import scale_features

def run_gmm(X, n_components=3):
    """
    Apply Gaussian Mixture Model (GMM) clustering to the data.
    
    :param X: Feature matrix
    :param n_components: The number of clusters/components
    :return: Fitted GMM model and the cluster labels
    """
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    
    # Cluster labels for each point
    cluster_labels = gmm.predict(X)
    
    return gmm, cluster_labels

def evaluate_clustering(X, cluster_labels, gmm):
    """
    Evaluate clustering performance using log-likelihood and silhouette score.
    
    :param X: Feature matrix
    :param cluster_labels: Labels for each point resulting from clustering
    :param gmm: The fitted GMM model
    :return: None (prints evaluation metrics)
    """
    log_likelihood = gmm.score(X)  # Average log-likelihood of the data
    silhouette_avg = silhouette_score(X, cluster_labels)  # Silhouette score
    
    print(f"Log-Likelihood: {log_likelihood:.2f}")
    print(f"Silhouette Score: {silhouette_avg:.2f}")

def plot_clusters(X, cluster_labels, n_components):
    """
    Use PCA to reduce dimensions and plot clusters.
    
    :param X: Feature matrix
    :param cluster_labels: Labels for each point resulting from clustering
    :param n_components: The number of clusters formed
    :return: None (displays a plot)
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.8)
    plt.title(f'EM Clustering (GMM) with {n_components} Components')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster Label')
    plt.show()

# Main block to execute the functions
if __name__ == "__main__":
    # Load and preprocess the data
    df = load_data()

    if df is not None:
        # Preprocess and scale the data for clustering
        df_processed = preprocess_data(df)

        # List of numerical features for clustering
        numerical_features = ['YearOfManufacture', 'MaterialPlastic', 'MaterialMetal', 'MaterialGlass', 
                              'RecyclabilityScore', 'CO2Emissions', 'RepairFrequency', 'RepairabilityScore']

        # Scale the features
        X_scaled, _ = scale_features(df_processed[numerical_features], df_processed[numerical_features], numerical_features)

        # Run GMM clustering
        n_components = 3  # You can experiment with different values of k
        gmm, cluster_labels = run_gmm(X_scaled, n_components)

        # Evaluate the clustering
        evaluate_clustering(X_scaled, cluster_labels, gmm)

        # Plot the clusters (optional)
        plot_clusters(X_scaled, cluster_labels, n_components)
