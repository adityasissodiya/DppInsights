import sys
import os
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, K2Score, HillClimbSearch
from pgmpy.inference import VariableElimination
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

# Helper function to save visualizations (e.g., feature distributions)
def save_visualization(plot_name):
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, f'{plot_name}.png'))
    plt.close()

def create_bayesian_network(df):
    """
    Create and fit a Bayesian Network model using the data.
    
    :param df: DataFrame with product attributes
    :return: Fitted Bayesian Network
    """
    # Define the structure of the Bayesian Network (based on domain knowledge or learned structure)
    structure = [('RepairFrequency', 'RepairabilityScore'), 
                 ('RecyclabilityScore', 'RepairFrequency'),
                 ('CO2Emissions', 'RecyclabilityScore')]
    
    # Create and fit the Bayesian Network
    model = BayesianNetwork(structure)
    model.fit(df, estimator=MaximumLikelihoodEstimator)
    
    return model

def detect_anomalies(model, df):
    """
    Detect anomalies by finding instances where the likelihood of observed data is low.
    
    :param model: Fitted Bayesian Network model
    :param df: DataFrame with product attributes
    :return: List of anomaly indices
    """
    anomalies = []
    inference = VariableElimination(model)
    
    # Iterate over each product and check the likelihood of its feature values
    for index, row in df.iterrows():
        # Use one or more variables as evidence and query others
        evidence = {'RepairFrequency': row['RepairFrequency'], 'RecyclabilityScore': row['RecyclabilityScore']}
        
        # Query variables not in evidence
        query = inference.map_query(variables=['CO2Emissions', 'RepairabilityScore'], evidence=evidence)
        
        # Compute the actual likelihood (approximated with a random number here)
        likelihood = np.random.rand()  # Replace this with a proper likelihood calculation
        if likelihood < 0.05:  # Threshold for anomaly detection
            anomalies.append(index)
    
    return anomalies

if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv('data/processed/pseudo_product_data.csv')
    
    # Select a subset of the data (or all of it) for Bayesian Network analysis
    selected_features = ['RepairFrequency', 'RecyclabilityScore', 'CO2Emissions', 'RepairabilityScore']
    df_selected = df[selected_features]
    
    # Create and fit the Bayesian Network
    model = create_bayesian_network(df_selected)
    
    # Detect anomalies
    anomalies = detect_anomalies(model, df_selected)
    
    # Print results
    print(f"Number of anomalies detected: {len(anomalies)}")
    print(f"Anomalous records at indices: {anomalies}")

    # Save the number of anomalies as a metric
    metrics = {
        'Model': 'Bayesian Network',
        'Number of Anomalies': len(anomalies)
    }
    save_metrics(metrics, 'bayesian_network')

    # Optionally visualize feature distributions or results (e.g., histogram of RepairFrequency)
    plt.hist(df['RepairFrequency'], bins=20, color='blue', edgecolor='black')
    plt.title('Distribution of Repair Frequency')
    plt.xlabel('Repair Frequency')
    plt.ylabel('Frequency')
    save_visualization('repair_frequency_distribution')
