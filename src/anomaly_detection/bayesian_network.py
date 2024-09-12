import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, K2Score, HillClimbSearch
from pgmpy.inference import VariableElimination
import numpy as np

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
        
        # You should compute the actual likelihood using the inference model (use approximate likelihood here)
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
    
    print(f"Number of anomalies detected: {len(anomalies)}")
    print(f"Anomalous records at indices: {anomalies}")
