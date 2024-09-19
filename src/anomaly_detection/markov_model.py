import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict

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

def create_state_labels(df):
    """
    Create labels representing the states for Markov Model based on product attributes.
    For example, classify products into 'high' or 'low' repair frequency states.
    
    :param df: DataFrame with product attributes
    :return: Series with state labels
    """
    # Example: Classify RepairFrequency as 'high' or 'low' based on a threshold
    threshold = df['RepairFrequency'].median()
    return np.where(df['RepairFrequency'] > threshold, 'high_repair', 'low_repair')

def calculate_transition_matrix(state_sequence):
    """
    Calculate the Markov transition matrix based on observed state transitions.
    
    :param state_sequence: List or array of state transitions
    :return: Transition matrix as a dictionary
    """
    transition_matrix = defaultdict(lambda: defaultdict(int))
    
    # Count transitions between states
    for (current_state, next_state) in zip(state_sequence[:-1], state_sequence[1:]):
        transition_matrix[current_state][next_state] += 1
    
    # Convert counts to probabilities
    for current_state, transitions in transition_matrix.items():
        total = sum(transitions.values())
        for next_state in transitions:
            transition_matrix[current_state][next_state] /= total
    
    return transition_matrix

def detect_anomalies(state_sequence, transition_matrix, threshold=0.05):
    """
    Detect anomalies in state transitions based on low transition probabilities.
    
    :param state_sequence: List or array of state transitions
    :param transition_matrix: Markov transition matrix
    :param threshold: Probability threshold for detecting anomalies
    :return: List of anomaly indices
    """
    anomalies = []
    for i in range(1, len(state_sequence)):
        current_state = state_sequence[i - 1]
        next_state = state_sequence[i]
        
        if transition_matrix[current_state][next_state] < threshold:
            anomalies.append(i)
    
    return anomalies

# Main block to execute the functions
if __name__ == "__main__":
    # Load and preprocess the data
    df = pd.read_csv('data/processed/pseudo_product_data.csv')
    
    # Create state labels for Markov Model (e.g., based on RepairFrequency)
    state_labels = create_state_labels(df)
    
    # Calculate transition matrix
    transition_matrix = calculate_transition_matrix(state_labels)
    
    # Detect anomalies based on low transition probabilities
    anomalies = detect_anomalies(state_labels, transition_matrix)

    print(f"Number of anomalies detected: {len(anomalies)}")
    print(f"Anomalous transitions at indices: {anomalies}")

    # Save metrics
    metrics = {
        'Model': 'Markov Model',
        'Number of Anomalies': len(anomalies)
    }
    save_metrics(metrics, 'markov_model')

    # Visualize and save state transitions distribution
    unique, counts = np.unique(state_labels, return_counts=True)
    plt.bar(unique, counts)
    plt.title('State Distribution (High vs Low Repair Frequency)')
    plt.xlabel('State')
    plt.ylabel('Count')
    save_visualization('markov_model_state_distribution')
