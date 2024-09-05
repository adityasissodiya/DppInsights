# **Digital Product Passport - Predictive Data Mining Project**

## **Project Overview**
This project focuses on using advanced data mining and machine learning techniques to enhance **Digital Product Passports (DPPs)** by predicting key outcomes like **repair frequency**, **recyclability**, and **environmental impact**. The goal is to provide manufacturers and consumers with actionable insights that help optimize product design for sustainability and support the circular economy.

The project will explore **classification**, **regression**, **clustering**, and **anomaly detection** using techniques such as **decision trees**, **random forests**, **support vector machines (SVM)**, **deep learning**, **Expectation Maximization (EM)**, **Markov models**, and **Bayesian networks**.

## **Project Objectives**
1. **Predict repair frequency, recyclability, and environmental impact** of products using advanced machine learning models.
2. **Cluster products** based on sustainability metrics.
3. **Detect anomalies** in product lifecycle data, such as unusually high repair frequency or low recyclability.
4. Provide **recommendations for improving product sustainability**.

### **Project Structure**

```
DPP_Predictive_Data_Mining/
│
├── data/                     # Directory to store raw and processed data
│   ├── raw/                   # Raw data files
│   ├── processed/             # Cleaned and transformed data
│
├── models/                    # Directory to save trained models
│   └── saved_models/          # Trained models (e.g., Random Forest, Neural Networks)
│
├── src/                       # Source code directory for different techniques
│   ├── data_processing.py      # Script for data cleaning and preprocessing
│   ├── feature_engineering.py  # Script for feature creation and selection
│
│   ├── classification/         # Directory for classification models
│   │   ├── decision_tree.py     # Decision Tree classification model
│   │   ├── random_forest.py     # Random Forest classification model
│   │   ├── svm.py               # Support Vector Machine (SVM) model
│   │   └── neural_network.py    # Neural Network model for classification
│
│   ├── regression/             # Directory for regression models
│   │   ├── random_forest_reg.py # Random Forest regression model
│   │   └── deep_learning_reg.py # Neural Network model for regression
│
│   ├── clustering/             # Directory for clustering techniques
│   │   ├── kmeans.py            # K-Means clustering algorithm
│   │   ├── em_clustering.py     # Expectation Maximization (EM) clustering
│
│   ├── anomaly_detection/      # Directory for anomaly detection models
│   │   ├── markov_model.py      # Markov Model for anomaly detection
│   │   ├── bayesian_network.py  # Bayesian Network for anomaly detection
│   │   └── isolation_forest.py  # Isolation Forest for anomaly detection
│
│   ├── train_models.py         # Script for orchestrating model training
│   ├── evaluate_models.py      # Script for evaluating models
│
├── results/                   # Directory to store results, reports, and visualizations
│   ├── metrics/                # Model evaluation metrics
│   └── visualizations/         # Data visualizations and clustering results
│
├── Dockerfile                 # Docker setup file
├── requirements.txt           # Python libraries and dependencies
├── README.md                  # Project overview and setup instructions
└── docker-compose.yml         # Docker Compose configuration
```

---

### **Explanation of Key Folders and Files:**

- **`data/`**:
  - **`raw/`**: Store raw data files such as material composition, repair history, recyclability scores, etc.
  - **`processed/`**: Cleaned and preprocessed data after running `data_processing.py`.

- **`models/`**:
  - **`saved_models/`**: This folder will store trained machine learning models (e.g., `.pkl` or `.h5` files for later use).

- **`src/`**:
  - **`data_processing.py`**: Handles data cleaning, missing value treatment, and preprocessing.
  - **`feature_engineering.py`**: Handles creating new features and selecting the most relevant ones for model training.
  
  - **`classification/`**:
    - **`decision_tree.py`**: Implements a decision tree model for classifying repairability or recyclability.
    - **`random_forest.py`**: Implements a random forest model for classification tasks (e.g., predicting sustainability metrics).
    - **`svm.py`**: Implements a Support Vector Machine (SVM) model for more complex classification tasks.
    - **`neural_network.py`**: Implements a deep learning neural network model for classification problems.
  
  - **`regression/`**:
    - **`random_forest_reg.py`**: Implements a random forest model for predicting continuous outcomes like carbon footprint.
    - **`deep_learning_reg.py`**: Implements a deep learning model for predicting continuous variables (e.g., repair costs).

  - **`clustering/`**:
    - **`kmeans.py`**: Implements K-Means clustering for grouping products based on sustainability metrics.
    - **`em_clustering.py`**: Implements Expectation Maximization (EM) for clustering based on probabilistic distributions.

  - **`anomaly_detection/`**:
    - **`markov_model.py`**: Implements Markov models for anomaly detection in product lifecycle events (e.g., unexpected repairs).
    - **`bayesian_network.py`**: Implements Bayesian Networks for anomaly detection, such as identifying rare or unusual product outcomes.
    - **`isolation_forest.py`**: Implements Isolation Forests for detecting outliers in product lifecycle data (e.g., products that have unusually low recyclability).

  - **`train_models.py`**: Orchestrates the process of training all models. This file imports the relevant techniques and runs them on the data.
  
  - **`evaluate_models.py`**: Script for evaluating the performance of all models. It calculates metrics such as accuracy, F1 score (for classification), R-squared (for regression), and visualizes results.

- **`results/`**:
  - **`metrics/`**: Stores evaluation metrics such as confusion matrices, classification reports, regression errors, and clustering scores.
  - **`visualizations/`**: Stores generated plots and visualizations, such as clustering visualizations, anomaly detection graphs, and predictive performance curves.

---

### **Setup Instructions** (updated for multiple techniques):

### 1. **Install Docker**:
Ensure Docker is installed and running on your machine. Download it from [here](https://www.docker.com/products/docker-desktop).

### 2. **Clone the Repository**:
```bash
git clone https://github.com/adityasissodiya/DppInsights.git
```

### 3. **Build the Docker Container**:
Build the Docker container which will encapsulate all the necessary dependencies for the project:
```bash
docker-compose up --build
```

### 4. **Run the Code**:
Once the Docker container is running, you can access the terminal of the container and execute the scripts.

- **Run a specific technique**, for example, training a decision tree:
```bash
docker exec -it <container_name> bash
python src/classification/decision_tree.py
```

- To train all models at once, you can run:
```bash
python src/train_models.py
```

### 5. **Install Python Dependencies Locally** (if needed):
If you prefer to run the project without Docker, you can install the necessary libraries locally:
```bash
pip install -r requirements.txt
```

---