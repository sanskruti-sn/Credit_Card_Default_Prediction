# Credit Card Default Prediction

## Project Overview
This project predicts whether a credit card customer will default on their payment next month using machine learning. The model is trained on historical credit card usage data and includes data preprocessing, feature engineering, model training, and evaluation.

## Features
- Data preprocessing and cleaning
- Feature scaling using StandardScaler
- Model training using XGBoost classifier
- Model evaluation with classification metrics
- Flask-based web application for prediction
- Unit tests for code reliability

## Project Structure

├── .gitignore
├── main.py
├── README.md
├── requirements.txt
├── .venv/ # Virtual environment folder
├── app/
│ ├── app.py # Flask app backend
│ └── templates/
│ └── index.html # Frontend HTML page
├── data/
│ └── UCI_Credit_Card.csv # Original dataset
├── notebooks/
│ ├── artifacts/
│ │ ├── model.pkl # Saved ML model
│ │ └── scaler.pkl # Saved scaler for features
│ ├── logs/
│ │ └── app.log # Log file for app
│ └── eda_and_modeling.ipynb # Jupyter notebook for EDA & modeling
├── src/
│ ├── init.py
│ ├── data_ingestion.py # Data loading and ingestion scripts
│ ├── model.py # Model training and evaluation scripts
│ ├── predict.py # Prediction helper functions
│ ├── preprocessing.py # Data preprocessing scripts
│ └── pycache/ # Python cache files
├── tests/
│ ├── test_model.py # Unit tests for model and preprocessing
│ └── pycache/ # Python cache files



## Installation

## Setup and Installation

1. **Clone the repository**  
   ```bash
   git clone <your_repo_url>
   cd credit_card_default_prediction

## Create and activate virtual environment

**On Windows (PowerShell):**

**powershell**

- python -m venv venv
- .\venv\Scripts\Activate.ps1

**On Linux/macOS:**

- python3 -m venv venv
- source venv/bin/activate


## Install dependencies

- pip install -r requirements.txt

## Usage -> Running the Flask Web Application

- python app/app.py

**Open your browser and navigate to http://127.0.0.1:5000/ to use the credit card default prediction interface.**


## Running Unit Tests

- python -m unittest tests/test_model.py


## Description of Key Files

- app/app.py: Flask app that provides a web interface for input and prediction.
- app/templates/index.html: Frontend HTML page.
- main.py: (If applicable) Entry point script.
- src/data_ingestion.py: Functions to load and ingest dataset.
- src/preprocessing.py: Data cleaning and preprocessing logic.
- src/model.py: Model training, evaluation, and saving.
- src/predict.py: Functions for model inference.
- notebooks/eda_and_modeling.ipynb: Notebook containing exploratory data analysis and model development.
- notebooks/artifacts/model.pkl: Saved XGBoost model.
- notebooks/artifacts/scaler.pkl: Saved StandardScaler object.
- tests/test_model.py: Unit tests to ensure code correctness.

## Model Details

- Model Type: XGBoost Classifier
- Data Scaling: StandardScaler applied on features before model training.
- Metrics: Accuracy, Precision, Recall, F1-score.
- Trained on: UCI Credit Card dataset.

## How to Extend

- Add new models in src/model.py.
- Extend preprocessing in src/preprocessing.py.
- Enhance frontend by modifying app/templates/index.html.
- Add more tests in tests/test_model.py.

## Acknowledgements

- Dataset from UCI Machine Learning Repository

## Contact

- For any issues or contributions, please raise an issue or submit a pull request.

## ⭐ Thank you for checking out this Credit Card Default Prediction project!

