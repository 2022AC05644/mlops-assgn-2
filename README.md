# MLOps Assignment 2: End-to-End Machine Learning Workflow

## Overview
This repository contains the code and resources for MLOps Assignment 2, demonstrating an end-to-end machine learning workflow. The project covers data preprocessing, model training, hyperparameter tuning, model evaluation, and deployment using Google Cloud Platform (GCP).

## Project Structure and File Descriptions

- **README.md**: Provides an overview of the project, its structure, and setup instructions.
- **dataset-files/**:
  - `diabetes.csv`: The raw dataset used for training, containing features related to diabetes prediction.
  - `preprocessed_diabetes_data.csv`: The dataset after preprocessing, ready for model training.
- **gcp-flask-app/**:
  - `Dockerfile`: Instructions for containerizing the Flask app using Docker.
  - `app.py`: The main Flask application file that serves the trained model for making predictions.
  - `best_diabetes_model.pkl`: The saved, trained model used for making predictions in the Flask app.
  - `requirements.txt`: Lists the dependencies required to run the Flask app.
  - `shap_explainer.pkl`: A saved SHAP explainer object for generating model explainability outputs.
  - **templates/**:
    - `index.html`: The HTML template for the Flask app's user interface.
- **gcp-serverless-function/**:
  - `main.py`: The code for deploying the model as a Google Cloud Function for serverless deployment.
  - `requirements.txt`: Lists the dependencies required to run the Google Cloud Function.
- **jupyter-notebooks/**:
  - `MLOps_assign2_group42_Task2_AutoML.ipynb`: Notebook for model selection, training, and evaluation using PyCaret's AutoML features.
  - `MLops_assgn2_group42_Task1_EDA.ipynb`: Notebook for performing exploratory data analysis (EDA) on the dataset.
  - `MLops_assign2_group42_Task3_ExplainableAI.ipynb`: Notebook for applying SHAP-based explainability techniques to the trained model.
- **pikle-files/**:
  - `best_diabetes_model.pkl`: A duplicate of the trained model for backup purposes.
  - `shap_explainer.pkl`: A duplicate of the SHAP explainer object for backup purposes.

## Setup Instructions
```bash
# Clone the repository
git clone https://github.com/2022AC05644/mlops-assgn-2.git
cd mlops-assgn-2

# Install required dependencies
pip install -r gcp-flask-app/requirements.txt

# Run the Flask app locally
cd gcp-flask-app
python app.py

# Deploy the app on GCP
gcloud app deploy
```

## Notebooks

- **Task 1 (EDA)**: Exploratory Data Analysis is performed in `MLops_assgn2_group42_Task1_EDA.ipynb`.
- **Task 2 (AutoML)**: AutoML models are trained and evaluated in `MLOps_assign2_group42_Task2_AutoML.ipynb` using PyCaret.
- **Task 3 (Explainable AI)**: SHAP explainability is applied to the trained model in `MLops_assign2_group42_Task3_ExplainableAI.ipynb`.

## GCP Deployment

- **Flask App**: A Flask app for serving predictions via an API, located in the `gcp-flask-app/` folder.
- **Serverless Function**: Cloud Functions are implemented in the `gcp-serverless-function/` folder for serverless deployment.

## Usage

Run the Flask app locally:

```bash
cd gcp-flask-app
python app.py
```

## Deploy the app on GCP:

```bash
gcloud app deploy
```




