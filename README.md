# Mental Health in Tech Predictor

This project develops a machine learning model to predict whether an individual working in the technology sector is likely to seek treatment for a mental health condition. The model is based on the OSMI (Open Sourcing Mental Health) survey dataset.

---

## Objective

The primary objective is to build and evaluate a predictive model that identifies key factors influencing the decision to seek mental health treatment. The project includes a full data science workflow, from data cleaning and feature engineering to model training, hyperparameter tuning, and interpretation.

---

## How to Run the Application Locally

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-folder-name>
    ```

2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

---

## Repository Contents

* **`Mental_Health_Capstone.ipynb`**: The primary Jupyter Notebook detailing all steps of the project, including data cleaning, model training, and evaluation.
* **`app.py`**: The Python script for the interactive Streamlit web application.
* **`requirements.txt`**: A file listing all necessary Python libraries for project replication.
* **`final_xgb_model.joblib`**: The serialized, pre-trained XGBoost model object.
* **`model_columns.joblib`**: A serialized list of the data columns required by the model for prediction.
