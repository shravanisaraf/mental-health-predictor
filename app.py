import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# --- 1. Load the Saved Model and Columns ---
# Use st.cache_data to prevent reloading the model on every interaction
@st.cache_data
def load_model():
    model = joblib.load('final_xgb_model.joblib')
    columns = joblib.load('model_columns.joblib')
    return model, columns

model, model_columns = load_model()

# --- 2. Create the App Interface ---
st.title("🧠 MindScope: Mental Health in Tech Predictor")
st.write(
    "This app uses a machine learning model to predict whether a tech worker "
    "might seek mental health treatment. Fill in the details below to get a prediction."
)

# --- 3. Create Input Fields for User ---
# We'll create input fields for some of the most important features.

# Using dictionaries for selectbox options
family_history_options = {'No': 0, 'Yes': 1}
work_interfere_options = {'Never': 0, 'Often': 1, 'Rarely': 2, 'Sometimes': 3}
benefits_options = {"Don't know": 0, 'No': 1, 'Yes': 2}
care_options_options = {'No': 0, 'Not sure': 1, 'Yes': 2}
age_group_options = {'18-25': 0, '26-35': 1, '36-45': 2, '46-75': 3}

# Create columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    age_group = st.selectbox("Age Group", options=list(age_group_options.keys()))
    family_history = st.selectbox("Family history of mental illness?", options=list(family_history_options.keys()))
    work_interfere = st.selectbox("Does your mental health condition interfere with your work?", options=list(work_interfere_options.keys()))

with col2:
    benefits = st.selectbox("Does your employer provide mental health benefits?", options=list(benefits_options.keys()))
    care_options = st.selectbox("Do you know the options for mental health care your employer provides?", options=list(care_options_options.keys()))
    support_score = st.slider("Workplace Support Score (from Benefits, Care Options, etc.)", min_value=0, max_value=4, value=2)

# --- 4. Create the Prediction Button ---
if st.button("Get Prediction", type="primary"):
    # Create a DataFrame from user inputs
    # Start with a dictionary of all zeros
    input_data = {col: [0] for col in model_columns}

    # Update the dictionary with user inputs, using the mapped numerical values
    input_data['age_group'] = [age_group_options[age_group]]
    input_data['family_history'] = [family_history_options[family_history]]
    input_data['work_interfere'] = [work_interfere_options[work_interfere]]
    input_data['benefits'] = [benefits_options[benefits]]
    input_data['care_options'] = [care_options_options[care_options]]
    input_data['support_score'] = [support_score]

    # Convert to DataFrame
    input_df = pd.DataFrame(input_data)
    
    # Ensure all columns are in the correct order
    input_df = input_df[model_columns]

    # --- Make Prediction ---
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.success("The model predicts this individual is **likely** to seek treatment.")
        st.write(f"Confidence: {prediction_proba[0][1]*100:.2f}%")
    else:
        st.info("The model predicts this individual is **not likely** to seek treatment.")
        st.write(f"Confidence: {prediction_proba[0][0]*100:.2f}%")

    st.write(
        "**Disclaimer**: This is an AI-powered prediction and not a medical diagnosis. "
        "Please consult a healthcare professional for any health concerns."
    )
