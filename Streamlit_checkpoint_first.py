import streamlit as st
import pandas as pd
import joblib  # To load your trained model
import requests

# URL of your model on GitHub
model_url = "https://raw.githubusercontent.com/<FadelDia>/<streamlit>/main/retrained_model.pkl"  

# Download the model
response = requests.get(model_url)
open("retrained_model.pkl", "wb").write(response.content)

# Load your trained model
model = joblib.load('retrained_model.pkl')

# Get feature names from the model
feature_names = model.feature_names_in_

# Create input fields for features
st.title('Churn Prediction App')

# Create input fields dynamically
input_data = {}
for feature in feature_names:
    # Assuming all features are numerical
    input_data[feature] = st.number_input(feature) 

# Create a validation button
if st.button('Predict'):
    # Create a DataFrame from the input values
    input_df = pd.DataFrame([input_data])

    # Make prediction using the loaded model
    prediction = model.predict(input_df)

    # Display the prediction
    st.write('Prediction:', prediction[0])
