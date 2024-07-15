!pip install streamlit
import streamlit as st
import pandas as pd
import joblib  # To load your trained model

# Load your trained model
model = joblib.load('trained_model.pkl') 

# Create input fields for features
st.title('Churn Prediction App')
# Replace with actual feature names from your dataset
feature1 = st.number_input('REGULARITY') 
feature2 = st.number_input('DATA_VOLUME')
# ... add input fields for all your features

# Create a validation button
if st.button('Predict'):
    # Create a DataFrame from the input values
    input_data = pd.DataFrame({
        'REGULARITY': [feature1],
        'DATA_VOLUME': [feature2],
        # ... add all your features
    })

    # Make prediction using the loaded model
    prediction = model.predict(input_data)

    # Display the prediction
    st.write('Prediction:', prediction[0])
