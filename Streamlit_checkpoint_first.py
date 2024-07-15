import streamlit as st
import pandas as pd
import joblib
from io import BytesIO

# Titre de l'application
st.title('Churn Prediction App')

# Téléchargement du fichier modèle
uploaded_file = st.file_uploader("trained_model.pkl", type="pkl")

if uploaded_file is not None:
    # Charger le modèle depuis le fichier téléchargé
    try:
        model = joblib.load(BytesIO(uploaded_file.read()))
        st.success("Modèle chargé avec succès!")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")

    # Créez les champs de saisie pour les fonctionnalités
    feature1 = st.number_input('REGULARITY')
    feature2 = st.number_input('DATA_VOLUME')
    # ... ajoutez des champs de saisie pour toutes vos fonctionnalités

    # Créez un bouton de validation
    if st.button('Predict'):
        input_data = pd.DataFrame({
            'REGULARITY': [feature1],
            'DATA_VOLUME': [feature2],
            # ... ajoutez toutes vos fonctionnalités
        })
        prediction = model.predict(input_data)
        st.write('Prediction:', prediction[0])
else:
    st.info("Veuillez télécharger le fichier trained_model.pkl pour continuer.")
