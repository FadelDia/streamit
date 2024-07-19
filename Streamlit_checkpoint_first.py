import streamlit as st
import pandas as pd
import joblib  # Pour charger votre modèle entraîné

# Charger votre modèle entraîné
model = joblib.load('retrained_model1.plk')

# Créer des champs de saisie pour les caractéristiques
st.title('Churn Prediction App')

# Remplacer par les noms réels des caractéristiques de votre dataset
feature1 = st.number_input('REGULARITY')
feature2 = st.number_input('DATA_VOLUME')
# ... ajoutez des champs de saisie pour toutes vos caractéristiques

# Créer un bouton de validation
if st.button('Predict'):
    # Créer un DataFrame à partir des valeurs saisies
    input_data = pd.DataFrame({
        'REGULARITY': [feature1],
        'DATA_VOLUME': [feature2],
        # ... ajoutez toutes vos caractéristiques
    })

    # Faire une prédiction en utilisant le modèle chargé
    prediction = model.predict(input_data)

    # Afficher la prédiction
    st.write('Prediction:', prediction[0])

