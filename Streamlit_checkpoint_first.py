import streamlit as st
import pandas as pd
import joblib

# Titre de l'application
st.title('Churn Prediction App')

# Chemin d'accès au fichier modèle
model_path = '/content/trained_model.pkl'

# Charger le modèle depuis le chemin spécifié
try:
    model = joblib.load(model_path)
    st.success("Modèle chargé avec succès!")
except FileNotFoundError:
    st.error("Le fichier 'trained_model.pkl' est introuvable.")
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    model = None

# Créez les champs de saisie pour les fonctionnalités
feature1 = st.number_input('REGULARITY')
feature2 = st.number_input('DATA_VOLUME')
# ... ajoutez des champs de saisie pour toutes vos fonctionnalités

# Créez un bouton de validation
if st.button('Predict'):
    if model is not None:
        input_data = pd.DataFrame({
            'REGULARITY': [feature1],
            'DATA_VOLUME': [feature2],
            # ... ajoutez toutes vos fonctionnalités
        })
        try:
            prediction = model.predict(input_data)
            st.write('Prediction:', prediction[0])
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
    else:
        st.error("Le modèle n'a pas pu être chargé.")
