import cv2
import streamlit as st

# Charger le classificateur de cascade de visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces():
    # Initialisation de la webcam
    cap = cv2.VideoCapture(0)
    while True:
        # Lecture des images de la webcam
        ret, frame = cap.read()
        if not ret:
            break  # Sortir de la boucle si la lecture échoue
        # Conversion des images en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Détection des visages à l'aide du classificateur de cascade de visages
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        # Dessin de rectangles autour des visages détectés
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Affichage des images
        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)
        # Sortie de la boucle lorsque 'q' est pressé
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Libération de la webcam et fermeture de toutes les fenêtres
    cap.release()
    cv2.destroyAllWindows()

def app():
    st.title("Face Detection using Viola-Jones Algorithm")
    st.write("Appuyez sur le bouton ci-dessous pour commencer la détection des visages depuis votre webcam")
    # Ajout d'un bouton pour démarrer la détection des visages
    if st.button("Détecter les visages"):
        # Appel de la fonction detect_faces
        detect_faces()

if __name__ == "__main__":
    app()
