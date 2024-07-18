import streamlit as st

def detect_faces():
    # Initialisation de la webcam
    cap = cv2.VideoCapture(0)
    while True:
        # Lecture des images de la webcam
        ret, frame = cap.read()
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
    st.write("Press the button below to start detecting faces from your webcam")
    # Ajout d'un bouton pour démarrer la détection des visages
    if st.button("Detect Faces"):
        # Appel de la fonction detect_faces
        detect_faces()

if __name__ == "__main__":
    app()
