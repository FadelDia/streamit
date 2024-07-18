import cv2
import streamlit as st
import numpy as np

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    st.write("Press 'q' to stop detecting faces")
    while True:
        # Read frames from the webcam
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if reading fails
        # Convert frames to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces using the face cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Convert the frame to RGB (required by Streamlit)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Display the frames in Streamlit
        st.image(frame_rgb, channels="RGB", use_column_width=True)
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the webcam
    cap.release()

if __name__ == "__main__":
    st.title("Face Detection using Viola-Jones Algorithm")
    detect_faces()
