import streamlit as st
import face_recognition
import os
from PIL import Image
import numpy as np

# Path to known faces
KNOWN_FACES_DIR = "known_faces"
STUDENT_PHOTOS_DIR = "student_photos"

# Function to register a student
def register_student(name, email, photo):
    # Save photo
    photo_path = os.path.join(STUDENT_PHOTOS_DIR, f"{name}.jpg")
    
    # Save the image as a file
    with open(photo_path, "wb") as f:
        f.write(photo.getbuffer())

    # Encode the student's face
    image = face_recognition.load_image_file(photo_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) > 0:
        # Save encoding and name for recognition later
        known_face_encodings.append(encodings[0])
        known_face_names.append(name)
        print(f"Student {name} registered successfully!")
    else:
        st.error(f"⚠️ No face detected in the image for {name}")

# Initialize known faces
known_face_encodings = []
known_face_names = []

# Load known faces from the "known_faces" directory
for file_name in os.listdir(KNOWN_FACES_DIR):
    if file_name.endswith(".jpg") or file_name.endswith(".png"):
        image_path = os.path.join(KNOWN_FACES_DIR, file_name)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            known_face_encodings.append(encodings[0])
            name = os.path.splitext(file_name)[0]
            known_face_names.append(name)
            print(f"Loaded known face: {name}")

# Streamlit UI for student registration
st.title("Student Registration")

# Form input fields
name = st.text_input("Name")
email = st.text_input("Email")
photo = st.file_uploader("Upload Photo", type=["jpg", "png"])

# Register button
if st.button("Register Student"):
    if name and email and photo:
        register_student(name, email, photo)
        st.success(f"Student {name} registered successfully!")
    else:
        st.error("Please fill all fields and upload a photo.")

