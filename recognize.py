import face_recognition
import cv2
import numpy as np
import os

# Paths
KNOWN_FACES_DIR = "known_faces"
STUDENT_PHOTOS_DIR = "student_photos"

# Initialize known faces
known_face_encodings = []
known_face_names = []

print("[INFO] Loading known faces...")

# Load faces from both known_faces and student_photos directories
for file_name in os.listdir(KNOWN_FACES_DIR):
    if file_name.endswith(".jpg") or file_name.endswith(".png"):
        image_path = os.path.join(KNOWN_FACES_DIR, file_name)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            known_face_encodings.append(encodings[0])
            name = os.path.splitext(file_name)[0]
            known_face_names.append(name)
            print(f"Loaded: {name}")

# Load student photos
for file_name in os.listdir(STUDENT_PHOTOS_DIR):
    if file_name.endswith(".jpg") or file_name.endswith(".png"):
        image_path = os.path.join(STUDENT_PHOTOS_DIR, file_name)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            known_face_encodings.append(encodings[0])
            name = os.path.splitext(file_name)[0]
            known_face_names.append(name)
            print(f"Loaded: {name}")

print("[INFO] Starting webcam...")

# Start the webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face locations and encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding, tolerance=0.5
        )

        name = "Unknown"

        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding
        )

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Show the result
    cv2.imshow("Face Recognition App", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
