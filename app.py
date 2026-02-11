import streamlit as st
import face_recognition
import os
from PIL import Image
import numpy as np
import cv2
import time

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Face Recognition System", layout="wide")

KNOWN_FACES_DIR = "known_faces"
STUDENT_PHOTOS_DIR = "student_photos"

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(STUDENT_PHOTOS_DIR, exist_ok=True)

# -----------------------------
# SESSION STATE INITIALIZATION
# -----------------------------
if 'known_face_encodings' not in st.session_state:
    st.session_state.known_face_encodings = []
    st.session_state.known_face_names = []

# -----------------------------
# LOAD KNOWN FACES
# -----------------------------
def load_known_faces():
    st.session_state.known_face_encodings.clear()
    st.session_state.known_face_names.clear()

    for file_name in os.listdir(KNOWN_FACES_DIR):
        if file_name.endswith((".jpg", ".png")):
            image_path = os.path.join(KNOWN_FACES_DIR, file_name)
            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)

                if len(encodings) > 0:
                    st.session_state.known_face_encodings.append(encodings[0])
                    name = os.path.splitext(file_name)[0]
                    st.session_state.known_face_names.append(name)
            except Exception as e:
                st.warning(f"Could not load {file_name}: {str(e)}")

# Load faces only once
if len(st.session_state.known_face_encodings) == 0:
    load_known_faces()

# -----------------------------
# REGISTER FUNCTION
# -----------------------------
def register_student(name, photo):
    photo_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")

    try:
        with open(photo_path, "wb") as f:
            f.write(photo.getbuffer())

        image = face_recognition.load_image_file(photo_path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            st.session_state.known_face_encodings.append(encodings[0])
            st.session_state.known_face_names.append(name)
            st.success(f"✅ {name} registered successfully!")
            return True
        else:
            os.remove(photo_path)
            st.error("⚠️ No face detected. Please upload a clear image.")
            return False
    except Exception as e:
        st.error(f"Error registering student: {str(e)}")
        if os.path.exists(photo_path):
            os.remove(photo_path)
        return False

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📁 Face App")

menu = st.sidebar.radio(
    "Navigation",
    ["Register Member", "Live Detection", "Manage Members"]
)

# -----------------------------
# PAGE: REGISTER MEMBER
# -----------------------------
if menu == "Register Member":

    st.markdown("## ➕ Register New Member")
    st.write("Add a new member to the face recognition system.")

    tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Capture from Webcam"])

    # ---------- Upload ----------
    with tab1:
        st.subheader("Upload Member Photo")

        name = st.text_input("Name", key="upload_name")
        photo = st.file_uploader("Choose an image (JPG, PNG)", type=["jpg", "png"])

        if st.button("Register Member", key="register_upload"):
            if name and photo:
                register_student(name, photo)
            else:
                st.error("Please fill all fields.")

    # ---------- Webcam ----------
    with tab2:
        st.subheader("Capture from Webcam")

        camera_photo = st.camera_input("Take a picture")

        if camera_photo is not None:
            name_web = st.text_input("Name (Webcam)", key="webcam_name")

            if st.button("Register from Webcam", key="register_webcam"):
                if name_web:
                    register_student(name_web, camera_photo)
                else:
                    st.error("Please enter name.")

    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    ✅ Clear lighting  
    ✅ Face straight to camera  
    ❌ No multiple faces  
    ❌ No blurry images  
    """)

# -----------------------------
# PAGE: LIVE DETECTION
# -----------------------------
elif menu == "Live Detection":

    st.title("📷 Live Face Recognition")
    st.write("Use your webcam to detect and recognize faces.")

    # Initialize session state for camera control
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("▶️ Start Camera"):
            st.session_state.camera_running = True
    
    with col2:
        if st.button("⏹️ Stop Camera"):
            st.session_state.camera_running = False

    FRAME_WINDOW = st.empty()
    
    if st.session_state.camera_running:
        video_capture = cv2.VideoCapture(0)
        
        # Set camera properties for better performance
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        video_capture.set(cv2.CAP_PROP_FPS, 30)

        frame_count = 0
        process_every_n_frames = 1  # Process every 2nd frame for better performance

        try:
            while st.session_state.camera_running:
                ret, frame = video_capture.read()
                
                if not ret:
                    st.error("Unable to access camera.")
                    break

                frame_count += 1

                # Process face recognition only on selected frames
                if frame_count % process_every_n_frames == 0:
                    # Resize frame for faster processing
                    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                    # Scale back up face locations
                    face_locations = [(top*2, right*2, bottom*2, left*2) 
                                     for (top, right, bottom, left) in face_locations]

                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

                        matches = []
                        name = "Unknown"

                        if len(st.session_state.known_face_encodings) > 0:
                            matches = face_recognition.compare_faces(
                                st.session_state.known_face_encodings, face_encoding, tolerance=0.5
                            )

                            face_distances = face_recognition.face_distance(
                                st.session_state.known_face_encodings, face_encoding
                            )
                            best_match_index = np.argmin(face_distances)

                            if matches[best_match_index]:
                                name = st.session_state.known_face_names[best_match_index]

                        # Draw box
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                        # Draw label background
                        cv2.rectangle(frame, (left, bottom - 35),
                                    (right, bottom), color, cv2.FILLED)

                        # Draw name
                        cv2.putText(frame, name,
                                  (left + 6, bottom - 6),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.8,
                                  (255, 255, 255),
                                  2)

                # Display frame
                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

                # Small delay to prevent overwhelming the system
                time.sleep(0.03)

        except Exception as e:
            st.error(f"Error during face detection: {str(e)}")
        
        finally:
            # Always release the camera properly
            video_capture.release()
            cv2.destroyAllWindows()
            st.session_state.camera_running = False

# -----------------------------
# PAGE: MANAGE MEMBERS
# -----------------------------
elif menu == "Manage Members":

    st.title("👥 Registered Members")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh List"):
            load_known_faces()
            st.rerun()

    members = os.listdir(KNOWN_FACES_DIR)
    members = [m for m in members if m.endswith(('.jpg', '.png'))]

    if members:
        st.write(f"**Total Members: {len(members)}**")
        st.markdown("---")
        
        for member in members:
            col1, col2, col3 = st.columns([1, 4, 1])
            
            with col1:
                try:
                    st.image(os.path.join(KNOWN_FACES_DIR, member), width=80)
                except:
                    st.write("🖼️")
            
            with col2:
                st.write("**" + os.path.splitext(member)[0] + "**")
            
            with col3:
                if st.button("🗑️", key=f"delete_{member}"):
                    try:
                        os.remove(os.path.join(KNOWN_FACES_DIR, member))
                        load_known_faces()
                        st.success(f"Deleted {os.path.splitext(member)[0]}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting: {str(e)}")
            
            st.markdown("---")
    else:
        st.info("No members registered yet.")