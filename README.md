# Face Recognition System

A Streamlit-based Face Recognition System built using OpenCV and the `face_recognition` library.  
This application allows you to register members, detect faces in real-time, and manage stored face data.

---

## Features

- ✅ Register Members (Upload Image or Use Webcam)
- ✅ Live Face Detection
- ✅ Manage Registered Members
- ✅ Delete Members
- ✅ Simple and Clean UI using Streamlit

---

## 🖥️ System Requirements

- Python **3.10 (Recommended)**
- Webcam (for live detection)
- Good lighting for better accuracy

---

##  Installation Guide

Follow these steps carefully:

### 1️⃣ Clone the Repository


If downloaded the file , simply navigate into the project folder:
OR

```
git clone <type-repository-url>
cd face_recognition_project
```

```bash
cd face_recognition
```

---

### 2️⃣ Install Python 3.10 (If Not Installed)

Check version:

```bash
python3.10 --version
```

If not installed (Ubuntu/Linux):

```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
```

---

### 3️⃣ Create Virtual Environment

```bash
python3.10 -m venv venv
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

After activation, you should see:

```
(venv)
```

in your terminal.

---

### 4️⃣ Install Required Dependencies

```bash
pip install -r requirements.txt
```

If requirements.txt is not available, install manually:

```bash
pip install streamlit opencv-python face-recognition pillow numpy
```

---

### 5️⃣ Run the Application

```bash
streamlit run app.py
```

The app will open automatically in your browser.

---

## 📂 Project Structure

```
face_recognition/
│
├── app.py
├── requirements.txt
├── known_faces/
├── student_photos/
└── README.md
```

---

## ⚠️ Important Notes

- Make sure your webcam is connected before starting.
- Ensure proper lighting for accurate face detection.
- Always activate the virtual environment before running the app.
- Recommended Python version: **3.10** for best compatibility with dlib and face_recognition.

---

## 🛠 Troubleshooting

If you face issues installing `face_recognition`, make sure:

- You are using Python 3.10
- `python3.10-dev` is installed
- Your virtual environment is activated

---

## 👨‍💻 Author

Developed by Joyal Joseph  
Streamlit + OpenCV + face_recognition Project

---

⭐ If you like this project, consider giving it a star!
