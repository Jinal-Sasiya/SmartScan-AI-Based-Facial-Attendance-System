HEAD
# 🎓 SmartScan – AI Based Facial Attendance System

An AI-powered real-time facial attendance management system built using **Python, OpenCV, Flask, SQLite, and LBPH Face Recognition**.

This system automates student attendance using live face recognition and provides a modern web dashboard for analytics, attendance monitoring, CSV export, and student registration.

---

# 🚀 Features

✅ Real-Time Face Detection
✅ LBPH Face Recognition
✅ Automatic Attendance Marking
✅ Student Registration System
✅ Automatic Face Dataset Collection
✅ Automatic Model Training
✅ Flask Web Dashboard
✅ Attendance Analytics
✅ CSV Export Support
✅ Secure Admin Login Authentication
✅ SQLite Database Integration
✅ Modern Responsive UI

---

# 🧠 Tech Stack

| Technology | Usage                        |
| ---------- | ---------------------------- |
| Python     | Core Programming             |
| OpenCV     | Face Detection & Recognition |
| Flask      | Web Framework                |
| SQLite     | Database                     |
| NumPy      | Numerical Processing         |
| Pandas     | CSV Export & Analytics       |
| Bootstrap  | Frontend UI                  |
| HTML/CSS   | Dashboard Design             |

---

# 📂 Project Structure

```bash
SmartScan-AI-Based-Facial-Attendance-System/
│
├── app.py
├── database.py
├── init_db.py
├── recognize_lbph.py
├── train_lbph.py
├── requirements.txt
├── Procfile
├── README.md
│
├── dataset/
├── models/
├── static/
├── templates/
│
├── attendance.db
├── attendance.csv
├── lbph_model.xml
├── label_map.npy
```

---

# ⚙️ System Workflow

```text
Admin Login
    ↓
Register Student
    ↓
Capture Face Images
    ↓
Train LBPH Model
    ↓
Real-Time Recognition
    ↓
Attendance Marking
    ↓
Analytics Dashboard
```

---

# 🔐 Authentication

The system includes a secure admin login module.

### Default Credentials

```text
Username: admin
Password: 1234
```

---

# 📸 Face Recognition Pipeline

### Face Detection

* OpenCV DNN Face Detector
* SSD-based face detection model

### Face Recognition

* LBPH (Local Binary Pattern Histogram)
* Real-time recognition using trained grayscale face images

---

# 📊 Dashboard Features

* Attendance Records Table
* Total Attendance Count
* Present Today Statistics
* System Status Indicator
* Student Search
* CSV Export
* Attendance Analytics

---

# 🛠️ Installation Guide

## 1️⃣ Clone Repository

```bash
git clone https://github.com/Jinal-Sasiya/SmartScan-AI-Based-Facial-Attendance-System.git
```

---

## 2️⃣ Navigate to Project

```bash
cd SmartScan-AI-Based-Facial-Attendance-System
```

---

## 3️⃣ Create Virtual Environment

```bash
python -m venv venv311
```

---

## 4️⃣ Activate Environment

### Windows

```bash
venv311\Scripts\activate
```

---

## 5️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 6️⃣ Run Flask Application

```bash
python app.py
```

---

# 🌐 Access Application

Open browser:

```text
http://127.0.0.1:5000
```

---

# 👨‍🎓 Student Registration

1. Login as Admin
2. Click **Register Student**
3. Enter Student Name
4. System captures face images automatically
5. Model trains automatically
6. Student becomes ready for attendance

---

# 📈 Attendance Analytics

The system provides:

* Daily attendance statistics
* Attendance records table
* CSV export support
* Real-time attendance tracking

---

# ☁️ Deployment

The project is deployment-ready using:

* Render
* Railway
* Gunicorn

---

# 🔮 Future Improvements

* Deep Learning Face Recognition (FaceNet / ArcFace)
* Anti-Spoofing Detection
* Multi-Camera Support
* Cloud Database Integration
* REST API Support
* Mobile Application
* Docker Deployment
* Email Notifications

---

# 🧪 Sample Use Cases

* College Attendance Management
* School Smart Attendance
* Office Employee Attendance
* AI Surveillance Systems
* Smart Classroom Automation

---

# 👨‍💻 Author

### Jinal Sasiya

Master’s Student — Data Science
AI / Machine Learning / Computer Vision Enthusiast

GitHub:
https://github.com/Jinal-Sasiya

---

# ⭐ Project Highlights

✔ End-to-End AI System
✔ Real-Time Face Recognition
✔ Full-Stack Flask Dashboard
✔ Database Integration
✔ Deployment Ready
✔ Resume & Portfolio Ready Project

---

# 📜 License

This project is developed for educational and research purposes.
=======
# 🚀 SmartScan – AI-Based Facial Attendance System

SmartScan is an end-to-end **computer vision system** designed to automate student attendance using facial recognition. The system replaces manual roll calls with a real-time pipeline that detects, tracks, and recognizes faces while maintaining stable identity over time.

---

## 📌 Overview

SmartScan is built to operate in real-world environments where lighting, movement, and partial occlusions can affect recognition performance.

To address instability in predictions, the system uses a **temporal voting mechanism**, which aggregates predictions across multiple frames to produce a consistent identity output.

In addition to recognition, the system implements **entry and exit logic**, allowing it to track when a person enters or leaves the frame.

---

## ⚙️ System Architecture

The system follows a modular pipeline:
Camera Feed (OpenCV)
↓
Face Detection (SSD / MTCNN)
↓
Face Tracking (Centroid Tracking)
↓
Face Recognition (Embeddings / LBPH)
↓
Temporal Voting (Stability Layer)
↓
Attendance Engine (Entry / Exit Logic)
↓
Storage (CSV / Database)


---


---

## 🛠️ Tech Stack

### 💻 Programming Language
- Python 3.10+

### 📚 Libraries & Frameworks
- OpenCV (cv2)
- NumPy
- Scikit-learn
- DeepFace

### 🧠 Models
- **Face Detection:** SSD (ResNet-10) / MTCNN  
- **Face Recognition:**
  - ArcFace embeddings  
  - LBPH (alternative lightweight model)

---

## 🗄️ Data Pipeline

### 📥 Data Collection
- Captures multiple face samples per individual  
- Performs face detection and cropping in real time  

### ⚙️ Embedding Generation
- Converts face images into numerical vectors  
- Stores embeddings in a serialized format (`.pkl`)  

### 🔍 Matching
- Uses cosine similarity to compare embeddings  
- Selects identity based on highest similarity score  

---

## 🔄 System Workflow

1. **Face Detection**  
   Faces are detected from each frame using a deep learning model.

2. **Tracking**  
   Each detected face is assigned a unique tracking ID to maintain continuity across frames.

3. **Recognition**  
   Face embeddings are generated and compared with stored embeddings.

4. **Temporal Voting**  
   Recent predictions are stored, and the most frequent identity is selected to reduce fluctuation.

5. **Attendance Logic**  
   - Entry is marked on first stable detection  
   - Exit is marked when a face is not detected for a defined number of frames  

---

## ✨ Key Features

- Real-time face detection and recognition  
- Multi-face tracking  
- Temporal voting for stable identity prediction  
- Entry and exit tracking  
- Scalable and modular architecture  

---

## 📂 Project Structure
```
SmartScan/
├── core/ # Core modules (tracking, recognition, utilities)
├── data/ # Embeddings and dataset
├── dataset/ # Collected face images
├── models/ # Pre-trained models
├── collect_faces.py # Data collection script
├── train_embeddings.py # Embedding generation
├── smart_recognition.py # Main recognition pipeline
├── recognize_lbph.py # Alternative recognition module
└── README.md
```


---

## ⚙️ Installation

### 1️⃣ Clone Repository
```bash
git clone (https://github.com/your-username/smartscan.git)
cd smartscan
```

### 2️⃣ Install Dependencies
```bash
pip install opencv-contrib-python numpy scikit-learn deepface
```

## ▶️ Usage
### 1️⃣ Collect Dataset
```bash
python collect_faces.py
```
### 2️⃣ Generate Embeddings
```bash
python train_embeddings.py
```
### 3️⃣ Run System
```bash
python smart_recognition.py
```

## 🚀 Future Improvements
- Web-based dashboard for attendance monitoring
- Database integration (SQLite / MySQL)
- Face anti-spoofing (liveness detection)
- Performance optimization for edge devices
- Analytics for attendance trends


## 👩‍💻 Author

### Jinal Sasiya
33ab21af58226e44e79ba0be47bd6725e0830600
