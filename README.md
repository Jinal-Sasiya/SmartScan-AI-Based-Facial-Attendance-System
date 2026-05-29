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
