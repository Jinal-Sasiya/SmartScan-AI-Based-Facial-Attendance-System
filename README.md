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
git clone https://github.com/your-username/smartscan.git
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
