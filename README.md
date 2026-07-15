# 🧠 AI-Based Stroke Risk Detection using Facial and Pose Analysis

A real-time computer vision application that analyzes facial landmarks and body posture to identify early signs associated with stroke. The system uses AI-based facial and pose analysis to estimate stroke risk and displays live analytics through an interactive dashboard.

---

## 📌 Overview

This project is a prototype for early stroke risk detection using a standard webcam. It continuously monitors facial symmetry and upper-body posture to identify potential stroke indicators such as facial drooping, head tilt, and arm weakness.

The processed data is visualized on a live analytics dashboard, making it easier to monitor the detection results in real time.

---

## ✨ Features

- 🎥 Real-time webcam monitoring
- 😀 Face landmark detection
- 🧍 Pose landmark detection
- 📐 Facial asymmetry analysis
- 📉 Head tilt detection
- 💪 Arm weakness estimation
- 📊 Stroke risk score calculation
- 📈 Live analytics dashboard
- 📝 CSV report generation
- 📄 JSON report generation

---

## 🛠 Tech Stack

### Frontend
- React
- TypeScript
- Tailwind CSS

### Backend
- Flask
- Flask-CORS

### AI / Computer Vision
- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas

---

## 📂 Project Structure

```text
Stroke-Detection/
│
├── backend.py
├── stroke_detector.py
├── dashboard/
│   ├── src/
│   ├── public/
│   └── package.json
│
├── stroke_logs/
├── reports/
└── README.md
```

---

## ⚙️ System Workflow

```text
Webcam
   │
   ▼
Face & Pose Detection
   │
   ▼
Feature Extraction
   │
   ▼
Stroke Risk Calculation
   │
   ▼
CSV / JSON Logging
   │
   ▼
Flask Backend API
   │
   ▼
React Dashboard
```

---

## 📊 Dashboard

The dashboard displays live analytics including:

- Current Stroke Risk
- Risk Level
- Live Risk Trend
- Feature Contribution
- Risk Meter
- System Status
- FPS
- Camera Status

---

## ▶️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/Stroke-Detection.git
cd Stroke-Detection
```

### Install Python Dependencies

```bash
pip install flask flask-cors opencv-python mediapipe pandas numpy
```

### Install Dashboard Dependencies

```bash
cd dashboard
npm install
```

---

## ▶️ Running the Project

### Start Backend

```bash
python backend.py
```

Backend runs on:

```text
http://127.0.0.1:5000/live
```

### Start Stroke Detection

```bash
python stroke_detector.py
```

### Start Dashboard

```bash
cd dashboard
npm start
```

---

## 📈 Current Status

✅ Face Detection

✅ Pose Detection

✅ Facial Landmark Analysis

✅ Stroke Risk Estimation

✅ Live Dashboard

✅ Backend API Integration

✅ CSV & JSON Report Generation

---

## 🔮 Future Improvements

- Improved stroke risk model
- Better facial feature analysis
- Historical analytics dashboard
- Cloud database integration
- Mobile application
- Wearable sensor integration
- Emergency notification system

---

## 📄 License

This project is intended for educational and research purposes.
