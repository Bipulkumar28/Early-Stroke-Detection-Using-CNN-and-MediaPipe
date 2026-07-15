# Early Stroke Detection Using CNN and MediaPipe

## Overview

Early Stroke Detection Using CNN and MediaPipe is a real-time AI-powered system that assists in identifying potential stroke symptoms using a standard webcam.

The system combines deep learning and computer vision techniques to analyze facial expressions, head posture, facial asymmetry, and upper-body movements. A Convolutional Neural Network (CNN) classifies facial images as Stroke or No Stroke, while MediaPipe extracts facial and pose landmarks to evaluate clinical indicators such as head tilt, mouth droop, eye asymmetry, and arm weakness.

Both outputs are fused to generate a final stroke risk score that is displayed through a real-time dashboard.

> **Note:** This project is intended for educational and research purposes only. It is **not a medical diagnostic device** and should not be used as a substitute for professional medical advice.

---

## Features

- Real-time webcam monitoring
- CNN-based facial stroke classification
- MediaPipe Face Landmark Detection
- MediaPipe Pose Detection
- Head tilt analysis
- Mouth droop detection
- Eye asymmetry detection
- Brow asymmetry detection
- Cheek asymmetry detection
- Arm weakness detection
- Hybrid AI + Rule-Based Risk Fusion
- Live Stroke Risk Dashboard
- Session logging (CSV)
- JSON report generation
- Automatic alert system
- Snapshot capture
- Voice alerts (Text-to-Speech)

---

## System Architecture

```
                   Webcam
                      │
                      ▼
                 OpenCV Camera
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
 MediaPipe Face & Pose         CNN Model
        │                           │
        ▼                           ▼
 Clinical Features        Stroke Probability
        │                           │
        └─────────────┬─────────────┘
                      ▼
              Decision Fusion
                      ▼
             Final Stroke Risk
                      ▼
              Dashboard & Alerts
```

---

## Technologies Used

- Python
- OpenCV
- TensorFlow / Keras
- MediaPipe
- NumPy
- pyttsx3

---

## Stroke Indicators Analysed

- Head Tilt
- Mouth Droop
- Eye Asymmetry
- Brow Asymmetry
- Cheek Asymmetry
- Facial Rigidity
- Arm Weakness
- CNN Facial Classification

---

## Project Structure

```
Early-Stroke-Detection-Using-CNN-and-MediaPipe
│
├── stroke_detector.py
├── train_model.ipynb
├── requirements.txt
├── README.md
├── screenshots/
└── .gitignore
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/Bipulkumar28/Early-Stroke-Detection-Using-CNN-and-MediaPipe.git
```

Move into the project

```bash
cd Early-Stroke-Detection-Using-CNN-and-MediaPipe
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python stroke_detector.py
```

---

## Dataset

The CNN model was trained on a dataset containing facial images categorized into:

- Stroke
- No Stroke

The dataset is **not included** in this repository due to size and licensing considerations.

---

## Results

The application provides:

- Rule-Based Stroke Risk
- CNN Stroke Probability
- Final Hybrid Stroke Risk
- Live Feature Visualization
- Alert Notifications
- Session Reports

---

## Future Improvements

- Mobile application integration
- Emergency contact notification
- Cloud-based prediction API
- Wearable sensor integration
- Speech analysis
- Eye gaze analysis
- Temporal CNN prediction smoothing


---

## License

This project is developed for academic and research purposes.Its not for Medical Purposes
