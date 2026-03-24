# 🎙️ AI-Powered Speech Emotion Detection for Human-Machine Interaction

## 📌 Overview
This project implements a **Speech Emotion Recognition (SER)** system using deep learning techniques. It analyzes human speech and classifies it into emotional categories, enabling machines to understand and respond to human emotions effectively.

The system uses **LSTM (Long Short-Term Memory)** networks along with **MFCC-based feature extraction** to capture both spectral and temporal patterns in audio signals.

---

## 🚀 Features
- 🎧 Emotion detection from speech audio
- 🧠 Deep learning model using LSTM
- 📊 Feature extraction:
  - MFCC (Mel-Frequency Cepstral Coefficients)
  - Delta features
  - Delta-Delta features
- 🔄 Combined datasets for better generalization:
  - TESS
  - RAVDESS
- 🎯 Emotion classification:
  - Angry
  - Calm
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - 
---

## 🧪 Technologies Used
- Python
- NumPy
- Librosa
- TensorFlow / Keras
- Scikit-learn

---

## 📂 Dataset
- **TESS (Toronto Emotional Speech Set)**
- **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**

These datasets are combined to improve robustness across different speakers and emotional variations.

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/EmotionEcho.git
cd EmotionEcho
pip install -r requirements.txt
