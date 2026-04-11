import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model

# -------------------------------
# Load Model & Encoder
# -------------------------------
model = load_model("voice_emotion_model.h5")

with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# -------------------------------
# Feature Extraction
# -------------------------------
def extract_mfcc(file_path, n_mfcc=40):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

# -------------------------------
# Predict Emotion
# -------------------------------
def predict_emotion(audio_path):
    mfcc = extract_mfcc(audio_path)
    mfcc = mfcc.reshape(1, 40, 1)

    prediction = model.predict(mfcc)
    emotion_index = np.argmax(prediction)

    emotion = encoder.inverse_transform([emotion_index])
    return emotion[0]

# -------------------------------
# Test with Audio File
# -------------------------------
audio_file = r"C:\Users\hp\Music\emotion_face_voice\dataset\Sad\03a05Tc.wav" # give your .wav file path
print("🎧 Predicted Emotion:", predict_emotion(audio_file))
