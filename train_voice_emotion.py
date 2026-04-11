import os
import numpy as np
import librosa
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten

# -------------------------------
# Feature Extraction
# -------------------------------
def extract_mfcc(file_path, n_mfcc=40):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

# -------------------------------
# Load Dataset
# -------------------------------
dataset_path = "dataset"   # dataset/Happy, Angry, Neutral, Sad
emotions = ["Happy", "Angry", "Neutral", "Sad"]

X, y = [], []

for emotion in emotions:
    folder = os.path.join(dataset_path, emotion)
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)
            X.append(extract_mfcc(path))
            y.append(emotion)

X = np.array(X)
y = np.array(y)

# -------------------------------
# Encode Labels
# -------------------------------
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Save encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = X_train.reshape(X_train.shape[0], 40, 1)
X_test = X_test.reshape(X_test.shape[0], 40, 1)

# -------------------------------
# CNN Model
# -------------------------------
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(40,1)),
    MaxPooling1D(2),
    Dropout(0.3),

    Conv1D(128, 3, activation='relu'),
    MaxPooling1D(2),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# Train Model
# -------------------------------
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32
)

# -------------------------------
# Save Model
# -------------------------------
model.save("voice_emotion_model.h5")

print("✅ Training completed & model saved")
