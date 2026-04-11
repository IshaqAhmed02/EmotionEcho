import os
import numpy as np
import librosa
import pickle
from flask import Flask, render_template, request, redirect, url_for, flash, session
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# -------------------------------
# Flask Configuration
# -------------------------------
app = Flask(__name__)
app.secret_key = "voice_emotion_secret"

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"wav"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------
# Load Model & Encoder
# -------------------------------
model = load_model("voice_emotion_model.h5")

with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# -------------------------------
# Helper Functions
# -------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_mfcc(file_path, n_mfcc=40):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)


def predict_emotion(audio_path):
    mfcc = extract_mfcc(audio_path)
    mfcc = mfcc.reshape(1, 40, 1)

    prediction = model.predict(mfcc, verbose=0)
    emotion_index = np.argmax(prediction)

    emotion = encoder.inverse_transform([emotion_index])[0]
    confidence = round(float(np.max(prediction)) * 100, 2)

    return emotion, confidence

# -------------------------------
# Routes
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        if "audio" not in request.files:
            flash("No file uploaded")
            return redirect(request.url)

        file = request.files["audio"]

        if file.filename == "":
            flash("No file selected")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            emotion, confidence = predict_emotion(file_path)

            print("File saved:", file_path)
            print("Prediction:", emotion, confidence)

            return render_template(
                "result.html",
                emotion=emotion,
                confidence=confidence,
                audio_file=filename
            )

        else:
            flash("Only WAV files are allowed")

    return render_template("index.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/about")
def about():
    return render_template("about.html")

# -------------------------------
# Run Server
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
