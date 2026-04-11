import os
import numpy as np
import librosa
import pickle
from flask import Flask, render_template, request, redirect, url_for, flash, session
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# -------------------------------
# Flask Configuration
# -------------------------------
app = Flask(__name__)
app.secret_key = "voice_emotion_secret_key_2024"

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"wav"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------
# Simple In-Memory User Store
# (Replace with a real DB like SQLite/PostgreSQL in production)
# -------------------------------
users_db = {}  # { username: { password_hash, email, firstname, lastname } }

# -------------------------------
# Load Model & Encoder
# -------------------------------
model = load_model("voice_emotion_model.h5")

with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# -------------------------------
# Auth Decorator
# -------------------------------
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            flash("Please log in to access this page.")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

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
# Routes — Auth
# -------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if "username" in session:
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            flash("Please fill in all fields.")
            return redirect(request.url)

        user = users_db.get(username)
        if user and check_password_hash(user["password_hash"], password):
            session["username"] = username
            session["firstname"] = user.get("firstname", username)
            flash(f"Welcome back, {user.get('firstname', username)}!")
            return redirect(url_for("index"))
        else:
            flash("Invalid username or password.")
            return redirect(request.url)

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if "username" in session:
        return redirect(url_for("index"))

    if request.method == "POST":
        firstname = request.form.get("firstname", "").strip()
        lastname = request.form.get("lastname", "").strip()
        email = request.form.get("email", "").strip()
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        # Validation
        if not all([firstname, lastname, email, username, password, confirm]):
            flash("All fields are required.", "error")
            return redirect(request.url)

        if password != confirm:
            flash("Passwords do not match.", "error")
            return redirect(request.url)

        if len(password) < 8:
            flash("Password must be at least 8 characters.", "error")
            return redirect(request.url)

        if username in users_db:
            flash("Username already taken. Please choose another.", "error")
            return redirect(request.url)

        # Check if email already registered
        for u in users_db.values():
            if u.get("email") == email:
                flash("An account with this email already exists.", "error")
                return redirect(request.url)

        # Save user
        users_db[username] = {
            "password_hash": generate_password_hash(password),
            "email": email,
            "firstname": firstname,
            "lastname": lastname,
        }

        flash("Account created successfully! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.")
    return redirect(url_for("login"))

# -------------------------------
# Routes — App
# -------------------------------

@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/", methods=["GET", "POST"])
@login_required
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

            try:
                emotion, confidence = predict_emotion(file_path)
            except Exception as e:
                flash(f"Error processing audio: {str(e)}")
                return redirect(request.url)

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

# -------------------------------
# Run Server
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
