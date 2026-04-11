import os
import pickle
from tensorflow.keras.models import load_model

def test_load():
    try:
        if not os.path.exists("voice_emotion_model.h5"):
            print("❌ voice_emotion_model.h5 is missing")
        else:
            model = load_model("voice_emotion_model.h5")
            print("✅ model loaded")
        
        if not os.path.exists("label_encoder.pkl"):
            print("❌ label_encoder.pkl is missing")
        else:
            with open("label_encoder.pkl", "rb") as f:
                encoder = pickle.load(f)
            print("✅ encoder loaded")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_load()
