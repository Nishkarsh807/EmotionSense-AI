"""
EmotionSense AI - Real-Time Live Webcam & Video Emotion Detection
Uses OpenCV for face detection and Mini-Xception for real-time inference.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "outputs" / "models" / "best_emotion_model.keras"

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Distinct aesthetic colors for each emotion (BGR format)
EMOTION_COLORS = {
    "Angry": (0, 0, 255),      # Red
    "Disgust": (0, 140, 255),  # Orange
    "Fear": (180, 0, 180),     # Purple
    "Happy": (0, 230, 0),      # Bright Green
    "Neutral": (200, 200, 200),# Gray/White
    "Sad": (255, 100, 0),      # Blue
    "Surprise": (0, 255, 255)  # Yellow
}

def load_emotion_model():
    """Load trained emotion recognition model."""
    if not MODEL_PATH.exists():
        fallback_path = BASE_DIR / "outputs" / "models" / "final_emotion_model.keras"
        if fallback_path.exists():
            print(f"[Model] Loading model from {fallback_path}")
            return tf.keras.models.load_model(str(fallback_path))
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH} or {fallback_path}. "
            "Please run 'python src/train.py' first to train the high-accuracy model."
        )
    print(f"[Model] Loading model from {MODEL_PATH}")
    return tf.keras.models.load_model(str(MODEL_PATH))

def start_realtime_detection(camera_index=0):
    """Start real-time webcam feed with live emotion prediction overlays."""
    model = load_emotion_model()

    # Load OpenCV Haar Cascade face detector
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[Error] Could not open camera with index {camera_index}.")
        return

    print("\n=======================================================")
    print(" EmotionSense AI - Live Facial Emotion Detection")
    print(" Press 'q' to exit.")
    print("=======================================================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(40, 40)
        )

        for (x, y, w, h) in faces:
            # Extract Region of Interest (Face)
            roi_gray = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi_gray, (48, 48))
            roi_normalized = np.expand_dims(roi_resized, axis=(0, -1)) # Shape: (1, 48, 48, 1)

            # Predict Emotion
            preds = model.predict(roi_normalized, verbose=0)[0]
            top_idx = np.argmax(preds)
            emotion = EMOTIONS[top_idx]
            confidence = preds[top_idx] * 100
            color = EMOTION_COLORS.get(emotion, (0, 255, 0))

            # Draw stylish bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y - 30), (x + w, y), color, cv2.FILLED)
            label = f"{emotion}: {confidence:.1f}%"
            cv2.putText(frame, label, (x + 6, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # Draw probability mini-bars on the side
            bar_y = y + 15
            for idx, emo in enumerate(EMOTIONS):
                prob = preds[idx]
                bar_len = int(prob * 70)
                emo_col = EMOTION_COLORS.get(emo, (200, 200, 200))
                cv2.putText(frame, emo[:3], (x + w + 8, bar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.rectangle(frame, (x + w + 35, bar_y - 8), (x + w + 35 + bar_len, bar_y - 2), emo_col, -1)
                bar_y += 14

        cv2.imshow("EmotionSense AI - Real-Time Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_realtime_detection()
