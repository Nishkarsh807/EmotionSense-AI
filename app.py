"""
EmotionSense AI - Streamlit Web Dashboard
Interactive Facial Emotion Recognition & Analytics
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import streamlit as st
import tensorflow as tf
import plotly.express as px
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "models" / "best_emotion_model.keras"

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
EMOJI_MAP = {
    "Angry": "😠",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😄",
    "Neutral": "😐",
    "Sad": "😢",
    "Surprise": "😲"
}

# Streamlit Page Config
st.set_page_config(
    page_title="EmotionSense AI",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0284c7;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 24px;
    }
    .metric-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load model cached in memory."""
    target_path = MODEL_PATH if MODEL_PATH.exists() else BASE_DIR / "outputs" / "models" / "final_emotion_model.keras"
    if target_path.exists():
        return tf.keras.models.load_model(str(target_path))
    return None

def detect_and_predict_face(image_np, model):
    """Detect faces and predict emotion."""
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))

    results = []
    annotated_image = image_np.copy()

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, (48, 48))
        roi_input = np.expand_dims(roi_resized, axis=(0, -1))

        if model is not None:
            preds = model.predict(roi_input, verbose=0)[0]
        else:
            # Fallback mock probability for demonstration before training finishes
            preds = np.array([0.05, 0.02, 0.03, 0.75, 0.10, 0.03, 0.02])

        top_idx = int(np.argmax(preds))
        top_emotion = EMOTIONS[top_idx]
        confidence = float(preds[top_idx])

        # Draw box on image
        cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 229, 255), 3)
        cv2.putText(
            annotated_image,
            f"{top_emotion} ({confidence*100:.1f}%)",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 229, 255),
            2
        )

        results.append({
            "box": (x, y, w, h),
            "emotion": top_emotion,
            "confidence": confidence,
            "probabilities": preds
        })

    return annotated_image, results

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/facial-recognition-scan.png", width=70)
st.sidebar.title("EmotionSense AI")
st.sidebar.markdown("**Intelligent Facial Expression Analytics**")
nav = st.sidebar.radio("Navigation", ["📸 Image Emotion Analysis", "📊 Dataset & Architecture Insights", "📹 Live Webcam Mode"])

st.markdown('<div class="main-title">EmotionSense AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Deep Residual Facial Expression Recognition Platform (FER2013)</div>', unsafe_allow_html=True)

model = load_model()
if model is None:
    st.warning("⚠️ **Trained model not yet found at `outputs/models/best_emotion_model.keras`**. Run `python src/train.py` or use `train_emotion_model.ipynb` to train the deep Mini-Xception model.")

if nav == "📸 Image Emotion Analysis":
    st.subheader("Upload an Image for Emotion Recognition")
    uploaded_file = st.file_uploader("Choose a JPG, JPEG, or PNG image...", type=["jpg", "jpeg", "png"])

    col1, col2 = st.columns([1, 1])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        with col1:
            st.markdown("#### Input Image")
            st.image(image, use_column_width=True)

        # Run Face Detection & Inference
        annotated_img, detections = detect_and_predict_face(image_np, model)

        with col2:
            st.markdown("#### Detection Result")
            st.image(annotated_img, use_column_width=True)

            if len(detections) > 0:
                st.success(f"Detected **{len(detections)}** face(s) in image.")
                for i, det in enumerate(detections):
                    st.markdown(f"### Face #{i+1}: {EMOJI_MAP.get(det['emotion'], '')} **{det['emotion']}** ({det['confidence']*100:.1f}%)")

                    # Probability Chart
                    df_probs = pd.DataFrame({
                        "Emotion": EMOTIONS,
                        "Probability (%)": [p * 100 for p in det["probabilities"]]
                    })
                    fig = px.bar(
                        df_probs,
                        x="Probability (%)",
                        y="Emotion",
                        orientation="h",
                        color="Probability (%)",
                        color_continuous_scale="Blues",
                        text_auto=".1f"
                    )
                    fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No face detected in this image. Ensure good lighting and a frontal facial view.")

elif nav == "📊 Dataset & Architecture Insights":
    st.subheader("FER2013 Dataset & Deep Residual Architecture")

    tab1, tab2 = st.tabs(["Dataset Distribution", "Why Accuracy Was Low & How It Was Solved"])

    with tab1:
        st.markdown("### FER2013 Class Imbalance")
        st.write("Notice the massive skew: `happy` has 7,215 images while `disgust` has only 436.")
        dist_img_path = BASE_DIR / "outputs" / "graphs" / "class_distribution.png"
        if dist_img_path.exists():
            st.image(str(dist_img_path), use_column_width=True)

    with tab2:
        st.markdown("""
        ### Root Causes of Low Accuracy in Basic FER Models:
        1. **16x Class Imbalance:** Without balanced class weights, naive CNNs minimize loss by predicting majority classes (`happy`, `neutral`) and completely failing on minority classes (`disgust`, `fear`).
        2. **Dense Layer Overfitting:** Traditional architectures connect Conv layers to large 1024-node Dense layers, creating millions of parameters that memorize 48x48 pixel noise.
        3. **Ambiguous Human Labels:** Human agreement on FER2013 is estimated at only ~65%. Label smoothing ($0.08$) prevents overconfidence on noisy ground truths.
        4. **Lack of Data Augmentation:** Facial expressions vary in head tilt, scale, and lighting. Real-time rotations and zooms are critical.

        ### The Mini-Xception Solution:
        - **Depthwise Separable Convolutions:** Decouples spatial filtering from cross-channel correlation.
        - **Residual Skip Connections:** Preserves gradient flow throughout 20+ layers.
        - **Global Average Pooling:** Eliminates 95% of weights, cutting model size to 205K parameters.
        """)

elif nav == "📹 Live Webcam Mode":
    st.subheader("Real-Time Webcam Emotion Recognition")
    st.markdown("""
    To launch the ultra-fast real-time OpenCV desktop detection window with dynamic FPS and side probability bars, execute:
    ```bash
    python src/realtime_detect.py
    ```
    """)
    img_file_buffer = st.camera_input("Or take a live snapshot with your browser camera:")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer).convert("RGB")
        image_np = np.array(image)
        annotated_img, detections = detect_and_predict_face(image_np, model)
        st.image(annotated_img, use_column_width=True)
        if detections:
            top_det = detections[0]
            st.write(f"### Detected: {EMOJI_MAP.get(top_det['emotion'], '')} {top_det['emotion']} ({top_det['confidence']*100:.1f}%)")
