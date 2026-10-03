# EmotionSense AI: Intelligent Real-Time Emotion Analytics Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Dashboard-red.svg)](https://streamlit.io)

**EmotionSense AI** is a deep learning-based facial emotion recognition system trained on the FER2013 dataset. It classifies human facial expressions into 7 categories in real time using an optimized Deep Residual Mini-Xception architecture.

---

## 📈 Why Baseline FER Models Suffer from Low Accuracy & How We Solved It

Standard CNN models on the FER2013 dataset often get stuck at **38%–48% accuracy** due to four structural bottlenecks:

1. **Severe Class Imbalance (16:1):**  
   `happy` contains 7,215 images, while `disgust` has only 436. Without balancing, models default to majority classes.  
   **Fix:** Automated calculation of balanced class weights ($w_c = \frac{N}{K \cdot N_c}$) during backpropagation.
2. **Dense Layer Overfitting on 48×48 Faces:**  
   Traditional CNNs connect feature maps to large dense layers (1024+ neurons), causing 15M+ parameters to memorize low-resolution noise.  
   **Fix:** **Mini-Xception Architecture** utilizing Depthwise Separable Convolutions and Global Average Pooling, reducing parameters to just **205K** while drastically boosting generalization.
3. **Overconfidence on Ambiguous Human Annotations:**  
   Human annotator agreement on FER2013 is estimated at only ~65%.  
   **Fix:** Label smoothing loss ($0.08$) prevents the network from over-fitting to noisy ground-truth labels.
4. **Lack of Dynamic Spatial Regularization:**  
   **Fix:** On-the-fly data augmentation (random horizontal flips, rotations $\pm 12^\circ$, zoom, contrast adjustment) coupled with residual skip connections and `ReduceLROnPlateau`.

### 🔬 Architecture Comparison

| Model Architecture | Parameter Count | Overfitting Risk | Expected Test Accuracy |
| :--- | :---: | :---: | :---: |
| Naive Plain CNN (Conv + Dense) | ~12M - 20M | Very High | 38% – 48% |
| Standard VGG-16 | ~15M | High | 52% – 58% |
| **EmotionSense Mini-Xception (Ours)** | **~205K** | **Low (Residual + GAP)** | **66% – 70%+** |

*(Note: Human baseline accuracy on FER2013 is estimated at 65% ± 5% due to resolution and subjective facial expressions).*

---

## 🎭 7 Emotions Detected

- 😠 **Angry**
- 🤢 **Disgust**
- 😨 **Fear**
- 😄 **Happy**
- 😐 **Neutral**
- 😢 **Sad**
- 😲 **Surprise**

---

## 📂 Project Structure

```
EmotionSense-AI/
├── data/
│   └── raw/
│       └── fer2013/
│           ├── train/                  # 28,709 training images across 7 classes
│           └── test/                   # 7,178 test images across 7 classes
├── outputs/
│   ├── graphs/                         # Distribution plots, training curves, confusion matrix
│   └── models/                         # Serialized trained model weights (.keras)
├── src/
│   ├── analyze_dataset.py              # Dataset statistics & class distribution analysis
│   ├── dataset.py                      # TensorFlow dataset loader
│   ├── model.py                        # Deep Mini-Xception residual architecture
│   ├── realtime_detect.py              # OpenCV real-time live webcam detector
│   ├── train.py                        # Advanced training script with class weights & callbacks
│   └── visualize.py                    # Sample face visualization
├── app.py                              # Streamlit interactive web dashboard
├── train_emotion_model.ipynb           # Complete Jupyter / Google Colab training notebook
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation
```

---

## 🚀 Quickstart & Usage

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Train the High-Accuracy Model
- **Option A (Google Colab with Free T4 GPU - Recommended for speed):**  
  Open `train_emotion_model.ipynb` in Google Colab, select GPU Runtime, and run all cells (takes ~6-8 minutes).
- **Option B (Local Terminal):**  
  ```bash
  python src/train.py
  ```

### 3. Launch Real-Time Webcam Detection
```bash
python src/realtime_detect.py
```
*Press `q` to quit the webcam feed.*

### 4. Launch Streamlit Web Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to upload photos or view dataset analytics!
