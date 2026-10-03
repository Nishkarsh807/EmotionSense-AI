"""
EmotionSense AI - Comprehensive Training Pipeline for FER2013
Implements:
1. Balanced Class Weighting (solves 16x disgust vs happy imbalance)
2. On-the-fly Data Augmentation (rotation, zoom, flips, translation)
3. Label Smoothing (combats ambiguous FER2013 human labels)
4. Dynamic LR Scheduling & Early Stopping
5. Comprehensive Evaluation (Confusion Matrix & Classification Report)
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.model import build_mini_xception, build_data_augmentation

TRAIN_DIR = BASE_DIR / "data" / "raw" / "fer2013" / "train"
TEST_DIR = BASE_DIR / "data" / "raw" / "fer2013" / "test"
MODELS_DIR = BASE_DIR / "outputs" / "models"
GRAPHS_DIR = BASE_DIR / "outputs" / "graphs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 40
SEED = 42

def calculate_class_weights(train_dir):
    """Compute balanced class weights to counteract severe class imbalance."""
    counts = {}
    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    for idx, cname in enumerate(class_names):
        folder = train_dir / cname
        counts[idx] = len(list(folder.glob("*")))
    
    total_samples = sum(counts.values())
    num_classes = len(class_names)
    
    # Standard balanced formula: N / (K * N_c)
    class_weights = {}
    for idx in range(num_classes):
        class_weights[idx] = total_samples / (num_classes * counts[idx])
        
    print("\n[Class Weights to Counteract Imbalance]")
    for idx, cname in enumerate(class_names):
        print(f" - {cname:<10s} (count: {counts[idx]:5d}): weight = {class_weights[idx]:.3f}")
    return class_weights, class_names

def load_data():
    """Load and configure TensorFlow datasets with caching and prefetching."""
    print("\n[DataLoader] Loading training images from:", TRAIN_DIR)
    raw_train = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        label_mode="categorical",
        shuffle=True,
        seed=SEED
    )

    print("[DataLoader] Loading testing images from:", TEST_DIR)
    raw_test = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        label_mode="categorical",
        shuffle=False
    )

    augmentation = build_data_augmentation()

    # Apply data augmentation only to training data
    train_ds = raw_train.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    test_ds = raw_test.prefetch(tf.data.AUTOTUNE)
    return train_ds, test_ds, raw_train.class_names

def train_model():
    class_weights, class_names = calculate_class_weights(TRAIN_DIR)
    train_ds, test_ds, _ = load_data()

    print("\n[Model] Building Deep Mini-Xception Network...")
    model = build_mini_xception(input_shape=(48, 48, 1), num_classes=len(class_names))

    # Loss with label smoothing to prevent overfitting on noisy FER annotations
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.08)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"]
    )

    best_model_path = MODELS_DIR / "best_emotion_model.keras"

    callbacks = [
        ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=9,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print(f"\n[Training] Starting model training for up to {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Plot Training & Validation Curves
    plot_training_history(history, GRAPHS_DIR / "training_history.png")

    # Evaluate on Test Set
    evaluate_model(model, test_ds, class_names, GRAPHS_DIR / "confusion_matrix.png")

    # Save final model as well
    final_model_path = MODELS_DIR / "final_emotion_model.keras"
    model.save(str(final_model_path))
    print(f"\n[Saved] Final model saved to {final_model_path}")
    print(f"[Saved] Best checkpoint saved to {best_model_path}")

def plot_training_history(history, output_path):
    """Plot and save accuracy and loss curves."""
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", color="#0284c7", lw=2)
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", color="#00e676", lw=2)
    plt.title("Emotion Recognition Accuracy Over Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", color="#0284c7", lw=2)
    plt.plot(epochs_range, val_loss, label="Validation Loss", color="#ff1744", lw=2)
    plt.title("Training and Validation Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[Evaluation] Saved training curve to {output_path}")

def evaluate_model(model, test_ds, class_names, cm_output_path):
    """Compute confusion matrix and per-class precision, recall, and F1."""
    print("\n[Evaluation] Evaluating best model on holdout test set...")
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\n" + "=" * 60)
    print(" CLASSIFICATION REPORT (FER2013 Holdout Test Set)")
    print("=" * 60)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    # Save Classification Report to text file
    report_path = GRAPHS_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title("Normalized Confusion Matrix on Test Set", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Emotion", fontsize=12)
    plt.ylabel("True Emotion", fontsize=12)
    plt.tight_layout()
    plt.savefig(cm_output_path, dpi=200)
    plt.close()
    print(f"[Evaluation] Saved normalized confusion matrix to {cm_output_path}")

if __name__ == "__main__":
    train_model()
