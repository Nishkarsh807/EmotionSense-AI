from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "data" / "raw" / "fer2013" / "train"
OUTPUT_DIR = BASE_DIR / "outputs" / "graphs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMG_SIZE = (48, 48)
BATCH_SIZE = 64
SEED = 42


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    shuffle=True,
    seed=SEED
)

class_names = dataset.class_names

print("Emotion Classes:")
print(class_names)


# --------------------------------------------------
# Display sample images
# --------------------------------------------------

plt.figure(figsize=(10, 8))

for images, labels in dataset.take(1):

    for i in range(min(16, len(images))):

        plt.subplot(4, 4, i + 1)

        plt.imshow(
            images[i].numpy().squeeze(),
            cmap="gray"
        )

        plt.title(
            class_names[labels[i].numpy()]
        )

        plt.axis("off")


plt.tight_layout()

output_path = OUTPUT_DIR / "sample_emotions.png"

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print(f"\nSaved visualization to:")
print(output_path)