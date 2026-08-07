from pathlib import Path
import tensorflow as tf


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "data" / "raw" / "fer2013" / "train"
TEST_DIR = BASE_DIR / "data" / "raw" / "fer2013" / "test"


IMG_SIZE = (48, 48)
BATCH_SIZE = 64
SEED = 42


def load_datasets():

    print("Loading training dataset...")
    print(f"Training path: {TRAIN_DIR}")

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        shuffle=True,
        seed=SEED
    )

    print("\nLoading testing dataset...")
    print(f"Testing path: {TEST_DIR}")

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        shuffle=False
    )

    print("\nEmotion Classes:")
    print(train_dataset.class_names)

    return train_dataset, test_dataset


if __name__ == "__main__":

    train_dataset, test_dataset = load_datasets()

    print("\nDataset loaded successfully!")

    print(
        "Training batches:",
        tf.data.experimental.cardinality(train_dataset).numpy()
    )

    print(
        "Testing batches:",
        tf.data.experimental.cardinality(test_dataset).numpy()
    )