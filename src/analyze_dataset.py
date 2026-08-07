from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "data" / "raw" / "fer2013" / "train"
OUTPUT_DIR = BASE_DIR / "outputs" / "graphs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Count images in each emotion folder

emotion_counts = {}

for emotion_folder in sorted(TRAIN_DIR.iterdir()):

    if emotion_folder.is_dir():

        image_count = len([
            file for file in emotion_folder.iterdir()
            if file.is_file()
        ])

        emotion_counts[emotion_folder.name] = image_count


print("\nDataset Distribution:")
print("-" * 30)

for emotion, count in emotion_counts.items():

    print(f"{emotion:<12} : {count}")


# --------------------------------------------------
# Plot distribution
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    emotion_counts.keys(),
    emotion_counts.values()
)

plt.title("FER2013 Training Dataset Distribution")

plt.xlabel("Emotion")

plt.ylabel("Number of Images")

plt.xticks(rotation=30)

plt.tight_layout()


output_path = OUTPUT_DIR / "class_distribution.png"

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print(f"\nSaved graph to:")
print(output_path)