import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report


# ============================================================
# ROOM ANALYSIS AI
# DATA SPLIT DIAGNOSTIC
# ============================================================

BASE_DIR = r"C:\Users\Daisi M\Desktop\RoomAnalysisAI\dataset"

MODEL_PATH = (
    r"C:\Users\Daisi M\Desktop\RoomAnalysisAI"
    r"\models\room_damage_cnn_v6_best.keras"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# LOAD ORIGINAL MODEL
# ============================================================

print("=" * 60)
print("LOADING V6 BEST CNN")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)

print("✅ Original model loaded!")


# ============================================================
# CHECK EACH DATASET
# ============================================================

datasets = [
    ("TRAIN", os.path.join(BASE_DIR, "train")),
    ("VALIDATION", os.path.join(BASE_DIR, "validation")),
    ("TEST", os.path.join(BASE_DIR, "test"))
]


for dataset_name, dataset_path in datasets:

    print("\n")
    print("=" * 60)
    print(f"{dataset_name} DATASET")
    print("=" * 60)

    dataset = tf.keras.utils.image_dataset_from_directory(

        dataset_path,

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        label_mode="int",

        shuffle=False
    )

    class_names = dataset.class_names

    true_labels = []
    predicted_labels = []

    for images, labels in dataset:

        predictions = model.predict(
            images,
            verbose=0
        )

        predicted = np.argmax(
            predictions,
            axis=1
        )

        true_labels.extend(
            labels.numpy()
        )

        predicted_labels.extend(
            predicted
        )

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    accuracy = np.mean(
        true_labels == predicted_labels
    )

    print(
        f"\n{dataset_name} accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print("\nClassification report:")

    print(
        classification_report(
            true_labels,
            predicted_labels,
            target_names=class_names,
            digits=3,
            zero_division=0
        )
    )


print("\n")
print("=" * 60)
print("✅ DATA SPLIT DIAGNOSTIC COMPLETED")
print("=" * 60)