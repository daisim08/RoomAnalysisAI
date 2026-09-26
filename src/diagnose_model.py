import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = r"C:\Users\Daisi M\Desktop\RoomAnalysisAI\dataset"

TEST_DIR = os.path.join(
    BASE_DIR,
    "test"
)

MODEL_PATH = (
    r"C:\Users\Daisi M\Desktop\RoomAnalysisAI"
    r"\models\room_damage_cnn_v5_best.keras"
)


# ============================================================
# 2. SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# 3. LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(

    TEST_DIR,

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="int",

    shuffle=False
)


class_names = test_dataset.class_names


print("\nClasses:")

for i, name in enumerate(class_names):

    print(
        f"{i} -> {name}"
    )


# ============================================================
# 4. LOAD MODEL
# ============================================================

print("\nLoading CNN V2 model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("✅ Model loaded successfully!")


# ============================================================
# 5. GET TRUE LABELS AND PREDICTIONS
# ============================================================

true_labels = []
predicted_labels = []


print("\nGenerating predictions...")


for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    true_labels.extend(
        labels.numpy()
    )

    predicted_labels.extend(
        predicted_classes
    )


true_labels = np.array(
    true_labels
)

predicted_labels = np.array(
    predicted_labels
)


# ============================================================
# 6. PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)

print(
    "PREDICTION DISTRIBUTION"
)

print("=" * 60)


for i, class_name in enumerate(class_names):

    count = np.sum(
        predicted_labels == i
    )

    percentage = (
        count /
        len(predicted_labels)
    ) * 100

    print(
        f"{class_name:15} : "
        f"{count:3} images "
        f"({percentage:.2f}%)"
    )


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)


print("\n" + "=" * 60)

print(
    "CONFUSION MATRIX"
)

print("=" * 60)

print(
    "\nRows = Actual class"
)

print(
    "Columns = Predicted class\n"
)


print(
    "              ",
    end=""
)

for name in class_names:

    print(
        f"{name[:12]:>14}",
        end=""
    )

print()


for i, row in enumerate(cm):

    print(
        f"{class_names[i]:15}",
        end=""
    )

    for value in row:

        print(
            f"{value:14}",
            end=""
        )

    print()


# ============================================================
# 8. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)

print(
    "CLASSIFICATION REPORT"
)

print("=" * 60)

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 9. OVERALL ACCURACY
# ============================================================

accuracy = np.mean(
    true_labels == predicted_labels
)


print(
    f"Overall Accuracy: "
    f"{accuracy * 100:.2f}%"
)


print("\n" + "=" * 60)

print(
    "✅ MODEL DIAGNOSIS COMPLETED"
)

print("=" * 60)