import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report


# ============================================================
# ROOM ANALYSIS AI - CNN V6 EVALUATION
# ============================================================

print("=" * 60)
print("ROOM ANALYSIS AI - CNN V6 EVALUATION")
print("=" * 60)


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"C:\Users\Daisi M\Desktop\RoomAnalysisAI"

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "room_damage_cnn_v6_best.keras"
)

TEST_DIR = os.path.join(
    PROJECT_DIR,
    "dataset",
    "test"
)


print("\nModel path:")
print(MODEL_PATH)

print("\nTest dataset:")
print(TEST_DIR)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading CNN V6 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("CNN V6 model loaded successfully!")


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\nLoading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int"
)


class_names = test_dataset.class_names

print("\nClasses detected:")

for index, class_name in enumerate(class_names):
    print(f"{index} -> {class_name}")


# ============================================================
# EVALUATE
# ============================================================

print("\nEvaluating CNN V6...")
print("-" * 60)

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)


print("\n" + "=" * 60)
print("CNN V6 TEST RESULTS")
print("=" * 60)

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy * 100:.2f}%"
)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(
    test_dataset,
    verbose=1
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)

actual_labels = np.concatenate(
    [
        labels.numpy()
        for _, labels in test_dataset
    ]
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    actual_labels,
    predicted_labels
)


print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(
    f"{'':20}",
    end=""
)

for class_name in class_names:
    print(
        f"{class_name:15}",
        end=""
    )

print()

for i, class_name in enumerate(class_names):

    print(
        f"{class_name:20}",
        end=""
    )

    for j in range(len(class_names)):

        print(
            f"{cm[i][j]:15}",
            end=""
        )

    print()


# ============================================================
# PER-CLASS RESULTS
# ============================================================

print("\n" + "=" * 60)
print("PER-CLASS RESULTS")
print("=" * 60)

for i, class_name in enumerate(class_names):

    total = np.sum(
        actual_labels == i
    )

    correct = cm[i][i]

    accuracy = (
        correct / total * 100
        if total > 0
        else 0
    )

    print(
        f"{class_name:20} "
        f"{correct}/{total} "
        f"({accuracy:.2f}%)"
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    actual_labels,
    predicted_labels,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(report)


# ============================================================
# FINAL
# ============================================================

print("=" * 60)
print("CNN V6 EVALUATION COMPLETED")
print("=" * 60)