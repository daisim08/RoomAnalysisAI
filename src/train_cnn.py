import tensorflow as tf
from tensorflow.keras import layers, models
import os

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = r"C:\Users\Daisi M\Desktop\RoomAnalysisAI\dataset"

TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")


# ============================================================
# 2. BASIC SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4
EPOCHS = 10


# ============================================================
# 3. LOAD TRAINING DATA
# ============================================================

print("\nLoading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=42
)


# ============================================================
# 4. LOAD TESTING DATA
# ============================================================

print("\nLoading testing dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)


# ============================================================
# 5. DISPLAY CLASS NAMES
# ============================================================

class_names = train_dataset.class_names

print("\nClasses detected by TensorFlow:")

for index, name in enumerate(class_names):
    print(index, "->", name)


# ============================================================
# 6. CREATE CNN MODEL
# ============================================================

model = models.Sequential([

    # Input
    layers.Input(shape=(224, 224, 3)),

    # Normalize pixels from 0-255 to 0-1
    layers.Rescaling(1.0 / 255),

    # -------------------------
    # Convolution Block 1
    # -------------------------
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # -------------------------
    # Convolution Block 2
    # -------------------------
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # -------------------------
    # Convolution Block 3
    # -------------------------
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # -------------------------
    # Convert feature maps
    # -------------------------
    layers.Flatten(),

    # Fully connected layer
    layers.Dense(
        128,
        activation="relu"
    ),

    # Dropout helps reduce overfitting
    layers.Dropout(0.5),

    # Output layer
    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# ============================================================
# 7. DISPLAY MODEL
# ============================================================

print("\nCNN Model Summary:\n")

model.summary()


# ============================================================
# 8. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 9. TRAIN MODEL
# ============================================================

print("\nStarting CNN training...\n")

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS
)


# ============================================================
# 10. EVALUATE MODEL
# ============================================================

print("\nEvaluating model...\n")

test_loss, test_accuracy = model.evaluate(
    test_dataset
)

print(
    f"\nTest Accuracy: {test_accuracy * 100:.2f}%"
)

print(
    f"Test Loss: {test_loss:.4f}"
)


# ============================================================
# 11. SAVE MODEL
# ============================================================

MODEL_DIR = "models"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "room_damage_cnn.keras"
)

model.save(MODEL_PATH)

print(
    f"\n✅ Model saved successfully at:"
)

print(MODEL_PATH)

print("\n🎉 CNN training completed!")