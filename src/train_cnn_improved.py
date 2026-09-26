import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = r"C:\Users\Daisi M\Desktop\RoomAnalysisAI\dataset"

TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")


# ============================================================
# 2. SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4
EPOCHS = 20


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
# 5. CLASS NAMES
# ============================================================

class_names = train_dataset.class_names

print("\nClasses detected:")

for index, name in enumerate(class_names):
    print(index, "->", name)


# ============================================================
# 6. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip(
        "horizontal"
    ),

    layers.RandomRotation(
        0.10
    ),

    layers.RandomZoom(
        0.10
    ),

    layers.RandomContrast(
        0.10
    )

], name="data_augmentation")


# ============================================================
# 7. CREATE IMPROVED CNN
# ============================================================

model = models.Sequential([

    layers.Input(
        shape=(224, 224, 3)
    ),

    # Data augmentation
    data_augmentation,

    # Normalize pixels
    layers.Rescaling(
        1.0 / 255
    ),


    # --------------------------------------------------------
    # CONVOLUTION BLOCK 1
    # --------------------------------------------------------

    layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # --------------------------------------------------------
    # CONVOLUTION BLOCK 2
    # --------------------------------------------------------

    layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # --------------------------------------------------------
    # CONVOLUTION BLOCK 3
    # --------------------------------------------------------

    layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # --------------------------------------------------------
    # CONVOLUTION BLOCK 4
    # --------------------------------------------------------

    layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # --------------------------------------------------------
    # GLOBAL AVERAGE POOLING
    # --------------------------------------------------------

    layers.GlobalAveragePooling2D(),


    # --------------------------------------------------------
    # FULLY CONNECTED LAYER
    # --------------------------------------------------------

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(
        0.5
    ),


    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )

])


# ============================================================
# 8. DISPLAY MODEL
# ============================================================

print("\nImproved CNN Model Summary:\n")

model.summary()


# ============================================================
# 9. COMPILE MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)


# ============================================================
# 10. CREATE MODELS DIRECTORY
# ============================================================

MODEL_DIR = r"C:\Users\Daisi M\Desktop\RoomAnalysisAI\models"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 11. CALLBACKS
# ============================================================

best_model_path = os.path.join(
    MODEL_DIR,
    "room_damage_cnn_best.keras"
)


early_stopping = callbacks.EarlyStopping(

    monitor="val_loss",

    patience=4,

    restore_best_weights=True

)


model_checkpoint = callbacks.ModelCheckpoint(

    best_model_path,

    monitor="val_accuracy",

    save_best_only=True,

    mode="max"

)


# ============================================================
# 12. TRAIN MODEL
# ============================================================

print("\nStarting improved CNN training...\n")

history = model.fit(

    train_dataset,

    validation_data=test_dataset,

    epochs=EPOCHS,

    callbacks=[
        early_stopping,
        model_checkpoint
    ]

)


# ============================================================
# 13. EVALUATE MODEL
# ============================================================

print("\nEvaluating improved CNN...\n")

test_loss, test_accuracy = model.evaluate(
    test_dataset
)


print(
    f"\nImproved Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Improved Test Loss: "
    f"{test_loss:.4f}"
)


# ============================================================
# 14. SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "room_damage_cnn_improved.keras"
)

model.save(
    final_model_path
)


print(
    "\n✅ Improved CNN saved at:"
)

print(
    final_model_path
)


print(
    "\n🎉 Improved CNN training completed!"
)