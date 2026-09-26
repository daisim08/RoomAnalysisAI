import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# ============================================================
# ROOM ANALYSIS AI - CNN V4
# ============================================================

print("=" * 60)
print("ROOM ANALYSIS AI - CNN V4")
print("=" * 60)

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

PROJECT_DIR = r"C:\Users\Daisi M\Desktop\RoomAnalysisAI"

DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "validation")
TEST_DIR = os.path.join(DATASET_DIR, "test")

MODEL_DIR = os.path.join(PROJECT_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 25

SEED = 42

NUM_CLASSES = 4

# ------------------------------------------------------------
# LOAD TRAINING DATA
# ------------------------------------------------------------

print("\nLoading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

# ------------------------------------------------------------
# LOAD VALIDATION DATA
# ------------------------------------------------------------

print("\nLoading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ------------------------------------------------------------
# LOAD TEST DATA
# ------------------------------------------------------------

print("\nLoading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ------------------------------------------------------------
# CLASS NAMES
# ------------------------------------------------------------

class_names = train_dataset.class_names

print("\nClasses detected by TensorFlow:")

for index, class_name in enumerate(class_names):
    print(f"{index} -> {class_name}")

# ------------------------------------------------------------
# PERFORMANCE OPTIMIZATION
# ------------------------------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)

validation_dataset = validation_dataset.prefetch(AUTOTUNE)

test_dataset = test_dataset.prefetch(AUTOTUNE)

# ------------------------------------------------------------
# CALCULATE CLASS WEIGHTS
# ------------------------------------------------------------

print("\nCalculating class weights...")

train_labels = []

for images, labels in train_dataset.unbatch():

    train_labels.append(int(labels.numpy()))

train_labels = np.array(train_labels)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_labels
)

class_weights = {
    i: float(class_weights_array[i])
    for i in range(NUM_CLASSES)
}

print("\nClass weights:")

for i, class_name in enumerate(class_names):

    print(
        f"{class_name:<15}: "
        f"{class_weights[i]:.4f}"
    )

# ------------------------------------------------------------
# DATA AUGMENTATION
# ------------------------------------------------------------

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.08
        ),

        layers.RandomZoom(
            0.10
        ),

        layers.RandomContrast(
            0.10
        ),
    ],
    name="data_augmentation"
)

# ------------------------------------------------------------
# BUILD CNN V4
# ------------------------------------------------------------

model = keras.Sequential(
    [

        keras.Input(
            shape=(224, 224, 3)
        ),

        # Data augmentation
        data_augmentation,

        # Normalize pixels
        layers.Rescaling(
            1.0 / 255
        ),

        # ----------------------------------------------------
        # BLOCK 1
        # ----------------------------------------------------

        layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

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

        layers.Dropout(
            0.20
        ),

        # ----------------------------------------------------
        # BLOCK 2
        # ----------------------------------------------------

        layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

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

        layers.Dropout(
            0.25
        ),

        # ----------------------------------------------------
        # BLOCK 3
        # ----------------------------------------------------

        layers.Conv2D(
            128,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

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

        layers.Dropout(
            0.30
        ),

        # ----------------------------------------------------
        # BLOCK 4
        # ----------------------------------------------------

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

        layers.Dropout(
            0.30
        ),

        # ----------------------------------------------------
        # CLASSIFICATION HEAD
        # ----------------------------------------------------

        layers.GlobalAveragePooling2D(),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.Dropout(
            0.40
        ),

        layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ],
    name="RoomDamageCNN_V4"
)

# ------------------------------------------------------------
# MODEL SUMMARY
# ------------------------------------------------------------

print("\nCNN V4 Model Summary:\n")

model.summary()

# ------------------------------------------------------------
# COMPILE MODEL
# ------------------------------------------------------------

model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)

# ------------------------------------------------------------
# CALLBACKS
# ------------------------------------------------------------

best_model_path = os.path.join(
    MODEL_DIR,
    "room_damage_cnn_v4_best.keras"
)

final_model_path = os.path.join(
    MODEL_DIR,
    "room_damage_cnn_v4.keras"
)

callbacks = [

    keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        mode="max",
        restore_best_weights=True,
        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# ------------------------------------------------------------
# TRAIN MODEL
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STARTING CNN V4 TRAINING")
print("=" * 60)

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks
)

# ------------------------------------------------------------
# EVALUATE TEST DATASET
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("EVALUATING CNN V4")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)

print(
    f"\nFINAL TEST ACCURACY: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"FINAL TEST LOSS: "
    f"{test_loss:.4f}"
)

# ------------------------------------------------------------
# SAVE FINAL MODEL
# ------------------------------------------------------------

model.save(final_model_path)

# ------------------------------------------------------------
# FINAL OUTPUT
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("✅ CNN V4 TRAINING COMPLETED!")
print("=" * 60)

print("\nBest model:")
print(best_model_path)

print("\nFinal model:")
print(final_model_path)

print("\n🎉 CNN V4 is ready!")