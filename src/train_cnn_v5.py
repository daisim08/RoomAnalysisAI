import os
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# ROOM ANALYSIS AI - CNN V5
# Stable CNN with controlled augmentation
# ============================================================

print("=" * 60)
print("ROOM ANALYSIS AI - CNN V5")
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
EPOCHS = 20
SEED = 42


# ------------------------------------------------------------
# LOAD TRAINING DATA
# ------------------------------------------------------------

print("\nLoading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    label_mode="int"
)


# ------------------------------------------------------------
# LOAD VALIDATION DATA
# ------------------------------------------------------------

print("\nLoading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int"
)


# ------------------------------------------------------------
# LOAD TEST DATA
# ------------------------------------------------------------

print("\nLoading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int"
)


# ------------------------------------------------------------
# CLASS NAMES
# ------------------------------------------------------------

class_names = train_dataset.class_names

print("\nClasses detected by TensorFlow:")

for index, class_name in enumerate(class_names):
    print(f"{index} -> {class_name}")


NUM_CLASSES = len(class_names)


# ------------------------------------------------------------
# PERFORMANCE
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

for _, labels in train_dataset.unbatch():
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


# ============================================================
# DATA AUGMENTATION
# ============================================================
#
# IMPORTANT:
# Validation and test datasets are NOT augmented.
#
# Only the training images go through augmentation.
# ============================================================

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.03),
        layers.RandomZoom(0.05),
    ],
    name="data_augmentation"
)


# ============================================================
# BUILD CNN V5
# ============================================================

model = keras.Sequential(
    [

        keras.Input(
            shape=(224, 224, 3)
        ),

        # ----------------------------------------------------
        # TRAINING AUGMENTATION
        # ----------------------------------------------------

        data_augmentation,

        # ----------------------------------------------------
        # NORMALIZATION
        # ----------------------------------------------------

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

        layers.MaxPooling2D(
            (2, 2)
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

        layers.MaxPooling2D(
            (2, 2)
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

        layers.MaxPooling2D(
            (2, 2)
        ),

        # ----------------------------------------------------
        # REGULARIZATION
        # ----------------------------------------------------

        layers.Dropout(
            0.25
        ),

        # ----------------------------------------------------
        # CLASSIFICATION HEAD
        # ----------------------------------------------------

        layers.GlobalAveragePooling2D(),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(
            0.30
        ),

        layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )

    ],
    name="RoomDamageCNN_V5"
)


# ------------------------------------------------------------
# MODEL SUMMARY
# ------------------------------------------------------------

print("\nCNN V5 Model Summary:\n")

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
# MODEL PATHS
# ------------------------------------------------------------

best_model_path = os.path.join(
    MODEL_DIR,
    "room_damage_cnn_v5_best.keras"
)

final_model_path = os.path.join(
    MODEL_DIR,
    "room_damage_cnn_v5.keras"
)


# ------------------------------------------------------------
# CALLBACKS
# ------------------------------------------------------------

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
        patience=5,
        mode="max",
        restore_best_weights=True,
        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

]


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 60)
print("STARTING CNN V5 TRAINING")
print("=" * 60)

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks

)


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("EVALUATING CNN V5")
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


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(final_model_path)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("CNN V5 TRAINING COMPLETED!")
print("=" * 60)

print("\nBest model:")
print(best_model_path)

print("\nFinal model:")
print(final_model_path)

print("\nDataset classes:")

for i, class_name in enumerate(class_names):
    print(f"{i} -> {class_name}")

print("\nTraining finished successfully.")