import os
import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# ROOM ANALYSIS AI
# CNN V3 - CLASS WEIGHTED TRAINING
# ============================================================

print("=" * 60)
print("ROOM ANALYSIS AI - CNN V3")
print("=" * 60)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = r"C:\Users\Daisi M\Desktop\RoomAnalysisAI\dataset"

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "train"
)

VAL_DIR = os.path.join(
    BASE_DIR,
    "validation"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "test"
)

MODEL_DIR = r"C:\Users\Daisi M\Desktop\RoomAnalysisAI\models"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 2. PARAMETERS
# ============================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 20

SEED = 42


# ============================================================
# 3. LOAD TRAIN DATASET
# ============================================================

print("\nLoading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(

    TRAIN_DIR,

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="int",

    shuffle=True,

    seed=SEED
)


# ============================================================
# 4. LOAD VALIDATION DATASET
# ============================================================

print("\nLoading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(

    VAL_DIR,

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="int",

    shuffle=False
)


# ============================================================
# 5. LOAD TEST DATASET
# ============================================================

print("\nLoading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(

    TEST_DIR,

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    label_mode="int",

    shuffle=False
)


# ============================================================
# 6. CLASS NAMES
# ============================================================

class_names = train_dataset.class_names

print("\nClasses:")

for i, name in enumerate(class_names):

    print(
        f"{i} -> {name}"
    )


# ============================================================
# 7. CALCULATE CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

train_labels = []

for _, labels in train_dataset:

    train_labels.extend(
        labels.numpy()
    )


train_labels = np.array(
    train_labels
)


class_weights_array = compute_class_weight(

    class_weight="balanced",

    classes=np.unique(
        train_labels
    ),

    y=train_labels
)


class_weights = {
    i: float(weight)
    for i, weight in enumerate(
        class_weights_array
    )
}


print("\nClass weights:")

for i, name in enumerate(class_names):

    print(
        f"{name:15} : "
        f"{class_weights[i]:.4f}"
    )


# ============================================================
# 8. PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    AUTOTUNE
)


# ============================================================
# 9. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip(
        "horizontal"
    ),

    tf.keras.layers.RandomRotation(
        0.08
    ),

    tf.keras.layers.RandomZoom(
        0.10
    ),

    tf.keras.layers.RandomContrast(
        0.10
    )

], name="data_augmentation")


# ============================================================
# 10. BUILD CNN V3
# ============================================================

model = tf.keras.Sequential([

    data_augmentation,

    tf.keras.layers.Rescaling(
        1.0 / 255
    ),


    # ------------------------------
    # BLOCK 1
    # ------------------------------

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),


    # ------------------------------
    # BLOCK 2
    # ------------------------------

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),


    # ------------------------------
    # BLOCK 3
    # ------------------------------

    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),


    # ------------------------------
    # BLOCK 4
    # ------------------------------

    tf.keras.layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),


    # ------------------------------
    # CLASSIFICATION
    # ------------------------------

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.5
    ),

    tf.keras.layers.Dense(
        len(class_names),
        activation="softmax"
    )

])


# ============================================================
# 11. COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# 12. MODEL SUMMARY
# ============================================================

print("\nCNN V3 Model Summary:\n")

model.summary()


# ============================================================
# 13. CALLBACKS
# ============================================================

best_model_path = os.path.join(

    MODEL_DIR,

    "room_damage_cnn_v3_best.keras"
)


callbacks = [

    tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=5,

        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(

        best_model_path,

        monitor="val_accuracy",

        save_best_only=True,

        mode="max"
    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=2,

        min_lr=1e-7
    )

]


# ============================================================
# 14. TRAIN
# ============================================================

print("\n")
print("=" * 60)
print("STARTING CNN V3 TRAINING")
print("=" * 60)


history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks
)


# ============================================================
# 15. TEST EVALUATION
# ============================================================

print("\n")
print("=" * 60)
print("EVALUATING CNN V3")
print("=" * 60)


test_loss, test_accuracy = model.evaluate(
    test_dataset
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
# 16. SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(

    MODEL_DIR,

    "room_damage_cnn_v3.keras"
)


model.save(
    final_model_path
)


print("\n")
print("=" * 60)

print(
    "✅ CNN V3 TRAINING COMPLETED!"
)

print("=" * 60)

print(
    "\nBest model:"
)

print(
    best_model_path
)

print(
    "\nFinal model:"
)

print(
    final_model_path
)

print("\n🎉 CNN V3 is ready!")