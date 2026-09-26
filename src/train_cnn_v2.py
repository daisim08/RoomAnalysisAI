import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = r"C:\Users\Daisi M\Desktop\RoomAnalysisAI\dataset"

TRAIN_DIR = os.path.join(BASE_DIR, "train")
VALIDATION_DIR = os.path.join(BASE_DIR, "validation")
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
# 4. LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)


# ============================================================
# 5. LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)


# ============================================================
# 6. DISPLAY CLASSES
# ============================================================

class_names = train_dataset.class_names

print("\nClasses detected by TensorFlow:")

for index, name in enumerate(class_names):
    print(index, "->", name)


# ============================================================
# 7. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip("horizontal"),

    layers.RandomRotation(0.05),

    layers.RandomZoom(0.05)

], name="data_augmentation")


# ============================================================
# 8. CNN MODEL
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
    # BLOCK 1
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
    # BLOCK 2
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
    # BLOCK 3
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
    # GLOBAL AVERAGE POOLING
    # --------------------------------------------------------

    layers.GlobalAveragePooling2D(),


    # --------------------------------------------------------
    # DENSE LAYER
    # --------------------------------------------------------

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(
        0.4
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
# 9. MODEL SUMMARY
# ============================================================

print("\nCNN V2 Model Summary:\n")

model.summary()


# ============================================================
# 10. COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)


# ============================================================
# 11. MODEL DIRECTORY
# ============================================================

MODEL_DIR = r"C:\Users\Daisi M\Desktop\RoomAnalysisAI\models"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 12. CALLBACKS
# ============================================================

best_model_path = os.path.join(
    MODEL_DIR,
    "room_damage_cnn_v2_best.keras"
)


early_stopping = callbacks.EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True

)


model_checkpoint = callbacks.ModelCheckpoint(

    best_model_path,

    monitor="val_accuracy",

    save_best_only=True,

    mode="max"

)


# ============================================================
# 13. TRAIN
# ============================================================

print("\nStarting CNN V2 training...\n")

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    callbacks=[
        early_stopping,
        model_checkpoint
    ]

)


# ============================================================
# 14. FINAL TEST EVALUATION
# ============================================================

print("\nEvaluating CNN V2 on TEST dataset...\n")

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
# 15. SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "room_damage_cnn_v2.keras"
)

model.save(
    final_model_path
)


print("\n" + "=" * 60)

print("✅ CNN V2 TRAINING COMPLETED!")

print("=" * 60)

print("\nBest model:")
print(best_model_path)

print("\nFinal model:")
print(final_model_path)