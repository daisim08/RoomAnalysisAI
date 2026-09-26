import os
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# ROOM ANALYSIS AI - CNN V6
# Transfer Learning with MobileNetV2
# ============================================================

print("=" * 60)
print("ROOM ANALYSIS AI - CNN V6")
print("=" * 60)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = r"C:\Users\Daisi M\Desktop\RoomAnalysisAI"

DATASET_DIR = os.path.join(
    PROJECT_DIR,
    "dataset"
)

TRAIN_DIR = os.path.join(
    DATASET_DIR,
    "train"
)

VALIDATION_DIR = os.path.join(
    DATASET_DIR,
    "validation"
)

TEST_DIR = os.path.join(
    DATASET_DIR,
    "test"
)

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 10
SEED = 42


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    label_mode="int"
)


print("\nLoading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int"
)


print("\nLoading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int"
)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = train_dataset.class_names

print("\nClasses detected:")

for index, class_name in enumerate(class_names):
    print(
        f"{index} -> {class_name}"
    )


NUM_CLASSES = len(class_names)


# ============================================================
# PERFORMANCE
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
# CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

train_labels = []

for _, labels in train_dataset.unbatch():

    train_labels.append(
        int(labels.numpy())
    )


train_labels = np.array(
    train_labels
)


class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_labels
)


class_weights = {
    i: float(
        class_weights_array[i]
    )
    for i in range(NUM_CLASSES)
}


print("\nClass weights:")

for i, class_name in enumerate(class_names):

    print(
        f"{class_name:15} : "
        f"{class_weights[i]:.4f}"
    )


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential(
    [

        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.05
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


# ============================================================
# MOBILE NET V2 BASE MODEL
# ============================================================

print("\nLoading MobileNetV2...")

base_model = tf.keras.applications.MobileNetV2(

    input_shape=(
        224,
        224,
        3
    ),

    include_top=False,

    weights="imagenet"
)


# Freeze pretrained layers initially

base_model.trainable = False


# ============================================================
# BUILD CNN V6
# ============================================================

inputs = keras.Input(
    shape=(
        224,
        224,
        3
    )
)


x = data_augmentation(
    inputs
)


# MobileNetV2 preprocessing

x = tf.keras.applications.mobilenet_v2.preprocess_input(
    x
)


x = base_model(
    x,
    training=False
)


x = layers.GlobalAveragePooling2D()(x)


x = layers.Dense(
    128,
    activation="relu"
)(x)


x = layers.Dropout(
    0.40
)(x)


outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)


model = keras.Model(
    inputs,
    outputs,
    name="RoomDamageCNN_V6"
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\nCNN V6 Model Summary:")

model.summary()


# ============================================================
# COMPILE - INITIAL TRAINING
# ============================================================

model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# MODEL PATH
# ============================================================

best_model_path = os.path.join(

    MODEL_DIR,

    "room_damage_cnn_v6_best.keras"

)


final_model_path = os.path.join(

    MODEL_DIR,

    "room_damage_cnn_v6.keras"

)


# ============================================================
# CALLBACKS
# ============================================================

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
# INITIAL TRAINING
# ============================================================

print("\n" + "=" * 60)

print(
    "STARTING CNN V6 INITIAL TRAINING"
)

print("=" * 60)


history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=INITIAL_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks

)


# ============================================================
# FINE TUNING
# ============================================================

print("\n" + "=" * 60)

print(
    "STARTING CNN V6 FINE-TUNING"
)

print("=" * 60)


# Unfreeze MobileNetV2

base_model.trainable = True


# Keep early layers frozen

for layer in base_model.layers[:-30]:

    layer.trainable = False


# Recompile with lower learning rate

model.compile(

    optimizer=keras.optimizers.Adam(

        learning_rate=1e-5

    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)


history_fine = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=FINE_TUNE_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks

)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best CNN V6 model...")

best_model = keras.models.load_model(
    best_model_path
)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n" + "=" * 60)

print(
    "EVALUATING CNN V6 ON TEST DATA"
)

print("=" * 60)


test_loss, test_accuracy = best_model.evaluate(

    test_dataset,

    verbose=1

)


print(
    f"\nCNN V6 TEST ACCURACY: "
    f"{test_accuracy * 100:.2f}%"
)


print(
    f"CNN V6 TEST LOSS: "
    f"{test_loss:.4f}"
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

best_model.save(
    final_model_path
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)

print(
    "CNN V6 TRAINING COMPLETED!"
)

print("=" * 60)


print("\nBest model:")

print(
    best_model_path
)


print("\nFinal model:")

print(
    final_model_path
)


print("\nClasses:")

for i, class_name in enumerate(
    class_names
):

    print(
        f"{i} -> {class_name}"
    )


print("\nTraining finished successfully.")