import tensorflow as tf
import numpy as np
from PIL import Image

# ============================================================
# ROOM ANALYSIS AI - TEST TRAINED CNN
# ============================================================

MODEL_PATH = r"C:\Users\Daisi M\Desktop\RoomAnalysisAI\models\room_damage_cnn_v6_best.keras"

IMAGE_PATH = r"C:\Users\Daisi M\Desktop\RoomAnalysisAI\dataset\test\crack"

CLASS_NAMES = [
    "crack",
    "dampness",
    "normal",
    "peeling_paint"
]

IMAGE_SIZE = (224, 224)

print("=" * 60)
print("ROOM ANALYSIS AI - CNN TEST")
print("=" * 60)

# Load model
print("\nLoading trained CNN...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# Find first image inside the crack test folder
import os

image_files = [
    f for f in os.listdir(IMAGE_PATH)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
]

if not image_files:
    print("\nERROR: No image found in:")
    print(IMAGE_PATH)
    raise SystemExit

image_path = os.path.join(
    IMAGE_PATH,
    image_files[0]
)

print("\nTesting image:")
print(image_path)

# Load image
image = Image.open(image_path).convert("RGB")
image = image.resize(IMAGE_SIZE)

# Convert to NumPy
image_array = np.array(image, dtype=np.float32)

# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

# MobileNetV2 preprocessing
image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
    image_array
)

# Prediction
print("\nRunning CNN prediction...")

predictions = model.predict(
    image_array,
    verbose=0
)[0]

predicted_index = int(np.argmax(predictions))
predicted_class = CLASS_NAMES[predicted_index]
confidence = float(predictions[predicted_index]) * 100

# Results
print("\n" + "=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

print(f"\nPredicted damage: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")

print("\nAll class probabilities:")

for index, class_name in enumerate(CLASS_NAMES):
    print(
        f"{class_name:15} : "
        f"{predictions[index] * 100:.2f}%"
    )

print("\n" + "=" * 60)
print("CNN TEST COMPLETED")
print("=" * 60)