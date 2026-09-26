import tensorflow as tf

MODEL_PATH = r"C:\Users\Daisi M\Desktop\RoomAnalysisAI\models\room_damage_cnn_v6_best.keras"

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print("Number of classes:", model.output_shape[-1])