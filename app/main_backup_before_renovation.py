
import os
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename


# ============================================================
# ROOM ANALYSIS AI
# FLASK BACKEND + CNN V6
# ============================================================

print("=" * 60)
print("ROOM ANALYSIS AI BACKEND - CNN V6")
print("=" * 60)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = r"C:\Users\Daisi M\Desktop\RoomAnalysisAI"

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "room_damage_cnn_v6_best.keras"
)

UPLOAD_FOLDER = os.path.join(
    PROJECT_DIR,
    "uploads"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ============================================================
# CNN SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "crack",
    "dampness",
    "normal",
    "peeling_paint"
]


# ============================================================
# LOAD CNN V6 MODEL
# ============================================================

print("\nLoading CNN V6 model...")

try:

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("CNN V6 model loaded successfully!")

except Exception as error:

    print("ERROR loading CNN V6 model:")
    print(error)

    model = None


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    if model is not None:

        return jsonify({
            "status": "running",
            "model": "CNN V6",
            "accuracy": "94.02%",
            "message": "Room Analysis AI Backend is ready!"
        })

    return jsonify({
        "status": "error",
        "message": "CNN V6 model is not loaded."
    }), 500


# ============================================================
# IMAGE PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if model is None:

        return jsonify({
            "error": "CNN V6 model is not loaded."
        }), 500


    # --------------------------------------------------------
    # Check uploaded image
    # --------------------------------------------------------

    if "image" not in request.files:

        return jsonify({
            "error": "No image uploaded."
        }), 400


    file = request.files["image"]


    if file.filename == "":

        return jsonify({
            "error": "No image selected."
        }), 400


    try:

        # ----------------------------------------------------
        # Save uploaded image
        # ----------------------------------------------------

        filename = secure_filename(
            file.filename
        )

        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(image_path)


        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = tf.keras.utils.load_img(
            image_path,
            target_size=IMAGE_SIZE
        )


        # ----------------------------------------------------
        # Convert image to array
        # ----------------------------------------------------

        image_array = tf.keras.utils.img_to_array(
            image
        )


        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # ----------------------------------------------------
        # CNN V6 prediction
        # ----------------------------------------------------

        predictions = model.predict(
            image_array,
            verbose=0
        )


        # ----------------------------------------------------
        # Find predicted class
        # ----------------------------------------------------

        predicted_index = int(
            np.argmax(
                predictions[0]
            )
        )


        predicted_class = CLASS_NAMES[
            predicted_index
        ]


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = float(
            predictions[0][predicted_index] * 100
        )

        confidence = round(
            confidence,
            2
        )


        # ----------------------------------------------------
        # All probabilities
        # ----------------------------------------------------

        probabilities = {}

        for i, class_name in enumerate(
            CLASS_NAMES
        ):

            probabilities[class_name] = round(
                float(
                    predictions[0][i] * 100
                ),
                2
            )


        # ----------------------------------------------------
        # Repair priority
        # ----------------------------------------------------

        if predicted_class == "normal":

            priority = (
                "Low Priority - "
                "No major visible damage detected."
            )

        elif predicted_class == "crack":

            if confidence >= 70:

                priority = (
                    "High Priority - "
                    "Crack damage should be inspected "
                    "and repaired."
                )

            else:

                priority = (
                    "Medium Priority - "
                    "Inspect the detected crack."
                )

        elif predicted_class == "dampness":

            if confidence >= 70:

                priority = (
                    "High Priority - "
                    "Dampness should be inspected "
                    "to identify the moisture source."
                )

            else:

                priority = (
                    "Medium Priority - "
                    "Inspect the affected area for moisture."
                )

        elif predicted_class == "peeling_paint":

            if confidence >= 70:

                priority = (
                    "Medium Priority - "
                    "Repair surface damage and repaint "
                    "the affected area."
                )

            else:

                priority = (
                    "Low Priority - "
                    "Monitor the affected paint area."
                )

        else:

            priority = "Low Priority"


        # ----------------------------------------------------
        # Material recommendation
        # ----------------------------------------------------

        if predicted_class == "crack":

            material = (
                "Wall Putty, Cement, Crack Filler"
            )

        elif predicted_class == "dampness":

            material = (
                "Waterproof Paint, Damp Proof Coating"
            )

        elif predicted_class == "peeling_paint":

            material = (
                "Wall Primer, Scraper, Interior Paint"
            )

        else:

            material = (
                "No immediate repair material required."
            )


        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print("\n" + "-" * 60)
        print("CNN V6 PREDICTION")
        print("-" * 60)

        print(
            "Condition:",
            predicted_class
        )

        print(
            "Confidence:",
            confidence,
            "%"
        )

        print(
            "Repair Priority:",
            priority
        )

        print(
            "Material:",
            material
        )

        print("-" * 60)


        # ----------------------------------------------------
        # Send result to frontend
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "class": predicted_class,

            "confidence": confidence,

            "probabilities": probabilities,

            "repair_priority": priority,

            "material_recommendation": material

        })


    except Exception as error:

        print("\nPrediction error:")
        print(error)

        return jsonify({
            "error": str(error)
        }), 500


# ============================================================
# RENOVATION PREFERENCE
# ============================================================

@app.route(
    "/renovation-options",
    methods=["GET"]
)
def renovation_options():

    return jsonify({

        "options": [

            {
                "id": "minimal",
                "name": "Minimal",
                "description": "Simple and clean renovation"
            },

            {
                "id": "contemporary",
                "name": "Contemporary",
                "description": "Modern and stylish renovation"
            },

            {
                "id": "luxury",
                "name": "Luxury",
                "description": "Premium and elegant renovation"
            },

            {
                "id": "budget",
                "name": "Budget-Friendly",
                "description": "Affordable renovation"
            },

            {
                "id": "eco",
                "name": "Eco-Friendly",
                "description": "Environmentally friendly renovation"
            }

        ]

    })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)

    print("ROOM ANALYSIS AI BACKEND")

    print("=" * 60)

    print(
        "CNN Model: CNN V6"
    )

    print(
        "Model Accuracy: 94.02%"
    )

    print(
        "Open on laptop: http://127.0.0.1:5000/"
    )

    print(
        "Open on phone: http://192.168.1.5:5000/"
    )

    print(
        "Health: http://192.168.1.5:5000/health"
    )

    print("=" * 60)


    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )

