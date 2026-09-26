# ============================================================
# ROOM ANALYSIS AI - FINAL BACKEND
# Flask + CNN Damage Detection
# + Priority-Based Recommendations
# + Room-Specific Recommendations
# + Budget Planner
# + Budget-Aware Renovation
# + Material Recommendations
# + Cloudflare FLUX.2 Klein 4B Renovation
# + ASYNCHRONOUS RENOVATION JOB PROCESSING
# ============================================================

import os
import io
import uuid
import json
import base64
import threading
import traceback
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from flask import (
    Flask,
    request,
    jsonify,
    send_file
)

from dotenv import load_dotenv

import tensorflow as tf
from tensorflow.keras.models import load_model


# ============================================================
# 1. LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# 2. BASIC CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "room_damage_cnn_v6_best.keras"
)

UPLOAD_FOLDER = (
    BASE_DIR
    / "uploads"
)

RENOVATION_FOLDER = (
    BASE_DIR
    / "renovation_outputs"
)

JOB_FOLDER = (
    RENOVATION_FOLDER
    / "jobs"
)

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

RENOVATION_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

JOB_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. CLOUDFLARE CONFIGURATION
# ============================================================

CLOUDFLARE_ACCOUNT_ID = os.getenv(
    "CLOUDFLARE_ACCOUNT_ID",
    "357d5a78f65add6c733702ab12f19c09"
)

CLOUDFLARE_API_TOKEN = os.getenv(
    "CLOUDFLARE_API_TOKEN"
)

CLOUDFLARE_MODEL = (
    "@cf/black-forest-labs/flux-2-klein-4b"
)


# ============================================================
# 4. FLASK APP
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = (
    15 * 1024 * 1024
)


# ============================================================
# 5. DAMAGE CLASSES
# ============================================================

CLASS_NAMES = [
    "crack",
    "dampness",
    "normal",
    "peeling_paint"
]


# ============================================================
# 6. STYLE NAMES
# ============================================================

STYLE_NAMES = {

    "minimal":
        "Minimal",

    "contemporary":
        "Contemporary",

    "luxury":
        "Luxury",

    "budget_friendly":
        "Budget-Friendly",

    "eco_friendly":
        "Eco-Friendly"
}


ALL_STYLES = [
    "minimal",
    "contemporary",
    "luxury",
    "budget_friendly",
    "eco_friendly"
]


# ============================================================
# 7. ROOM TYPES
# ============================================================

ROOM_TYPES = {

    "living_room": {

        "name":
            "Living Room",

        "styles":
            ALL_STYLES.copy()
    },

    "bedroom": {

        "name":
            "Bedroom",

        "styles":
            ALL_STYLES.copy()
    },

    "kitchen": {

        "name":
            "Kitchen",

        "styles":
            ALL_STYLES.copy()
    }
}


# ============================================================
# 8. LOAD CNN MODEL
# ============================================================

print("=" * 70)
print("ROOM ANALYSIS AI - BACKEND STARTING")
print("=" * 70)

print("Model path:")
print(MODEL_PATH)

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"CNN model not found: {MODEL_PATH}"
    )

print("Loading CNN model...")

model = load_model(
    MODEL_PATH,
    compile=False
)

print("CNN model loaded successfully.")

print(
    "Model input shape:",
    model.input_shape
)

print(
    "Model output shape:",
    model.output_shape
)


# ============================================================
# 9. IMAGE UTILITIES
# ============================================================

def load_image_from_bytes(image_bytes):

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    return image


def image_to_model_array(image):

    image = image.resize(
        (224, 224)
    )

    array = np.array(
        image,
        dtype=np.float32
    )

    array = np.expand_dims(
        array,
        axis=0
    )

    return array


def save_uploaded_image(
    image,
    filename=None
):

    if filename is None:

        filename = (
            f"{uuid.uuid4().hex}_original.jpg"
        )

    output_path = (
        UPLOAD_FOLDER
        / filename
    )

    image.convert(
        "RGB"
    ).save(
        output_path,
        format="JPEG",
        quality=95
    )

    return filename


def get_uploaded_file():

    uploaded_file = (
        request.files.get("file")
        or request.files.get("image")
    )

    return uploaded_file


def build_absolute_url(path):

    host = request.host
    scheme = request.scheme

    return (
        f"{scheme}://{host}{path}"
    )


def build_absolute_url_from_base(
    base_url,
    path
):

    base_url = (
        str(base_url)
        .rstrip("/")
    )

    if not path.startswith("/"):

        path = "/" + path

    return (
        f"{base_url}{path}"
    )


# ============================================================
# 10. CNN DAMAGE PREDICTION
# ============================================================

def predict_damage(image):

    model_input = image_to_model_array(
        image
    )

    raw_predictions = model.predict(
        model_input,
        verbose=0
    )[0]

    raw_predictions = np.asarray(
        raw_predictions,
        dtype=np.float64
    )

    print()
    print("=" * 70)
    print("CNN PREDICTION")
    print("=" * 70)

    print(
        "Raw model output:",
        raw_predictions
    )

    if len(raw_predictions) != len(
        CLASS_NAMES
    ):

        raise RuntimeError(
            "CNN output does not contain exactly "
            "four class predictions."
        )

    if (
        np.all(raw_predictions >= 0)
        and
        np.all(raw_predictions <= 1)
        and
        np.isclose(
            np.sum(raw_predictions),
            1.0,
            atol=0.05
        )
    ):

        probabilities_array = (
            raw_predictions
        )

        print(
            "CNN output interpreted as probabilities."
        )

    elif (
        np.all(raw_predictions >= 0)
        and
        np.all(raw_predictions <= 1)
    ):

        total = float(
            np.sum(raw_predictions)
        )

        if total <= 0:

            probabilities_array = (
                np.ones(
                    len(CLASS_NAMES)
                )
                /
                len(CLASS_NAMES)
            )

        else:

            probabilities_array = (
                raw_predictions
                /
                total
            )

        print(
            "CNN output normalized as probability scores."
        )

    else:

        shifted = (
            raw_predictions
            -
            np.max(raw_predictions)
        )

        exp_values = np.exp(
            shifted
        )

        probabilities_array = (
            exp_values
            /
            np.sum(exp_values)
        )

        print(
            "CNN output interpreted as logits "
            "and converted using softmax."
        )

    probabilities_array = np.asarray(
        probabilities_array,
        dtype=np.float64
    )

    probabilities_array = np.clip(
        probabilities_array,
        0.0,
        1.0
    )

    total = float(
        np.sum(probabilities_array)
    )

    if total <= 0:

        probabilities_array = (
            np.ones(
                len(CLASS_NAMES)
            )
            /
            len(CLASS_NAMES)
        )

    else:

        probabilities_array = (
            probabilities_array
            /
            total
        )

    highest_index = int(
        np.argmax(
            probabilities_array
        )
    )

    condition = (
        CLASS_NAMES[
            highest_index
        ]
    )

    confidence = (
        float(
            probabilities_array[
                highest_index
            ]
        )
        *
        100.0
    )

    probabilities = {

        CLASS_NAMES[i]:
            round(
                float(
                    probabilities_array[i]
                )
                *
                100.0,
                2
            )

        for i in range(
            len(CLASS_NAMES)
        )
    }

    rounded_total = round(
        sum(
            probabilities.values()
        ),
        2
    )

    difference = round(
        100.0 - rounded_total,
        2
    )

    if difference != 0:

        corrected_value = round(
            probabilities[condition]
            +
            difference,
            2
        )

        probabilities[
            condition
        ] = max(
            0.0,
            corrected_value
        )

    final_total = round(
        sum(
            probabilities.values()
        ),
        2
    )

    print(
        "Condition:",
        condition
    )

    print(
        "Confidence:",
        round(
            confidence,
            2
        ),
        "%"
    )

    print(
        "Probabilities:",
        probabilities
    )

    print(
        "Probability total:",
        final_total,
        "%"
    )

    return (
        condition,
        round(
            confidence,
            2
        ),
        probabilities
    )


# ============================================================
# 11. CONDITION DISPLAY NAME
# ============================================================

def format_condition(condition):

    names = {

        "crack":
            "Crack",

        "dampness":
            "Dampness",

        "normal":
            "Normal",

        "peeling_paint":
            "Peeling Paint"
    }

    return names.get(
        condition,
        str(condition)
        .replace(
            "_",
            " "
        )
        .title()
    )


# ============================================================
# 12. PRIORITY-BASED RECOMMENDATION
# ============================================================

def get_priority(
    condition,
    confidence
):

    if condition in [
        "dampness",
        "crack"
    ]:

        return "High"

    if condition == "peeling_paint":

        if confidence >= 80:

            return "High"

        return "Medium"

    if condition == "normal":

        return "Low"

    return "Medium"


def get_priority_recommendation(
    condition,
    confidence
):

    priority = get_priority(
        condition,
        confidence
    )

    recommendations = {

        "crack": [

            "Repair the detected wall cracks first.",

            "Clean and prepare the affected surface.",

            "Use a suitable crack repair or filling system.",

            "Apply wall putty where required.",

            "Apply primer and repaint the repaired surface.",

            "Continue with aesthetic renovation after crack repair."
        ],

        "dampness": [

            "Treat the detected moisture or dampness first.",

            "Investigate and address the moisture source.",

            "Repair the affected wall surface.",

            "Use a suitable damp-proofing or waterproofing system.",

            "Apply suitable primer and moisture-resistant finish.",

            "Continue with aesthetic renovation after the affected area is treated."
        ],

        "peeling_paint": [

            "Remove loose and peeling paint first.",

            "Clean and prepare the affected wall surface.",

            "Repair uneven areas with suitable wall putty.",

            "Apply suitable primer.",

            "Repaint using an appropriate interior coating.",

            "Continue with aesthetic renovation after surface preparation."
        ],

        "normal": [

            "The detected wall condition is Normal.",

            "No major visible damage is indicated by the CNN.",

            "Major repair work is not the first priority.",

            "Proceed with suitable aesthetic improvements.",

            "Select the room type, style and budget.",

            "Use the budget planner to prioritize practical improvements."
        ]
    }

    actions = recommendations.get(
        condition,
        recommendations["normal"]
    )

    return {

        "priority":
            priority,

        "detected_condition":
            format_condition(
                condition
            ),

        "confidence":
            round(
                confidence,
                2
            ),

        "recommendation":
            actions,

        "priority_message":
            (
                f"{format_condition(condition)} "
                f"has the highest CNN confidence of "
                f"{confidence:.2f}%. "
                f"Priority: {priority}."
            )
    }


# ============================================================
# 13. ROOM TYPE NORMALIZATION
# ============================================================

def normalize_room_type(room_type):

    if not room_type:

        return "living_room"

    value = (
        str(room_type)
        .strip()
        .lower()
    )

    aliases = {

        "living":
            "living_room",

        "livingroom":
            "living_room",

        "living room":
            "living_room",

        "hall":
            "living_room",

        "bed":
            "bedroom",

        "bed room":
            "bedroom",

        "bedroom":
            "bedroom",

        "kitchen":
            "kitchen",

        "cooking":
            "kitchen"
    }

    value = aliases.get(
        value,
        value
    )

    if value not in ROOM_TYPES:

        return "living_room"

    return value


# ============================================================
# 14. BUDGET LEVEL
# ============================================================

def get_budget_level(budget):

    try:

        budget = float(budget)

    except Exception:

        return "Basic"

    if budget < 25000:

        return "Basic"

    if budget <= 75000:

        return "Standard"

    return "Premium"


# ============================================================
# 15. RENOVATION SCOPE
# ============================================================

def get_renovation_scope(
    budget_level
):

    scopes = {

        "Basic":
            "Essential and limited renovation. "
            "Prioritize detected damage, surface repair, "
            "simple painting, basic lighting and small "
            "affordable improvements.",

        "Standard":
            "Moderate practical renovation. "
            "Include essential repairs, improved finishes, "
            "lighting and selected furniture or decor "
            "changes while avoiding unnecessary structural work.",

        "Premium":
            "More extensive coordinated renovation. "
            "Allow higher-quality finishes, furniture, "
            "lighting, storage and decorative improvements "
            "while preserving the original room structure."
    }

    return scopes.get(
        budget_level,
        scopes["Basic"]
    )


# ============================================================
# 16. MATERIAL RECOMMENDATIONS
# ============================================================

def get_material_recommendations(
    condition,
    room_type,
    budget_level="Standard"
):

    room_type = normalize_room_type(
        room_type
    )

    if room_type == "living_room":

        recommendations = {

            "crack": [

                "Compatible crack repair / filling system",
                "White-cement-based polymeric wall putty",
                "Suitable interior primer",
                "Interior decorative wall coating",
                "Basic lighting materials"
            ],

            "dampness": [

                "Suitable damp-proofing / waterproofing system",
                "Compatible surface repair system",
                "White-cement-based polymeric wall putty",
                "Suitable interior primer",
                "Moisture-resistant interior decorative coating"
            ],

            "peeling_paint": [

                "Surface preparation tools",
                "White-cement-based polymeric wall putty",
                "Suitable interior primer",
                "Interior decorative wall coating",
                "Basic lighting materials"
            ],

            "normal": [

                "Surface preparation materials",
                "White-cement-based polymeric wall putty where required",
                "Suitable interior primer",
                "Interior decorative wall coating",
                "Simple lighting materials"
            ]
        }

    elif room_type == "bedroom":

        recommendations = {

            "crack": [

                "Compatible crack repair / filling system",
                "White-cement-based polymeric wall putty",
                "Suitable interior primer",
                "Interior decorative wall coating",
                "Simple bedroom lighting materials"
            ],

            "dampness": [

                "Suitable damp-proofing / waterproofing system",
                "Compatible surface repair system",
                "White-cement-based polymeric wall putty",
                "Suitable interior primer",
                "Moisture-resistant interior coating"
            ],

            "peeling_paint": [

                "Surface preparation materials",
                "White-cement-based polymeric wall putty",
                "Suitable interior primer",
                "Interior decorative wall coating",
                "Simple bedroom lighting materials"
            ],

            "normal": [

                "Surface preparation materials",
                "Wall putty where required",
                "Suitable interior primer",
                "Interior decorative wall coating",
                "Simple bedroom lighting materials"
            ]
        }

    else:

        recommendations = {

            "crack": [

                "Compatible crack repair / filling system",
                "Suitable kitchen wall or tile repair system",
                "Moisture-resistant primer",
                "Washable moisture-resistant wall finish",
                "Basic cabinet or hardware improvement materials"
            ],

            "dampness": [

                "Suitable damp-proofing / waterproofing system",
                "Moisture-resistant surface repair system",
                "Suitable kitchen primer",
                "Washable moisture-resistant wall finish",
                "Moisture-resistant storage or cabinet materials"
            ],

            "peeling_paint": [

                "Surface preparation materials",
                "Suitable kitchen wall repair system",
                "Moisture-resistant primer",
                "Washable moisture-resistant coating",
                "Small backsplash or cabinet improvement materials"
            ],

            "normal": [

                "Kitchen surface preparation materials",
                "Suitable kitchen primer",
                "Washable moisture-resistant wall finish",
                "Cabinet hardware improvement materials",
                "Storage or backsplash improvement materials"
            ]
        }

    return recommendations.get(
        condition,
        recommendations["normal"]
    )


def get_budget_material_recommendations(
    condition,
    room_type,
    budget
):

    budget_level = get_budget_level(
        budget
    )

    base = get_material_recommendations(
        condition,
        room_type,
        budget_level
    )

    if budget_level == "Basic":

        guidance = [

            "Prioritize essential damage repair first.",
            "Use economical but suitable standard materials.",
            "Prefer surface repair and repainting over major remodeling.",
            "Reuse existing furniture wherever possible.",
            "Avoid premium finishes and unnecessary decorative items."
        ]

    elif budget_level == "Standard":

        guidance = [

            "Use standard-quality durable materials.",
            "Prioritize damage repair before aesthetic improvements.",
            "Upgrade selected furniture elements where appropriate.",
            "Use coordinated wall, lighting and decor finishes.",
            "Avoid unnecessary structural changes."
        ]

    else:

        guidance = [

            "Use higher-quality durable materials.",
            "Allow coordinated furniture and lighting upgrades.",
            "Use higher-quality decorative finishes.",
            "Include appropriate storage and room-specific improvements.",
            "Allow more extensive visual renovation while preserving structure."
        ]

    return {

        "budget_level":
            budget_level,

        "recommended_materials":
            base,

        "material_guidance":
            guidance
    }


# ============================================================
# 17. BUDGET PLAN
# ============================================================

def create_budget_plan(
    room_type,
    condition,
    budget,
    furniture_enabled=True
):

    try:

        budget = float(budget)

    except Exception:

        budget = 0.0

    budget = max(
        0.0,
        budget
    )

    budget_level = get_budget_level(
        budget
    )

    room_type = normalize_room_type(
        room_type
    )

    damaged_conditions = [
        "crack",
        "dampness",
        "peeling_paint"
    ]

    if not furniture_enabled:

        if room_type in [
            "living_room",
            "bedroom"
        ]:

            if condition in damaged_conditions:

                items = [

                    (
                        "Damage repair and surface preparation",
                        0.45
                    ),

                    (
                        "Wall putty, primer and paint",
                        0.30
                    ),

                    (
                        "Basic lighting",
                        0.10
                    ),

                    (
                        "Curtains / existing soft furnishing improvement",
                        0.05
                    ),

                    (
                        "Miscellaneous",
                        0.10
                    )
                ]

            else:

                items = [

                    (
                        "Wall preparation and paint",
                        0.40
                    ),

                    (
                        "Basic lighting",
                        0.15
                    ),

                    (
                        "Curtains / soft furnishing improvement",
                        0.15
                    ),

                    (
                        "Surface finishing",
                        0.20
                    ),

                    (
                        "Miscellaneous",
                        0.10
                    )
                ]

        else:

            items = [

                (
                    "Wall / surface repair",
                    0.40
                ),

                (
                    "Primer and washable finish",
                    0.25
                ),

                (
                    "Basic kitchen lighting",
                    0.10
                ),

                (
                    "Existing cabinet / hardware improvement",
                    0.15
                ),

                (
                    "Miscellaneous",
                    0.10
                )
            ]

    elif budget_level == "Basic":

        if room_type in [
            "living_room",
            "bedroom"
        ]:

            if condition in damaged_conditions:

                items = [

                    (
                        "Essential damage repair",
                        0.40
                    ),

                    (
                        "Wall putty, primer and paint",
                        0.30
                    ),

                    (
                        "Basic lighting",
                        0.10
                    ),

                    (
                        "Minimal low-cost furniture",
                        0.10
                    ),

                    (
                        "Miscellaneous",
                        0.10
                    )
                ]

            else:

                items = [

                    (
                        "Wall preparation and paint",
                        0.35
                    ),

                    (
                        "Basic lighting",
                        0.15
                    ),

                    (
                        "Minimal low-cost furniture",
                        0.15
                    ),

                    (
                        "Simple curtains / decor",
                        0.15
                    ),

                    (
                        "Miscellaneous",
                        0.20
                    )
                ]

        else:

            items = [

                (
                    "Essential wall / surface repair",
                    0.35
                ),

                (
                    "Washable wall finish",
                    0.25
                ),

                (
                    "Basic lighting",
                    0.10
                ),

                (
                    "Minimal cabinet / storage improvement",
                    0.15
                ),

                (
                    "Miscellaneous",
                    0.15
                )
            ]

    elif budget_level == "Standard":

        if room_type == "living_room":

            items = [

                (
                    "Surface repair and paint",
                    0.25
                ),

                (
                    "Furniture upgrade",
                    0.25
                ),

                (
                    "Lighting",
                    0.15
                ),

                (
                    "Curtains / rug / decor",
                    0.20
                ),

                (
                    "Miscellaneous",
                    0.15
                )
            ]

        elif room_type == "bedroom":

            items = [

                (
                    "Surface repair and paint",
                    0.25
                ),

                (
                    "Bed / headboard improvement",
                    0.25
                ),

                (
                    "Storage / furniture improvement",
                    0.15
                ),

                (
                    "Lighting",
                    0.10
                ),

                (
                    "Curtains / rug / decor",
                    0.15
                ),

                (
                    "Miscellaneous",
                    0.10
                )
            ]

        else:

            items = [

                (
                    "Wall / surface repair",
                    0.25
                ),

                (
                    "Cabinet improvement",
                    0.25
                ),

                (
                    "Countertop / backsplash",
                    0.20
                ),

                (
                    "Lighting",
                    0.10
                ),

                (
                    "Storage / hardware",
                    0.10
                ),

                (
                    "Miscellaneous",
                    0.10
                )
            ]

    else:

        if room_type == "living_room":

            items = [

                (
                    "Surface repair and premium finish",
                    0.20
                ),

                (
                    "Furniture upgrade",
                    0.30
                ),

                (
                    "Lighting",
                    0.15
                ),

                (
                    "Curtains / rug / decor",
                    0.20
                ),

                (
                    "Storage / feature elements",
                    0.10
                ),

                (
                    "Miscellaneous",
                    0.05
                )
            ]

        elif room_type == "bedroom":

            items = [

                (
                    "Surface repair and premium finish",
                    0.20
                ),

                (
                    "Bed / headboard",
                    0.25
                ),

                (
                    "Wardrobe / storage",
                    0.20
                ),

                (
                    "Lighting",
                    0.10
                ),

                (
                    "Curtains / rug / decor",
                    0.20
                ),

                (
                    "Miscellaneous",
                    0.05
                )
            ]

        else:

            items = [

                (
                    "Wall / surface repair",
                    0.15
                ),

                (
                    "Cabinet upgrade",
                    0.30
                ),

                (
                    "Countertop / backsplash",
                    0.20
                ),

                (
                    "Lighting",
                    0.10
                ),

                (
                    "Storage / hardware",
                    0.15
                ),

                (
                    "Sink / fittings",
                    0.05
                ),

                (
                    "Miscellaneous",
                    0.05
                )
            ]

    result = []

    for name, percentage in items:

        amount = round(
            budget * percentage
        )

        result.append({

            "category":
                name,

            "percentage":
                round(
                    percentage * 100,
                    1
                ),

            "estimated_amount":
                amount
        })

    calculated_total = sum(

        item[
            "estimated_amount"
        ]

        for item in result
    )

    difference = round(
        budget
        -
        calculated_total
    )

    if result and difference != 0:

        result[-1][
            "estimated_amount"
        ] += difference

    return {

        "room_type":
            room_type,

        "room_name":
            ROOM_TYPES[
                room_type
            ]["name"],

        "condition":
            format_condition(
                condition
            ),

        "budget":
            round(
                budget,
                2
            ),

        "budget_level":
            budget_level,

        "furniture_enabled":
            furniture_enabled,

        "renovation_scope":
            get_renovation_scope(
                budget_level
            ),

        "items":
            result,

        "total_planned_amount":
            round(
                budget,
                2
            )
    }


# ============================================================
# 18. BUILD FLUX PROMPT
# ============================================================

def build_room_style_prompt(
    room_type,
    style,
    condition,
    budget,
    furniture_enabled=True
):

    room_type = normalize_room_type(
        room_type
    )

    style = (
        str(style)
        .strip()
        .lower()
    )

    if style not in STYLE_NAMES:

        style = "minimal"

    room_name = ROOM_TYPES[
        room_type
    ]["name"]

    style_name = STYLE_NAMES[
        style
    ]

    budget_level = get_budget_level(
        budget
    )

    renovation_scope = get_renovation_scope(
        budget_level
    )

    if condition == "crack":

        repair_text = (
            "Repair the visible wall cracks realistically. "
            "Restore only the affected surfaces before applying "
            "the new wall finish."
        )

    elif condition == "dampness":

        repair_text = (
            "Visually repair the visible dampness and moisture "
            "stains. Restore the affected wall surface and apply "
            "a suitable moisture-resistant finish."
        )

    elif condition == "peeling_paint":

        repair_text = (
            "Remove the visible peeling and loose paint. "
            "Prepare the affected surface and restore it with "
            "a clean, realistic wall finish."
        )

    else:

        repair_text = (
            "The detected condition is NORMAL. Do not invent "
            "damage or repair work. Keep the existing walls "
            "structurally unchanged and make only suitable "
            "aesthetic improvements."
        )

    if style == "minimal":

        style_text = (
            "Minimal interior design with clean simple forms, "
            "light neutral tones, uncluttered surfaces, practical "
            "finishes and restrained decoration."
        )

    elif style == "contemporary":

        style_text = (
            "Contemporary interior with clean lines, balanced "
            "neutral colors, modern but practical finishes and "
            "limited tasteful decoration."
        )

    elif style == "luxury":

        style_text = (
            "Elegant luxury interior with refined finishes, "
            "coordinated lighting, sophisticated textures and "
            "premium-looking details."
        )

    elif style == "budget_friendly":

        style_text = (
            "Affordable practical interior using economical "
            "standard finishes, simple improvements and limited "
            "decoration."
        )

    else:

        style_text = (
            "Eco-friendly interior using natural-looking materials, "
            "energy-efficient lighting, sustainable-looking finishes "
            "and calm natural tones."
        )

    if budget_level == "Basic":

        budget_text = (
            f"BASIC renovation with a strict total planning "
            f"budget of INR {budget:.0f}. "
            "This is a LOW-BUDGET renovation. "
            "Make only small, realistic and affordable improvements. "
            "Prioritize necessary surface repair and repainting. "
            "Use economical standard materials. "
            "Do not perform structural remodeling. "
            "Do not create an expensive designer interior. "
            "Do not add premium furniture. "
            "Do not add large amounts of new furniture. "
            "Keep most existing room elements unchanged."
        )

    elif budget_level == "Standard":

        budget_text = (
            f"STANDARD renovation with a planning budget of "
            f"INR {budget:.0f}. "
            "Make moderate practical improvements. "
            "Use standard-quality finishes and limited coordinated "
            "furniture or decoration. Preserve the original layout."
        )

    else:

        budget_text = (
            f"PREMIUM renovation with a planning budget of "
            f"INR {budget:.0f}. "
            "Allow higher-quality coordinated finishes, furniture, "
            "lighting and decoration while preserving the original "
            "room structure."
        )

    if not furniture_enabled:

        furniture_text = (
            "FURNITURE OFF. "
            "Do NOT add any new furniture or decor. "
            "Do not add a sofa, chair, table, bed, coffee table, "
            "TV unit, wardrobe, cabinet, shelf or other furniture. "
            "Keep existing furniture unchanged where visible. "
            "Focus only on wall treatment, surface finishing and "
            "permitted non-furniture improvements."
        )

    elif budget_level == "Basic":

        furniture_text = (
            "FURNITURE ON, BUT BASIC BUDGET. "
            "Furniture must remain extremely limited and inexpensive. "
            "Prefer reusing the existing furniture. "
            "At most one small practical low-cost furniture improvement "
            "may be introduced if clearly suitable for the room. "
            "Do not create a complete furniture makeover."
        )

    elif budget_level == "Standard":

        furniture_text = (
            "FURNITURE ON. "
            "Add only a limited number of practical coordinated "
            "furniture or decor elements suitable for the room "
            "and selected style."
        )

    else:

        furniture_text = (
            "FURNITURE ON. "
            "Allow coordinated furniture and decor appropriate "
            "for the room, style and premium budget."
        )

    if budget_level == "Basic":

        room_changes = (
            f"For the {room_name}, make only minimal renovation "
            "changes. Keep the existing architecture, layout, "
            "windows, doors, flooring and major elements. "
            "Focus on repair, repainting, simple lighting and "
            "very limited affordable styling."
        )

    elif budget_level == "Standard":

        room_changes = (
            f"For the {room_name}, make moderate coordinated "
            "improvements to walls, lighting and selected "
            "furniture or decor while preserving the original layout."
        )

    else:

        room_changes = (
            f"For the {room_name}, allow a more complete coordinated "
            "renovation using suitable furniture, lighting, finishes "
            "and decor while preserving the original architecture."
        )

    prompt = f"""
Photorealistic renovation of the SAME {room_name} shown in the
input photograph.

ROOM TYPE:
{room_name}

SELECTED STYLE:
{style_name}

TOTAL PLANNING BUDGET:
INR {budget:.0f}

RENOVATION LEVEL:
{budget_level}

DETECTED CONDITION:
{format_condition(condition)}

RENOVATION SCOPE:
{renovation_scope}

DAMAGE REPAIR:
{repair_text}

STYLE DESCRIPTION:
{style_text}

BUDGET RULE:
{budget_text}

ROOM CHANGES:
{room_changes}

FURNITURE RULE:
{furniture_text}

IMPORTANT IMAGE REQUIREMENTS:

Create a realistic photograph of the SAME ROOM AFTER RENOVATION.

Preserve the exact original:
walls,
doors,
windows,
ceiling shape,
floor geometry,
room proportions,
architectural layout,
camera viewpoint,
camera angle,
camera perspective.

Do not create a different room.

Do not move doors.
Do not move windows.
Do not add windows.
Do not remove windows.
Do not change room dimensions.
Do not change the camera angle.
Do not change the perspective.
Do not perform structural remodeling.

The renovation must look physically realistic and practically achievable.

The selected budget is a STRICT renovation constraint.

For BASIC renovation, especially for a budget such as INR 20000,
make the result visibly modest, affordable and realistic.

A BASIC INR 20000 renovation must NOT look like a luxury
interior, premium designer showroom or expensive complete remodel.

Use simple economical wall finishing, repainting, basic lighting
and very limited affordable improvements.

Prefer retaining existing kitchen cabinets, counters, appliances,
flooring and major room elements when possible.

Do not replace expensive kitchen cabinets or appliances for a
low budget.

Do not introduce expensive marble, premium stone, luxury
woodwork, elaborate false ceilings, large designer furniture
or extensive remodeling for a BASIC budget.

The budget controls the SCALE and EXTENT of renovation.

The style controls the visual character, but the budget remains
the strict financial constraint.

The detected condition must be respected.

If the condition is NORMAL, do not invent cracks, dampness,
peeling paint or other damage.

If the condition is a damage class, repair that visible condition
before applying the new aesthetic finish.

FURNITURE AND DECOR MUST FOLLOW THE FURNITURE SETTING.

If furniture is OFF, do not add any new furniture or decorative
furniture items.

If furniture is ON with a BASIC budget, use furniture extremely
sparingly and prefer existing furniture.

============================================================
STRICT NO-TEXT / NO-WATERMARK REQUIREMENT
============================================================

The generated image must contain NO readable or decorative text.

Do NOT generate:
text,
words,
letters,
numbers,
prices,
currency symbols,
captions,
labels,
logos,
brand names,
signs,
posters with text,
advertisements,
watermarks,
signatures,
typography,
written measurements,
UI elements,
menus,
interface elements,
product labels.

Do not display the budget inside the image.

Do not display INR.
Do not display ₹.
Do not display 20000.
Do not display the room name.
Do not display the style name.

Do not add a watermark anywhere in the image.

Avoid signs, posters, packages, labels or objects containing
visible writing.

The final result must be a clean interior photograph without
textual overlays.

============================================================
REALISM REQUIREMENT
============================================================

Photorealistic interior photography.

Realistic wall materials.
Realistic paint.
Realistic kitchen surfaces.
Realistic furniture proportions.
Realistic lighting.
Natural shadows.
Natural reflections.
Physically plausible materials.
Consistent perspective.
Consistent room geometry.

The output must look like a genuine photograph taken after
a practical renovation.

It must NOT look like a poster,
advertisement,
catalogue image,
concept board,
3D showroom,
fantasy interior,
or luxury architectural visualization.

The final image should show the original room improved
realistically according to the selected condition, room,
style and strict budget.
"""

    return " ".join(
        prompt.split()
    )


# ============================================================
# 19. CLOUDFLARE FLUX GENERATION
# ============================================================

def prepare_cloudflare_reference_image(
    image
):

    image = image.convert(
        "RGB"
    )

    max_dimension = 511

    width, height = image.size

    scale = min(
        max_dimension / width,
        max_dimension / height,
        1.0
    )

    new_width = max(
        1,
        int(round(width * scale))
    )

    new_height = max(
        1,
        int(round(height * scale))
    )

    if (
        new_width != width
        or
        new_height != height
    ):

        image = image.resize(
            (
                new_width,
                new_height
            ),
            Image.Resampling.LANCZOS
        )

    return image


def generate_with_cloudflare(
    image,
    prompt,
    output_path
):

    if not CLOUDFLARE_API_TOKEN:

        raise RuntimeError(
            "CLOUDFLARE_API_TOKEN is missing from .env."
        )

    endpoint = (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{CLOUDFLARE_ACCOUNT_ID}"
        f"/ai/run/"
        f"{CLOUDFLARE_MODEL}"
    )

    reference_image = (
        prepare_cloudflare_reference_image(
            image
        )
    )

    image_buffer = io.BytesIO()

    reference_image.save(
        image_buffer,
        format="JPEG",
        quality=90
    )

    image_buffer.seek(0)

    files = {

        "input_image_0": (

            "room.jpg",

            image_buffer,

            "image/jpeg"
        )
    }

    data = {

        "prompt":
            prompt,

        "width":
            "512",

        "height":
            "512"
    }

    headers = {

        "Authorization":
            f"Bearer {CLOUDFLARE_API_TOKEN}"
    }

    print()
    print("=" * 70)
    print("CLOUDFLARE RENOVATION GENERATION")
    print("=" * 70)

    print(
        "Model:",
        CLOUDFLARE_MODEL
    )

    print(
        "Reference image size:",
        reference_image.size
    )

    print(
        "Sending image to Cloudflare..."
    )

    try:

        response = requests.post(

            endpoint,

            headers=headers,

            files=files,

            data=data,

            timeout=300
        )

    except requests.RequestException as exc:

        raise RuntimeError(
            f"Unable to connect to Cloudflare: {exc}"
        ) from exc

    print(
        "Cloudflare HTTP status:",
        response.status_code
    )

    if response.status_code != 200:

        print(
            "Cloudflare response:",
            response.text[:5000]
        )

        raise RuntimeError(
            "Cloudflare image generation failed. "
            f"HTTP {response.status_code}: "
            f"{response.text[:1000]}"
        )

    try:

        result = response.json()

    except Exception as exc:

        raise RuntimeError(
            "Cloudflare returned an invalid JSON response."
        ) from exc

    if not result.get(
        "success",
        False
    ):

        print(
            "Cloudflare error response:",
            result
        )

        errors = result.get(
            "errors",
            []
        )

        raise RuntimeError(
            "Cloudflare returned an unsuccessful response: "
            f"{errors}"
        )

    result_data = result.get(
        "result"
    )

    if not result_data:

        raise RuntimeError(
            "Cloudflare response does not contain result."
        )

    image_base64 = result_data.get(
        "image"
    )

    if not image_base64:

        raise RuntimeError(
            "Cloudflare response does not contain image data."
        )

    try:

        generated_bytes = base64.b64decode(
            image_base64,
            validate=True
        )

    except Exception as exc:

        raise RuntimeError(
            "Unable to decode Cloudflare image."
        ) from exc

    if not generated_bytes:

        raise RuntimeError(
            "Cloudflare returned empty image data."
        )

    try:

        generated_image = Image.open(
            io.BytesIO(
                generated_bytes
            )
        ).convert("RGB")

    except Exception as exc:

        raise RuntimeError(
            "Cloudflare returned invalid image data."
        ) from exc

    generated_image.save(
        output_path,
        format="PNG"
    )

    if (
        not output_path.exists()
        or
        output_path.stat().st_size <= 0
    ):

        raise RuntimeError(
            "Generated renovation image was not saved correctly."
        )

    print(
        "Renovation image saved:",
        output_path
    )

    print(
        "Generated image size:",
        generated_image.size
    )

    print(
        "Generated image file size:",
        output_path.stat().st_size,
        "bytes"
    )

    return output_path


# ============================================================
# 20. JOB FILE UTILITIES
# ============================================================

def get_job_path(job_id):

    return (
        JOB_FOLDER
        /
        f"{job_id}.json"
    )


def write_job_status(
    job_id,
    data
):

    job_path = get_job_path(
        job_id
    )

    temporary_path = (
        JOB_FOLDER
        /
        f"{job_id}.tmp"
    )

    with open(
        temporary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )

    os.replace(
        temporary_path,
        job_path
    )


def read_job_status(job_id):

    job_path = get_job_path(
        job_id
    )

    if not job_path.exists():

        return None

    try:

        with open(
            job_path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:

        return None


# ============================================================
# 21. BACKGROUND RENOVATION PROCESS
# ============================================================

def process_renovation_job(
    job_id,
    input_path,
    room_type,
    style,
    budget,
    furniture_enabled,
    base_url
):

    try:

        print()
        print("=" * 70)
        print(
            f"STARTING BACKGROUND RENOVATION JOB: {job_id}"
        )
        print("=" * 70)

        write_job_status(
            job_id,
            {
                "success":
                    True,

                "job_id":
                    job_id,

                "status":
                    "processing",

                "message":
                    "Room analysis and renovation generation are in progress."
            }
        )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        with open(
            input_path,
            "rb"
        ) as file:

            image_bytes = file.read()

        image = load_image_from_bytes(
            image_bytes
        )

        # ----------------------------------------------------
        # Budget
        # ----------------------------------------------------

        budget_level = get_budget_level(
            budget
        )

        renovation_scope = (
            get_renovation_scope(
                budget_level
            )
        )

        # ----------------------------------------------------
        # CNN
        # ----------------------------------------------------

        (
            condition,
            confidence,
            probabilities
        ) = predict_damage(
            image
        )

        print(
            "Background job condition:",
            condition
        )

        # ----------------------------------------------------
        # Priority
        # ----------------------------------------------------

        priority_result = (
            get_priority_recommendation(
                condition,
                confidence
            )
        )

        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        prompt = build_room_style_prompt(

            room_type=
                room_type,

            style=
                style,

            condition=
                condition,

            budget=
                budget,

            furniture_enabled=
                furniture_enabled
        )

        # ----------------------------------------------------
        # Output filename
        # ----------------------------------------------------

        filename = (

            f"{job_id}_"
            f"{room_type}_"
            f"{style}_"
            f"{budget_level.lower()}.png"
        )

        output_path = (

            RENOVATION_FOLDER
            /
            filename
        )

        # ----------------------------------------------------
        # Cloudflare generation
        # ----------------------------------------------------

        generate_with_cloudflare(

            image=
                image,

            prompt=
                prompt,

            output_path=
                output_path
        )

        # ----------------------------------------------------
        # Materials
        # ----------------------------------------------------

        recommendations = (

            get_budget_material_recommendations(

                condition,

                room_type,

                budget
            )
        )

        # ----------------------------------------------------
        # Budget plan
        # ----------------------------------------------------

        budget_plan_result = (

            create_budget_plan(

                room_type,

                condition,

                budget,

                furniture_enabled
            )
        )

        # ----------------------------------------------------
        # Planning values
        # ----------------------------------------------------

        optimistic_estimate = round(
            budget * 0.90,
            2
        )

        realistic_estimate = round(
            budget,
            2
        )

        # ----------------------------------------------------
        # Probability list
        # ----------------------------------------------------

        probabilities_list = [

            {
                "condition":
                    CLASS_NAMES[i],

                "condition_display":
                    format_condition(
                        CLASS_NAMES[i]
                    ),

                "percentage":
                    probabilities[
                        CLASS_NAMES[i]
                    ]
            }

            for i in range(
                len(CLASS_NAMES)
            )
        ]

        probability_total = round(
            sum(
                probabilities.values()
            ),
            2
        )

        # ----------------------------------------------------
        # Renovated image URL
        # ----------------------------------------------------

        image_path = (
            f"/renovation/image/{filename}"
        )

        image_url = (
            build_absolute_url_from_base(
                base_url,
                image_path
            )
        )

        image_url_with_cache_bust = (
            f"{image_url}?v={job_id}"
        )

        # ----------------------------------------------------
        # Original image
        # ----------------------------------------------------

        original_filename = (
            save_uploaded_image(
                image
            )
        )

        original_path = (
            f"/uploads/{original_filename}"
        )

        original_url = (
            build_absolute_url_from_base(
                base_url,
                original_path
            )
        )

        # ----------------------------------------------------
        # Completed result
        # ----------------------------------------------------

        final_result = {

            "success":
                True,

            "job_id":
                job_id,

            "status":
                "completed",

            "message":
                "Renovation generated successfully.",

            "room_type":
                room_type,

            "room_name":
                ROOM_TYPES[
                    room_type
                ]["name"],

            "style":
                style,

            "style_name":
                STYLE_NAMES[
                    style
                ],

            "condition":
                condition,

            "condition_display":
                format_condition(
                    condition
                ),

            "confidence":
                confidence,

            "highest_confidence":
                confidence,

            "probabilities":
                probabilities,

            "probabilities_list":
                probabilities_list,

            "probability_total":
                probability_total,

            "priority":
                priority_result[
                    "priority"
                ],

            "priority_recommendation":
                priority_result[
                    "recommendation"
                ],

            "priority_message":
                priority_result[
                    "priority_message"
                ],

            "budget":
                budget,

            "budget_level":
                budget_level,

            "renovation_scope":
                renovation_scope,

            "optimistic_estimate":
                optimistic_estimate,

            "realistic_estimate":
                realistic_estimate,

            "furniture_enabled":
                furniture_enabled,

            "material_recommendations":
                recommendations,

            "budget_plan":
                budget_plan_result,

            "original_image_url":
                original_url,

            "renovated_image_url":
                image_url_with_cache_bust,

            "renovation_image_url":
                image_url_with_cache_bust,

            "image_url":
                image_url_with_cache_bust,

            "image_path":
                image_path,

            "filename":
                filename
        }

        write_job_status(
            job_id,
            final_result
        )

        print()
        print("=" * 70)
        print(
            f"RENOVATION JOB COMPLETED: {job_id}"
        )
        print("=" * 70)

    except Exception as exc:

        traceback.print_exc()

        error_result = {

            "success":
                False,

            "job_id":
                job_id,

            "status":
                "failed",

            "message":
                "Renovation generation failed.",

            "error":
                str(exc)
        }

        try:

            write_job_status(
                job_id,
                error_result
            )

        except Exception:

            traceback.print_exc()


# ============================================================
# 22. HOME
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return jsonify({

        "project":
            "Room Analysis AI",

        "status":
            "Backend running",

        "model":
            "RoomDamageCNN_V6",

        "renovation_model":
            CLOUDFLARE_MODEL,

        "room_types": [
            "living_room",
            "bedroom",
            "kitchen"
        ],

        "styles":
            ALL_STYLES,

        "budget_levels": {

            "basic":
                "Below ₹25,000",

            "standard":
                "₹25,000–₹75,000",

            "premium":
                "Above ₹75,000"
        },

        "renovation_processing":
            "asynchronous"
    })


# ============================================================
# 23. HEALTH
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "healthy",

        "cnn_model":
            "loaded",

        "cloudflare":
            bool(
                CLOUDFLARE_API_TOKEN
            )
    })


# ============================================================
# 24. RENOVATION OPTIONS
# ============================================================

@app.route(
    "/renovation-options",
    methods=["GET"]
)
def renovation_options():

    rooms = {}

    for room_id, room_info in ROOM_TYPES.items():

        rooms[
            room_id
        ] = {

            "name":
                room_info["name"],

            "options": [

                {
                    "id":
                        style,

                    "name":
                        STYLE_NAMES[
                            style
                        ]
                }

                for style in ALL_STYLES
            ]
        }

    return jsonify({

        "rooms":
            rooms,

        "budget_levels": [

            {
                "id":
                    "basic",

                "name":
                    "Basic",

                "range":
                    "Below ₹25,000"
            },

            {
                "id":
                    "standard",

                "name":
                    "Standard",

                "range":
                    "₹25,000–₹75,000"
            },

            {
                "id":
                    "premium",

                "name":
                    "Premium",

                "range":
                    "Above ₹75,000"
            }
        ],

        "furniture_options": [

            {
                "id":
                    "true",

                "name":
                    "Furniture Enabled"
            },

            {
                "id":
                    "false",

                "name":
                    "Furniture Disabled"
            }
        ]
    })


# ============================================================
# 25. PREDICT
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict_endpoint():

    try:

        uploaded_file = get_uploaded_file()

        if uploaded_file is None:

            return jsonify({

                "success":
                    False,

                "error":
                    "No image uploaded. "
                    "Expected multipart field 'file' or 'image'."

            }), 400

        image_bytes = (
            uploaded_file.read()
        )

        if not image_bytes:

            return jsonify({

                "success":
                    False,

                "error":
                    "Uploaded image is empty."

            }), 400

        image = load_image_from_bytes(
            image_bytes
        )

        original_filename = save_uploaded_image(
            image
        )

        original_path = (
            f"/uploads/{original_filename}"
        )

        original_url = build_absolute_url(
            original_path
        )

        (
            condition,
            confidence,
            probabilities
        ) = predict_damage(
            image
        )

        probabilities_list = [

            {
                "condition":
                    CLASS_NAMES[i],

                "condition_display":
                    format_condition(
                        CLASS_NAMES[i]
                    ),

                "percentage":
                    probabilities[
                        CLASS_NAMES[i]
                    ]
            }

            for i in range(
                len(CLASS_NAMES)
            )
        ]

        priority_result = (
            get_priority_recommendation(
                condition,
                confidence
            )
        )

        return jsonify({

            "success":
                True,

            "condition":
                condition,

            "condition_display":
                format_condition(
                    condition
                ),

            "confidence":
                confidence,

            "highest_confidence":
                confidence,

            "probabilities":
                probabilities,

            "probabilities_list":
                probabilities_list,

            "probability_total":
                round(
                    sum(
                        probabilities.values()
                    ),
                    2
                ),

            "priority":
                priority_result[
                    "priority"
                ],

            "priority_recommendation":
                priority_result[
                    "recommendation"
                ],

            "priority_message":
                priority_result[
                    "priority_message"
                ],

            "original_image_url":
                original_url,

            "original_image_path":
                original_path
        })

    except Exception as exc:

        traceback.print_exc()

        return jsonify({

            "success":
                False,

            "error":
                str(exc)

        }), 500


# ============================================================
# 26. MATERIAL RECOMMENDATION
# ============================================================

@app.route(
    "/material-recommendation",
    methods=["POST"]
)
def material_recommendation():

    try:

        data = (
            request.get_json(
                silent=True
            )
            or {}
        )

        condition = data.get(
            "condition",
            "normal"
        )

        room_type = normalize_room_type(
            data.get(
                "room_type",
                "living_room"
            )
        )

        budget = data.get(
            "budget"
        )

        furniture_enabled = data.get(
            "furniture_enabled",
            True
        )

        if isinstance(
            furniture_enabled,
            str
        ):

            furniture_enabled = (
                furniture_enabled
                .strip()
                .lower()
                not in [
                    "false",
                    "0",
                    "no",
                    "off"
                ]
            )

        if budget is None:

            recommendations = (
                get_material_recommendations(
                    condition,
                    room_type
                )
            )

            return jsonify({

                "success":
                    True,

                "room_type":
                    room_type,

                "room_name":
                    ROOM_TYPES[
                        room_type
                    ]["name"],

                "condition":
                    format_condition(
                        condition
                    ),

                "furniture_enabled":
                    furniture_enabled,

                "recommended_materials":
                    recommendations
            })

        result = (
            get_budget_material_recommendations(
                condition,
                room_type,
                budget
            )
        )

        result["success"] = True

        result["room_type"] = room_type

        result["room_name"] = (
            ROOM_TYPES[
                room_type
            ]["name"]
        )

        result["condition"] = (
            format_condition(
                condition
            )
        )

        result["furniture_enabled"] = (
            furniture_enabled
        )

        return jsonify(
            result
        )

    except Exception as exc:

        traceback.print_exc()

        return jsonify({

            "success":
                False,

            "error":
                str(exc)

        }), 500


# ============================================================
# 27. BUDGET PLAN
# ============================================================

@app.route(
    "/budget-plan",
    methods=["POST"]
)
def budget_plan():

    try:

        data = (
            request.get_json(
                silent=True
            )
            or {}
        )

        budget = data.get(
            "budget",
            0
        )

        condition = data.get(
            "condition",
            "normal"
        )

        room_type = normalize_room_type(
            data.get(
                "room_type",
                "living_room"
            )
        )

        furniture_enabled = data.get(
            "furniture_enabled",
            True
        )

        if isinstance(
            furniture_enabled,
            str
        ):

            furniture_enabled = (
                furniture_enabled
                .strip()
                .lower()
                not in [
                    "false",
                    "0",
                    "no",
                    "off"
                ]
            )

        result = create_budget_plan(

            room_type,

            condition,

            budget,

            furniture_enabled
        )

        result["success"] = True

        return jsonify(
            result
        )

    except Exception as exc:

        traceback.print_exc()

        return jsonify({

            "success":
                False,

            "error":
                str(exc)

        }), 500


# ============================================================
# 28. ASYNCHRONOUS RENOVATION GENERATION
# ============================================================

@app.route(
    "/renovation/generate",
    methods=["POST"]
)
def renovation_generate():

    try:

        # ----------------------------------------------------
        # Get uploaded image
        # ----------------------------------------------------

        uploaded_file = get_uploaded_file()

        if uploaded_file is None:

            return jsonify({

                "success":
                    False,

                "error":
                    "No image uploaded. "
                    "Expected multipart field 'file' or 'image'."

            }), 400

        image_bytes = (
            uploaded_file.read()
        )

        if not image_bytes:

            return jsonify({

                "success":
                    False,

                "error":
                    "Uploaded image is empty."

            }), 400

        # ----------------------------------------------------
        # Validate image BEFORE creating job
        # ----------------------------------------------------

        try:

            image = load_image_from_bytes(
                image_bytes
            )

        except Exception as exc:

            return jsonify({

                "success":
                    False,

                "error":
                    f"Invalid image: {exc}"

            }), 400

        # ----------------------------------------------------
        # Room type
        # ----------------------------------------------------

        room_type = normalize_room_type(

            request.form.get(
                "room_type",
                request.form.get(
                    "roomType",
                    "living_room"
                )
            )
        )

        # ----------------------------------------------------
        # Style
        # ----------------------------------------------------

        style = (

            request.form.get(
                "style",
                "minimal"
            )
            .strip()
            .lower()
        )

        if style not in STYLE_NAMES:

            style = "minimal"

        # ----------------------------------------------------
        # Furniture
        # ----------------------------------------------------

        furniture_value = (

            request.form.get(
                "furniture_enabled",
                request.form.get(
                    "furnitureEnabled",
                    "true"
                )
            )
            .strip()
            .lower()
        )

        furniture_enabled = (

            furniture_value
            not in [
                "false",
                "0",
                "no",
                "off"
            ]
        )

        # ----------------------------------------------------
        # Budget
        # ----------------------------------------------------

        budget_value = request.form.get(
            "budget",
            "0"
        )

        try:

            budget = float(
                budget_value
            )

        except Exception:

            budget = 0.0

        budget = max(
            0.0,
            budget
        )

        # ----------------------------------------------------
        # Create unique job
        # ----------------------------------------------------

        job_id = uuid.uuid4().hex

        input_filename = (
            f"{job_id}_input.jpg"
        )

        input_path = (
            JOB_FOLDER
            /
            input_filename
        )

        # ----------------------------------------------------
        # Save input image
        # ----------------------------------------------------

        image.save(
            input_path,
            format="JPEG",
            quality=90
        )

        # ----------------------------------------------------
        # Capture public base URL BEFORE leaving request
        # ----------------------------------------------------

        external_url = os.getenv(
            "RENDER_EXTERNAL_URL"
        )

        if external_url:

            base_url = (
                external_url.rstrip("/")
            )

        else:

            base_url = (
                f"{request.scheme}://"
                f"{request.host}"
            )

        # ----------------------------------------------------
        # Initial job status
        # ----------------------------------------------------

        initial_status = {

            "success":
                True,

            "job_id":
                job_id,

            "status":
                "queued",

            "message":
                "Renovation job created successfully.",

            "room_type":
                room_type,

            "style":
                style,

            "style_name":
                STYLE_NAMES[
                    style
                ],

            "budget":
                budget,

            "budget_level":
                get_budget_level(
                    budget
                ),

            "furniture_enabled":
                furniture_enabled,

            "status_url":
                build_absolute_url_from_base(

                    base_url,

                    f"/renovation/job/{job_id}"
                )
        }

        write_job_status(
            job_id,
            initial_status
        )

        # ----------------------------------------------------
        # Start background worker
        # ----------------------------------------------------

        worker = threading.Thread(

            target=
                process_renovation_job,

            args=(

                job_id,

                str(input_path),

                room_type,

                style,

                budget,

                furniture_enabled,

                base_url
            ),

            daemon=True
        )

        worker.start()

        print()
        print("=" * 70)
        print(
            "RENOVATION JOB QUEUED"
        )
        print("=" * 70)

        print(
            "Job ID:",
            job_id
        )

        print(
            "Status URL:",
            initial_status[
                "status_url"
            ]
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Return immediately.
        # Do NOT wait for Cloudflare.
        # ----------------------------------------------------

        return jsonify(
            initial_status
        ), 202

    except Exception as exc:

        traceback.print_exc()

        return jsonify({

            "success":
                False,

            "status":
                "failed",

            "error":
                str(exc)

        }), 500


# ============================================================
# 29. INDIVIDUAL RENOVATION JOB STATUS
# ============================================================

@app.route(
    "/renovation/job/<job_id>",
    methods=["GET"]
)
def renovation_job_status(
    job_id
):

    job_id = (
        str(job_id)
        .strip()
    )

    job = read_job_status(
        job_id
    )

    if job is None:

        return jsonify({

            "success":
                False,

            "status":
                "not_found",

            "error":
                "Renovation job not found."

        }), 404

    return jsonify(
        job
    )


# ============================================================
# 30. RENOVATION IMAGE SERVING
# ============================================================

@app.route(
    "/renovation/image/<filename>",
    methods=["GET"]
)
def renovation_image(
    filename
):

    filename = Path(
        filename
    ).name

    safe_path = (
        RENOVATION_FOLDER
        /
        filename
    )

    if not safe_path.exists():

        return jsonify({

            "success":
                False,

            "error":
                "Renovation image not found."

        }), 404

    suffix = (
        safe_path.suffix
        .lower()
    )

    if suffix == ".png":

        mimetype = "image/png"

    elif suffix in [
        ".jpg",
        ".jpeg"
    ]:

        mimetype = "image/jpeg"

    elif suffix == ".webp":

        mimetype = "image/webp"

    else:

        mimetype = "application/octet-stream"

    return send_file(

        safe_path,

        mimetype=mimetype,

        conditional=True,

        max_age=0
    )


# ============================================================
# 31. UPLOADED IMAGE SERVING
# ============================================================

@app.route(
    "/uploads/<filename>",
    methods=["GET"]
)
def uploaded_image(
    filename
):

    filename = Path(
        filename
    ).name

    safe_path = (
        UPLOAD_FOLDER
        /
        filename
    )

    if not safe_path.exists():

        return jsonify({

            "success":
                False,

            "error":
                "Uploaded image not found."

        }), 404

    return send_file(

        safe_path,

        mimetype="image/jpeg",

        conditional=True,

        max_age=0
    )


# ============================================================
# 32. RENOVATION STATUS / GENERATED IMAGES
# ============================================================

@app.route(
    "/renovation-status",
    methods=["GET"]
)
def renovation_status():

    images = []

    paths = (

        list(
            RENOVATION_FOLDER.glob(
                "*.png"
            )
        )
        +
        list(
            RENOVATION_FOLDER.glob(
                "*.jpg"
            )
        )
        +
        list(
            RENOVATION_FOLDER.glob(
                "*.jpeg"
            )
        )
    )

    paths = sorted(

        paths,

        key=lambda p:
            p.stat().st_mtime,

        reverse=True
    )

    for path in paths:

        image_path = (

            f"/renovation/image/"
            f"{path.name}"
        )

        image_url = build_absolute_url(
            image_path
        )

        images.append({

            "filename":
                path.name,

            "url":
                image_path,

            "image_url":
                image_url
        })

    return jsonify({

        "success":
            True,

        "count":
            len(images),

        "images":
            images[:20]
    })


# ============================================================
# 33. FILE SIZE ERROR
# ============================================================

@app.errorhandler(
    413
)
def file_too_large(
    error
):

    return jsonify({

        "success":
            False,

        "error":
            "Image file is too large. "
            "Maximum size is 15 MB."

    }), 413


# ============================================================
# 34. GENERAL ERROR HANDLER
# ============================================================

@app.errorhandler(
    Exception
)
def handle_general_error(
    error
):

    traceback.print_exc()

    return jsonify({

        "success":
            False,

        "error":
            str(error)

    }), 500


# ============================================================
# 35. START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("ROOM ANALYSIS AI SERVER")
    print("=" * 70)

    print(
        "CNN:",
        MODEL_PATH
    )

    print(
        "Cloudflare model:",
        CLOUDFLARE_MODEL
    )

    print(
        "Upload folder:",
        UPLOAD_FOLDER
    )

    print(
        "Renovation folder:",
        RENOVATION_FOLDER
    )

    print(
        "Job folder:",
        JOB_FOLDER
    )

    print()
    print("Budget levels:")

    print(
        "Basic    : Below ₹25,000"
    )

    print(
        "Standard : ₹25,000–₹75,000"
    )

    print(
        "Premium  : Above ₹75,000"
    )

    print()
    print("Furniture OFF:")

    print(
        "No new furniture will be requested."
    )

    print()
    print("Renovation processing:")

    print(
        "ASYNC / BACKGROUND JOB"
    )

    print()
    print("=" * 70)

    app.run(

        host="0.0.0.0",

        port=int(
            os.getenv(
                "PORT",
                5000
            )
        ),

        debug=False,

        threaded=True
    )