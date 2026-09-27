# ============================================================
# ROOM ANALYSIS AI - FINAL BACKEND
# Flask + CNN Damage Detection
# + Priority-Based Recommendations
# + Room-Specific Recommendations
# + Budget Planner
# + Budget-Aware Renovation
# + Material Recommendations
# + Cloudflare FLUX.2 Klein 4B Renovation
# ============================================================

import os
import io
import uuid
import base64
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
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

    """
    Convert uploaded image bytes into RGB PIL image.
    """

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    return image


def image_to_model_array(image):

    """
    Prepare image for CNN.

    The saved model already contains its required
    preprocessing layer, so preprocessing is not
    applied again here.
    """

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

    """
    Save original room image so Android can load it
    from the Flask server.
    """

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

    """
    Accept both multipart field names:

    file
    image

    This keeps Android and other clients compatible.
    """

    uploaded_file = (
        request.files.get("file")
        or request.files.get("image")
    )

    return uploaded_file


def build_absolute_url(
    path
):

    """
    Convert an API path into an absolute URL.

    Example:

    /renovation/image/test.png

    becomes:

    http://192.168.1.6:5000/renovation/image/test.png
    """

    host = request.host

    scheme = request.scheme

    return (
        f"{scheme}://{host}{path}"
    )


# ============================================================
# 10. CNN DAMAGE PREDICTION
# ============================================================

def predict_damage(image):

    """
    Predict room damage.

    Returns:

        condition
        confidence
        probabilities
    """

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

    # --------------------------------------------------------
    # Validate output length
    # --------------------------------------------------------

    if len(raw_predictions) != len(
        CLASS_NAMES
    ):

        raise RuntimeError(
            "CNN output does not contain exactly "
            "four class predictions."
        )

    # --------------------------------------------------------
    # Convert model output safely to probabilities
    #
    # Case 1:
    # Already valid probabilities:
    # values between 0 and 1 and sum approximately 1.
    #
    # Case 2:
    # Values between 0 and 1 but not normalized:
    # normalize them.
    #
    # Case 3:
    # Logits / arbitrary scores:
    # apply softmax.
    # --------------------------------------------------------

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

        # ----------------------------------------------------
        # Stable softmax
        # ----------------------------------------------------

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
            "CNN output interpreted as logits and "
            "converted using softmax."
        )

    # --------------------------------------------------------
    # Final safety normalization
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Highest confidence condition
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Percentages
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Correct any tiny rounding difference
    # --------------------------------------------------------

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

        probabilities[
            condition
        ] = round(
            probabilities[
                condition
            ]
            +
            difference,
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
        sum(
            probabilities.values()
        ),
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

def format_condition(
    condition
):

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
        str(
            condition
        )
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

    """
    Determine renovation priority using the detected
    condition and its highest confidence.

    The detected damage type has the main role.
    Confidence is used to communicate detection strength.
    """

    if condition == "dampness":

        return "High"

    if condition == "crack":

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

    """
    Give recommendation based on the highest-confidence
    detected condition.
    """

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

def normalize_room_type(
    room_type
):

    if not room_type:

        return "living_room"

    value = (
        str(
            room_type
        )
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

def get_budget_level(
    budget
):

    try:

        budget = float(
            budget
        )

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

        budget = float(
            budget
        )

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

    # --------------------------------------------------------
    # Furniture OFF
    # --------------------------------------------------------

    if not furniture_enabled:

        if room_type == "living_room":

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

        elif room_type == "bedroom":

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
                        "Basic bedroom lighting",
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

    # --------------------------------------------------------
    # Furniture ON + Basic
    # --------------------------------------------------------

    elif budget_level == "Basic":

        if room_type == "living_room":

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

        elif room_type == "bedroom":

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

    # --------------------------------------------------------
    # Furniture ON + Standard
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Furniture ON + Premium
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Convert percentages into amounts
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Correct rounding difference
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DAMAGE REPAIR
    # --------------------------------------------------------

    if condition == "crack":

        repair_text = (
            "Repair visible wall cracks and restore "
            "affected surfaces before applying the "
            "new finish."
        )

    elif condition == "dampness":

        repair_text = (
            "Visually remove visible dampness and "
            "moisture stains, restore affected wall "
            "surfaces and use an appropriate moisture-"
            "resistant finish."
        )

    elif condition == "peeling_paint":

        repair_text = (
            "Remove visible peeling and loose paint, "
            "prepare the affected wall and restore "
            "the surface with a clean finished coating."
        )

    else:

        repair_text = (
            "Keep the existing room surfaces in good "
            "condition and make appropriate aesthetic "
            "improvements without unnecessary repair work."
        )

    # --------------------------------------------------------
    # STYLE
    # --------------------------------------------------------

    if style == "minimal":

        style_text = (
            "Clean minimalist interior, simple forms, "
            "uncluttered surfaces, light neutral tones "
            "and practical design."
        )

    elif style == "contemporary":

        style_text = (
            "Contemporary interior with clean lines, "
            "balanced neutral colors, modern finishes "
            "and subtle decorative details."
        )

    elif style == "luxury":

        style_text = (
            "Elegant luxury interior with refined finishes, "
            "sophisticated textures, coordinated lighting "
            "and premium-looking details."
        )

    elif style == "budget_friendly":

        style_text = (
            "Affordable practical interior with simple "
            "durable finishes, economical improvements "
            "and limited decorative changes."
        )

    else:

        style_text = (
            "Eco-friendly interior using natural-looking "
            "materials, efficient lighting, sustainable-"
            "looking furniture and calm natural tones."
        )

    # --------------------------------------------------------
    # BUDGET
    # --------------------------------------------------------

    if budget_level == "Basic":

        budget_text = (
            "BASIC renovation below INR 25,000. "
            "Keep changes small and practical. "
            "Prioritize detected damage and wall repair. "
            "Use simple finishes and basic lighting. "
            "Do not create an expensive complete makeover. "
            "Avoid premium furniture, major structural work "
            "and unnecessary expensive materials."
        )

    elif budget_level == "Standard":

        budget_text = (
            "STANDARD renovation from INR 25,000 to INR 75,000. "
            "Apply moderate practical improvements including "
            "essential repairs, improved wall finishes, lighting "
            "and selected furniture or decor."
        )

    else:

        budget_text = (
            "PREMIUM renovation above INR 75,000. "
            "Allow more extensive coordinated improvements, "
            "higher-quality finishes, furniture, lighting, "
            "storage and decor while preserving the room structure."
        )

    # --------------------------------------------------------
    # FURNITURE
    # --------------------------------------------------------

    if not furniture_enabled:

        furniture_text = (
            "FURNITURE OFF. Do not add new furniture. "
            "Do not add a sofa, chair, table, bed, coffee table, "
            "TV unit, wardrobe or new storage. Do not replace "
            "existing furniture. Keep existing furniture unchanged "
            "where visible. Focus only on permitted surfaces, "
            "lighting and non-furniture improvements."
        )

    elif budget_level == "Basic":

        furniture_text = (
            "FURNITURE ON with BASIC budget. Use furniture "
            "extremely sparingly. Add at most one simple, "
            "inexpensive practical furniture element if suitable. "
            "Do not add large sofa sets, premium furniture, "
            "large coffee tables, expensive TV units, wardrobes "
            "or multiple new furniture pieces."
        )

    elif budget_level == "Standard":

        furniture_text = (
            "FURNITURE ON with STANDARD budget. Allow a limited "
            "number of practical coordinated furniture and decor "
            "elements appropriate to the selected room type. "
            "Avoid excessive luxury furniture."
        )

    else:

        furniture_text = (
            "FURNITURE ON with PREMIUM budget. Allow more complete "
            "coordinated furniture and decor appropriate to the "
            "selected room type and selected style."
        )

    # --------------------------------------------------------
    # ROOM CHANGES
    # --------------------------------------------------------

    if budget_level == "Basic":

        room_changes = (
            f"For the {room_name}, make minimal changes. "
            "Keep most existing room elements. Focus mainly "
            "on damage repair, repainting, simple lighting and "
            "a very small amount of affordable styling."
        )

    elif budget_level == "Standard":

        room_changes = (
            f"For the {room_name}, make moderate coordinated "
            "changes. Improve walls, lighting and selected "
            "furniture or decor while keeping the original layout."
        )

    else:

        room_changes = (
            f"For the {room_name}, allow a more complete "
            "coordinated renovation using appropriate furniture, "
            "lighting, finishes and decor."
        )

    # --------------------------------------------------------
    # FINAL PROMPT
    # --------------------------------------------------------

    prompt = f"""
Photorealistic renovation of the SAME {room_name} shown in the
input photograph.

SELECTED STYLE:
{style_name}

BUDGET:
INR {budget:.0f}

RENOVATION LEVEL:
{budget_level}

RENOVATION SCOPE:
{renovation_scope}

DETECTED CONDITION:
{format_condition(condition)}

DAMAGE REPAIR:
{repair_text}

STYLE:
{style_text}

BUDGET RULE:
{budget_text}

ROOM CHANGES:
{room_changes}

FURNITURE RULE:
{furniture_text}

IMPORTANT:
The final image must look like the SAME photographed room
after renovation.

Preserve the exact room structure.
Preserve walls.
Preserve doors.
Preserve windows.
Preserve ceiling shape.
Preserve floor geometry.
Preserve room proportions.
Preserve the original camera viewpoint.
Preserve the original perspective.
Preserve the architectural layout.

Do not create a different room.
Do not move doors.
Do not move windows.
Do not change the number of windows.
Do not change the room dimensions.
Do not change the camera angle.
Do not perform structural remodeling.

The selected style controls the visual appearance.
The budget controls the extent and quality of renovation.
The furniture setting must be strictly respected.

For BASIC renovation, keep changes visibly minimal.
For STANDARD renovation, make moderate coordinated improvements.
For PREMIUM renovation, allow more extensive coordinated improvements.

Repair the detected condition before aesthetic renovation.

Realistic interior materials.
Natural lighting.
Realistic shadows.
Photorealistic interior photography.
High-quality realistic renovation.
"""

    prompt = " ".join(
        prompt.split()
    )

    return prompt


# ============================================================
# 19. CLOUDFLARE FLUX GENERATION
# ============================================================

def generate_with_cloudflare(
    image,
    prompt,
    output_path
):

    if not CLOUDFLARE_API_TOKEN:

        raise RuntimeError(
            "CLOUDFLARE_API_TOKEN is missing from .env."
        )

    import requests

    endpoint = (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{CLOUDFLARE_ACCOUNT_ID}"
        f"/ai/run/"
        f"{CLOUDFLARE_MODEL}"
    )

    # --------------------------------------------------------
    # Input image
    # --------------------------------------------------------

    image_buffer = io.BytesIO()

    image.convert(
        "RGB"
    ).save(
        image_buffer,
        format="JPEG",
        quality=95
    )

    image_buffer.seek(0)

    files = {

        "input_image": (

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
        "Sending image to Cloudflare..."
    )

    response = requests.post(

        endpoint,

        headers=headers,

        files=files,

        data=data,

        timeout=300
    )

    print(
        "Cloudflare HTTP status:",
        response.status_code
    )

    if response.status_code != 200:

        print(
            "Cloudflare response:",
            response.text[:3000]
        )

        raise RuntimeError(
            "Cloudflare image generation failed."
        )

    result = response.json()

    if not result.get(
        "success",
        False
    ):

        raise RuntimeError(
            "Cloudflare returned an unsuccessful response."
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

    # --------------------------------------------------------
    # Decode Base64
    # --------------------------------------------------------

    try:

        generated_bytes = base64.b64decode(
            image_base64
        )

    except Exception as exc:

        raise RuntimeError(
            "Unable to decode Cloudflare image."
        ) from exc

    # --------------------------------------------------------
    # IMPORTANT:
    # Decode the generated bytes using PIL.
    #
    # This prevents us from incorrectly calling a PNG
    # image a JPEG.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Always save a valid PNG.
    # --------------------------------------------------------

    generated_image.save(
        output_path,
        format="PNG"
    )

    print(
        "Renovation image saved:"
    )

    print(
        output_path
    )

    print(
        "Generated image size:",
        generated_image.size
    )

    return output_path


# ============================================================
# 20. HOME
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
        }
    })


# ============================================================
# 21. HEALTH
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
# 22. RENOVATION OPTIONS
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
# 23. PREDICT
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

        # ----------------------------------------------------
        # Save original image
        # ----------------------------------------------------

        original_filename = save_uploaded_image(
            image
        )

        original_path = (
            f"/uploads/{original_filename}"
        )

        original_url = build_absolute_url(
            original_path
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

        # ----------------------------------------------------
        # Priority recommendation
        # ----------------------------------------------------

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
# 24. MATERIAL RECOMMENDATION
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

        condition = (
            data.get(
                "condition",
                "normal"
            )
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

        furniture_enabled = (
            data.get(
                "furniture_enabled",
                True
            )
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

        result["room_type"] = (
            room_type
        )

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
# 25. BUDGET PLAN
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

        furniture_enabled = (
            data.get(
                "furniture_enabled",
                True
            )
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
# 26. ASYNC RENOVATION JOB SUPPORT
# ============================================================

# Render uses a small instance, so only one expensive renovation
# job is processed at a time. ThreadPoolExecutor is used instead
# of creating a daemon Thread for every request.
RENOVATION_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="renovation"
)


def log_message(*args):

    """Print immediately so Render logs are visible without delay."""

    print(
        *args,
        flush=True
    )


def get_job_file(job_id):

    return JOB_FOLDER / f"{job_id}.json"


def write_job_status(
    job_id,
    **data
):

    """Write the current renovation-job state to disk."""

    job_file = get_job_file(
        job_id
    )

    payload = {
        "job_id": job_id,
        **data
    }

    temporary_file = job_file.with_suffix(
        ".tmp"
    )

    temporary_file.write_text(
        __import__("json").dumps(
            payload,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    temporary_file.replace(
        job_file
    )

    return payload


def read_job_status(job_id):

    """Read a saved renovation-job state."""

    job_file = get_job_file(
        job_id
    )

    if not job_file.exists():

        return None

    return __import__("json").loads(
        job_file.read_text(
            encoding="utf-8"
        )
    )


def build_absolute_url_from_base(
    base_url,
    path
):

    """Build an image/job URL without depending on Flask request context."""

    base_url = (
        str(base_url or "")
        .strip()
        .rstrip("/")
    )

    path = "/" + str(path).lstrip("/")

    return f"{base_url}{path}"


def process_renovation_job(
    job_id,
    input_path,
    room_type,
    style,
    budget,
    furniture_enabled,
    base_url
):

    """Process one renovation job in the managed background executor."""

    try:

        log_message()
        log_message("=" * 70)
        log_message(
            f"STARTING BACKGROUND RENOVATION JOB: {job_id}"
        )
        log_message("=" * 70)

        write_job_status(
            job_id,
            status="processing",
            success=True,
            message="Room analysis and renovation generation are in progress."
        )

        log_message("STEP 1: Background function entered.")
        log_message("STEP 2: Job status changed to processing.")
        log_message("STEP 2A: Loading saved input image.")

        input_file = Path(
            input_path
        )

        image_bytes = input_file.read_bytes()

        log_message(
            "Input image bytes:",
            len(image_bytes)
        )

        image = load_image_from_bytes(
            image_bytes
        )

        log_message(
            "STEP 2B: Input image loaded.",
            image.size
        )

        log_message(
            "STEP 2C: Calculating budget level."
        )

        budget_level = get_budget_level(
            budget
        )

        renovation_scope = get_renovation_scope(
            budget_level
        )

        log_message(
            "Budget:",
            budget
        )

        log_message(
            "Budget level:",
            budget_level
        )

        write_job_status(
            job_id,
            status="analyzing",
            success=True,
            message="Analyzing room condition using the CNN model.",
            room_type=room_type,
            room_name=ROOM_TYPES[room_type]["name"],
            style=style,
            style_name=STYLE_NAMES[style],
            budget=budget,
            budget_level=budget_level,
            furniture_enabled=furniture_enabled
        )

        # ----------------------------------------------------
        # CNN
        # ----------------------------------------------------

        log_message(
            "STEP 3: Calling CNN damage detection."
        )

        log_message(
            "STEP 3: Starting CNN prediction."
        )

        (
            condition,
            confidence,
            probabilities
        ) = predict_damage(
            image
        )

        log_message(
            "STEP 4: CNN prediction completed."
        )

        log_message(
            "Background job condition:",
            condition
        )

        log_message(
            "Background job confidence:",
            confidence
        )

        priority_result = get_priority_recommendation(
            condition,
            confidence
        )

        log_message(
            "STEP 5: Creating priority recommendations."
        )

        write_job_status(
            job_id,
            status="generating",
            success=True,
            message="Room condition analyzed. Generating renovation visualization.",
            room_type=room_type,
            room_name=ROOM_TYPES[room_type]["name"],
            style=style,
            style_name=STYLE_NAMES[style],
            condition=condition,
            condition_display=format_condition(condition),
            confidence=confidence,
            highest_confidence=confidence,
            probabilities=probabilities,
            priority=priority_result["priority"],
            budget=budget,
            budget_level=budget_level,
            furniture_enabled=furniture_enabled
        )

        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        log_message(
            "STEP 6: Building renovation prompt."
        )

        prompt = build_room_style_prompt(
            room_type=room_type,
            style=style,
            condition=condition,
            budget=budget,
            furniture_enabled=furniture_enabled
        )

        log_message(
            "Prompt length:",
            len(prompt)
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
            / filename
        )

        log_message(
            "Output path:",
            output_path
        )

        # ----------------------------------------------------
        # Cloudflare generation
        # ----------------------------------------------------

        log_message(
            "STEP 7: Starting Cloudflare generation."
        )

        generate_with_cloudflare(
            image=image,
            prompt=prompt,
            output_path=output_path
        )

        log_message(
            "STEP 8: Cloudflare generation completed."
        )

        # ----------------------------------------------------
        # Recommendations and budget plan
        # ----------------------------------------------------

        log_message(
            "STEP 9: Creating material recommendations."
        )

        recommendations = get_budget_material_recommendations(
            condition,
            room_type,
            budget
        )

        log_message(
            "STEP 10: Creating budget plan."
        )

        budget_plan_result = create_budget_plan(
            room_type,
            condition,
            budget,
            furniture_enabled
        )

        # ----------------------------------------------------
        # Estimates retained for compatibility with the
        # existing Android/backend result structure.
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
                "condition": class_name,
                "condition_display": format_condition(class_name),
                "percentage": probabilities[class_name]
            }
            for class_name in CLASS_NAMES
        ]

        probability_total = round(
            sum(probabilities.values()),
            2
        )

        # ----------------------------------------------------
        # Image URLs
        # ----------------------------------------------------

        image_path = (
            f"/renovation/image/{filename}"
        )

        image_url = build_absolute_url_from_base(
            base_url,
            image_path
        )

        # ----------------------------------------------------
        # Original uploaded image
        # ----------------------------------------------------

        log_message(
            "STEP 11: Saving original image."
        )

        original_filename = save_uploaded_image(
            image,
            filename=f"{job_id}_original.jpg"
        )

        original_path = (
            f"/uploads/{original_filename}"
        )

        original_url = build_absolute_url_from_base(
            base_url,
            original_path
        )

        # ----------------------------------------------------
        # Completed result
        # ----------------------------------------------------

        log_message(
            "STEP 12: Writing completed job status."
        )

        completed_result = {
            "success": True,
            "message": "Renovation generated successfully.",
            "job_id": job_id,
            "status": "completed",
            "room_type": room_type,
            "room_name": ROOM_TYPES[room_type]["name"],
            "style": style,
            "style_name": STYLE_NAMES[style],
            "condition": condition,
            "condition_display": format_condition(condition),
            "confidence": confidence,
            "highest_confidence": confidence,
            "probabilities": probabilities,
            "probabilities_list": probabilities_list,
            "probability_total": probability_total,
            "priority": priority_result["priority"],
            "priority_recommendation": priority_result["recommendation"],
            "priority_message": priority_result["priority_message"],
            "budget": budget,
            "budget_level": budget_level,
            "renovation_scope": renovation_scope,
            "furniture_enabled": furniture_enabled,
            "material_recommendations": recommendations,
            "budget_plan": budget_plan_result,
            "optimistic_estimate": optimistic_estimate,
            "realistic_estimate": realistic_estimate,
            "filename": filename,
            "image_path": image_path,
            "image_url": image_url,
            "renovated_image_url": image_url,
            "renovation_image_url": image_url,
            "original_image_url": original_url
        }

        write_job_status(
            **completed_result
        )

        log_message()
        log_message("=" * 70)
        log_message(
            f"RENOVATION JOB COMPLETED: {job_id}"
        )
        log_message("=" * 70)

        return completed_result

    except Exception as exc:

        log_message()
        log_message("=" * 70)
        log_message(
            f"RENOVATION JOB FAILED: {job_id}"
        )
        log_message(
            "Error:",
            str(exc)
        )
        log_message("=" * 70)

        traceback.print_exc()

        try:

            write_job_status(
                job_id,
                status="failed",
                success=False,
                message="Room analysis and renovation generation failed.",
                error=str(exc)
            )

        except Exception:

            traceback.print_exc()

        return None


def renovation_future_done(
    future
):

    """Log unexpected executor-level failures."""

    try:

        future.result()

        log_message(
            "Background renovation future finished."
        )

    except Exception as exc:

        log_message(
            "Background renovation future crashed:",
            str(exc)
        )

        traceback.print_exc()


# ============================================================
# 27. RENOVATION GENERATION
# ============================================================

@app.route(
    "/renovation/generate",
    methods=["POST"]
)
def renovation_generate():

    try:

        log_message()
        log_message("=" * 70)
        log_message("NEW RENOVATION REQUEST")
        log_message("=" * 70)

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        uploaded_file = get_uploaded_file()

        if uploaded_file is None:

            return jsonify({
                "success": False,
                "error":
                    "No image uploaded. "
                    "Expected multipart field 'file' or 'image'."
            }), 400

        image_bytes = uploaded_file.read()

        if not image_bytes:

            return jsonify({
                "success": False,
                "error": "Uploaded image is empty."
            }), 400

        image = load_image_from_bytes(
            image_bytes
        )

        log_message(
            "Uploaded image:",
            image.size
        )

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

        furniture_value = request.form.get(
            "furniture_enabled",
            request.form.get(
                "furnitureEnabled",
                "true"
            )
        )

        furniture_value = (
            str(furniture_value)
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

        budget_level = get_budget_level(
            budget
        )

        log_message(
            "Room type:",
            room_type
        )

        log_message(
            "Style:",
            style
        )

        log_message(
            "Budget:",
            budget
        )

        log_message(
            "Budget level:",
            budget_level
        )

        log_message(
            "Furniture enabled:",
            furniture_enabled
        )

        # ----------------------------------------------------
        # Create job ID and save input
        # ----------------------------------------------------

        job_id = uuid.uuid4().hex

        input_path = (
            JOB_FOLDER
            / f"{job_id}_input.jpg"
        )

        image.convert(
            "RGB"
        ).save(
            input_path,
            format="JPEG",
            quality=95
        )

        log_message(
            "Saved job input:",
            input_path
        )

        # ----------------------------------------------------
        # Public base URL
        # ----------------------------------------------------

        render_url = os.getenv(
            "RENDER_EXTERNAL_URL"
        )

        if render_url:

            base_url = render_url.rstrip(
                "/"
            )

        else:

            base_url = request.host_url.rstrip(
                "/"
            )

        log_message(
            "Base URL:",
            base_url
        )

        # ----------------------------------------------------
        # Initial job status
        # ----------------------------------------------------

        status_url = (
            f"{base_url}"
            f"/renovation/job/{job_id}"
        )

        write_job_status(
            job_id,
            status="queued",
            success=True,
            message="Renovation job created successfully.",
            room_type=room_type,
            room_name=ROOM_TYPES[room_type]["name"],
            style=style,
            style_name=STYLE_NAMES[style],
            budget=budget,
            budget_level=budget_level,
            furniture_enabled=furniture_enabled,
            status_url=status_url
        )

        log_message(
            "Initial job status written."
        )

        # ----------------------------------------------------
        # Submit to managed executor
        # ----------------------------------------------------

        log_message(
            f"Submitting background worker: renovation-{job_id}"
        )

        future = RENOVATION_EXECUTOR.submit(
            process_renovation_job,
            job_id,
            str(input_path),
            room_type,
            style,
            budget,
            furniture_enabled,
            base_url
        )

        future.add_done_callback(
            renovation_future_done
        )

        log_message(
            "Background worker submitted to ThreadPoolExecutor."
        )

        return jsonify({
            "success": True,
            "message": "Renovation job created successfully.",
            "job_id": job_id,
            "status": "queued",
            "status_url": status_url,
            "room_type": room_type,
            "style": style,
            "style_name": STYLE_NAMES[style],
            "budget": budget,
            "budget_level": budget_level,
            "furniture_enabled": furniture_enabled
        }), 202

    except Exception as exc:

        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(exc)
        }), 500


# ============================================================
# 28. RENOVATION JOB STATUS
# ============================================================

@app.route(
    "/renovation/job/<job_id>",
    methods=["GET"]
)
def renovation_job_status(
    job_id
):

    try:

        job = read_job_status(
            job_id
        )

        if job is None:

            return jsonify({
                "success": False,
                "error": "Renovation job not found.",
                "job_id": job_id
            }), 404

        return jsonify(
            job
        )

    except Exception as exc:

        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(exc),
            "job_id": job_id
        }), 500


# ============================================================
# 29. RENOVATION IMAGE SERVING
# ============================================================


@app.route(
    "/renovation/image/<filename>",
    methods=["GET"]
)
def renovation_image(
    filename
):

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

    # --------------------------------------------------------
    # Determine MIME type from actual extension
    # --------------------------------------------------------

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

        conditional=True
    )


# ============================================================
# 28. UPLOADED IMAGE SERVING
# ============================================================

@app.route(
    "/uploads/<filename>",
    methods=["GET"]
)
def uploaded_image(
    filename
):

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

        conditional=True
    )


# ============================================================
# 29. RENOVATION STATUS
# ============================================================

@app.route(
    "/renovation-status",
    methods=["GET"]
)
def renovation_status():

    images = []

    for path in sorted(

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
        ),

        key=lambda p:
            p.stat().st_mtime,

        reverse=True
    ):

        image_path = (

            f"/renovation/image/"
            f"{path.name}"
        )

        images.append({

            "filename":
                path.name,

            "url":
                image_path,

            "image_url":
                build_absolute_url(
                    image_path
                )
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
# 30. FILE SIZE ERROR
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
# 31. GENERAL ERROR HANDLER
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
# 32. START SERVER
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
        "ASYNC / BACKGROUND JOB (ThreadPoolExecutor)"
    )

    print()
    print("Room types:")

    print(
        "Living Room"
    )

    print(
        "Bedroom"
    )

    print(
        "Kitchen"
    )

    print()
    print("Renovation styles:")

    print(
        "Minimal"
    )

    print(
        "Contemporary"
    )

    print(
        "Luxury"
    )

    print(
        "Budget-Friendly"
    )

    print(
        "Eco-Friendly"
    )

    print()
    print("=" * 70)

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=False,

        threaded=True
    )