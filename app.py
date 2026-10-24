import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "tomato_disease_model.keras"
CLASS_PATH = BASE_DIR / "class_names.json"
IMAGE_SIZE = (224, 224)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
logging.basicConfig(level=logging.INFO)

model = None
class_names = []

# Recommendations are general educational guidance. Confirm diagnosis and
# treatment choices with a local agricultural specialist.
TREATMENTS = {
    "Tomato___Bacterial_spot": {
        "name": "Bacterial Spot",
        "description": "A bacterial disease that may cause small, dark, water-soaked spots on leaves and fruit.",
        "treatment": [
            "Remove severely affected plant material where practical and dispose of it safely.",
            "Avoid handling plants while foliage is wet.",
            "Ask a local agricultural specialist about locally approved management products."
        ],
        "prevention": [
            "Use healthy seedlings and reputable seed sources.",
            "Avoid overhead watering where possible.",
            "Clean tools and wash hands after handling affected plants."
        ]
    },
    "Tomato___Early_blight": {
        "name": "Early Blight",
        "description": "A fungal disease often associated with brown leaf spots that may develop concentric rings.",
        "treatment": [
            "Remove badly affected lower leaves if doing so will not excessively weaken the plant.",
            "Keep foliage dry and water near the soil.",
            "Use fungicides only if locally approved and exactly as directed on the product label."
        ],
        "prevention": [
            "Rotate tomato crops with non-solanaceous crops when practical.",
            "Provide good spacing and air circulation.",
            "Remove diseased plant debris after harvest."
        ]
    },
    "Tomato___Late_blight": {
        "name": "Late Blight",
        "description": "A potentially destructive disease that can cause rapidly spreading dark lesions, especially in cool, wet conditions.",
        "treatment": [
            "Seek advice from an agricultural specialist promptly if late blight is suspected.",
            "Avoid moving potentially infected plant material between growing areas.",
            "Follow local agricultural guidance for any approved control products."
        ],
        "prevention": [
            "Inspect plants frequently during cool, wet weather.",
            "Avoid prolonged leaf wetness and improve airflow.",
            "Use healthy seedlings and follow local disease advisories."
        ]
    },
    "Tomato___Leaf_Mold": {
        "name": "Leaf Mold",
        "description": "A disease favored by high humidity; affected leaves may show yellow patches above and mold growth underneath.",
        "treatment": [
            "Remove severely affected leaves when practical.",
            "Improve ventilation and reduce excess humidity around plants.",
            "Seek local advice before applying a disease-management product."
        ],
        "prevention": [
            "Space plants to improve air circulation.",
            "Water at the base of plants where possible.",
            "Avoid excessive humidity in protected growing areas."
        ]
    },
    "Tomato___Septoria_leaf_spot": {
        "name": "Septoria Leaf Spot",
        "description": "A fungal disease commonly associated with many small, round leaf spots with pale centers and dark margins.",
        "treatment": [
            "Remove affected lower leaves when practical.",
            "Avoid splashing soil onto foliage during watering.",
            "Consult local agricultural guidance about approved treatments."
        ],
        "prevention": [
            "Rotate crops when practical.",
            "Mulch to reduce soil splash.",
            "Clean up infected plant debris."
        ]
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "name": "Two-Spotted Spider Mite Damage",
        "description": "Feeding by spider mites may cause fine stippling, yellowing, and sometimes webbing on leaves.",
        "treatment": [
            "Check the undersides of leaves carefully to confirm whether mites are present.",
            "Ask a local agricultural specialist about suitable, crop-safe control options.",
            "Avoid unnecessary pesticide use before confirming the cause."
        ],
        "prevention": [
            "Inspect leaf undersides regularly.",
            "Reduce dust and plant stress where possible.",
            "Monitor nearby plants for signs of infestation."
        ]
    },
    "Tomato___Target_Spot": {
        "name": "Target Spot",
        "description": "A leaf disease that may produce brown lesions with concentric patterns.",
        "treatment": [
            "Remove severely affected plant material when practical.",
            "Keep foliage dry and improve airflow.",
            "Consult local agricultural guidance for approved management options."
        ],
        "prevention": [
            "Avoid overcrowding plants.",
            "Water at the base where possible.",
            "Remove infected debris and monitor plants regularly."
        ]
    },
    "Tomato___Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "description": "A viral disease that may cause mottled leaves, distortion, or stunted growth.",
        "treatment": [
            "There is no curative treatment for an infected plant; confirm the diagnosis with an expert.",
            "Remove and dispose of confirmed infected plants according to local guidance.",
            "Clean tools carefully to reduce mechanical spread."
        ],
        "prevention": [
            "Use clean seed and healthy transplants.",
            "Wash hands and sanitize tools between plants.",
            "Avoid handling healthy plants after handling suspect plants without cleaning."
        ]
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "description": "A viral disease that can cause upward leaf curling, yellowing, and stunted growth.",
        "treatment": [
            "Confirm the suspected virus with an agricultural specialist.",
            "Ask about local whitefly management and crop-removal guidance.",
            "Do not assume a pesticide will cure a viral infection."
        ],
        "prevention": [
            "Use healthy transplants and monitor for whiteflies.",
            "Remove weeds that may harbor pests, following local guidance.",
            "Use locally recommended exclusion or pest-management practices."
        ]
    },
    "Tomato___healthy": {
        "name": "Healthy Tomato Leaf",
        "description": "The model classified this image as a healthy tomato leaf.",
        "treatment": [
            "No disease-specific treatment is indicated by this prediction.",
            "Continue regular plant inspection and normal care."
        ],
        "prevention": [
            "Water appropriately and maintain good drainage.",
            "Provide suitable sunlight and plant spacing.",
            "Monitor leaves regularly for changes."
        ]
    }
}


def load_model_files():
    global model, class_names
    if MODEL_PATH.exists() and CLASS_PATH.exists():
        model = tf.keras.models.load_model(MODEL_PATH)
        with CLASS_PATH.open("r", encoding="utf-8") as f:
            class_names = json.load(f)
        app.logger.info("Loaded model with %d classes", len(class_names))
    else:
        app.logger.warning(
            "Model files not found. Run train_model.py first to enable predictions."
        )


def analyze_image(image):
    if model is None or not class_names:
        raise RuntimeError(
            "The trained AI model is not installed yet. Add a labeled dataset and run train_model.py."
        )

    image = image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    scores = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(scores))
    class_name = class_names[idx]
    confidence = float(scores[idx]) * 100.0

    details = TREATMENTS.get(class_name, {
        "name": class_name.replace("Tomato___", "").replace("_", " "),
        "description": "No reviewed recommendation is available for this model class.",
        "treatment": ["Consult an agricultural specialist to confirm the cause before treatment."],
        "prevention": ["Monitor the plant and maintain good crop hygiene."]
    })

    return {
        "prediction": details["name"],
        "class_name": class_name,
        "confidence": round(confidence, 2),
        "description": details["description"],
        "treatment": details["treatment"],
        "prevention": details["prevention"],
        "notice": (
            "AI predictions are preliminary and may be wrong. Similar symptoms can have "
            "different causes. Confirm uncertain results with an agricultural expert."
        )
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_ready": model is not None and bool(class_names),
        "class_count": len(class_names)
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Please select or capture a leaf image."}), 400

    upload = request.files["image"]
    if not upload.filename:
        return jsonify({"error": "No image was selected."}), 400

    try:
        image = Image.open(upload.stream)
        image.load()
        if image.width < 50 or image.height < 50:
            return jsonify({"error": "Please use a larger, clearer image."}), 400
        return jsonify(analyze_image(image))
    except UnidentifiedImageError:
        return jsonify({"error": "The selected file is not a valid image."}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        app.logger.exception("Image prediction failed")
        return jsonify({"error": "Prediction failed. Please try another image."}), 500


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "Image is too large. Maximum file size is 10 MB."}), 413


if __name__ == "__main__":
    load_model_files()
    # Local development only. Do not use Flask's debug server for public deployment.
    app.run(host="127.0.0.1", port=5000, debug=True)
