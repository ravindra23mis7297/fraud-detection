"""
app.py
======
Flask backend for the food-delivery fraud detection system.

Instructions:
    1. Install dependencies:
           pip install flask flask-cors tensorflow pillow numpy pandas \
                       scikit-learn xgboost joblib
    2. Ensure these files exist in the same directory:
           image_fraud_model.h5
           behavior_model.pkl
           behavior_scaler.pkl
    3. Start the server:
           python app.py
    4. Open http://localhost:5000 in your browser.

Endpoint:
    POST /predict
        multipart/form-data:
            - image    : food image file
            - data     : JSON string of user/order features
        Response: JSON { fraud_score, risk_level, image_score, behavior_score }
"""

import os
import json
import uuid
import tempfile

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from utils import predict_fraud

# ── App Setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".")
CORS(app)   # Allow cross-origin requests from the HTML frontend

UPLOAD_FOLDER = tempfile.mkdtemp()   # Temporary directory for uploaded images
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif"}


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the HTML frontend."""
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept a food image + JSON user data and return a fraud assessment.

    Form fields:
        image  (file)   – the food photo submitted with the complaint
        data   (string) – JSON-encoded user/order feature dictionary
    """
    # ── Validate image ────────────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty image filename."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported image format."}), 400

    # Save to temp file
    ext       = file.filename.rsplit(".", 1)[1].lower()
    tmp_path  = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.{ext}")
    file.save(tmp_path)

    # ── Parse user/order data ─────────────────────────────────────────────────
    raw_data = request.form.get("data", "{}")
    try:
        user_data = json.loads(raw_data)
    except json.JSONDecodeError:
        os.remove(tmp_path)
        return jsonify({"error": "Invalid JSON in 'data' field."}), 400

    # ── Run prediction ────────────────────────────────────────────────────────
    try:
        result = predict_fraud(tmp_path, user_data)
    except Exception as exc:
        os.remove(tmp_path)
        return jsonify({"error": str(exc)}), 500
    finally:
        # Always clean up the temp image
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return jsonify(result), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
