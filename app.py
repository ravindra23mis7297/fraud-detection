import os
import json
import uuid
import tempfile
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from utils import predict_fraud

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif"}


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty image filename."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported image format."}), 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    tmp_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.{ext}")
    file.save(tmp_path)

    raw_data = request.form.get("data", "{}")
    try:
        user_data = json.loads(raw_data)
    except json.JSONDecodeError:
        os.remove(tmp_path)
        return jsonify({"error": "Invalid JSON in 'data' field."}), 400

    try:
        result = predict_fraud(tmp_path, user_data)
    except Exception as exc:
        os.remove(tmp_path)
        return jsonify({"error": str(exc)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return jsonify(result), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)