import os
import time
import gc
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
from models import run_all

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ✅ file reader (optimized)
def read_file(file, filename):
    filename = filename.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(file)

    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(file, engine="openpyxl")

    else:
        raise ValueError("Unsupported file format")


# serve frontend
@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        train_file = request.files.get("train")
        test_file = request.files.get("test")

        train_name = request.form.get("train_name")
        test_name = request.form.get("test_name")

        # 🔥 CASE 1: Uploaded files (NO SAVE → DIRECT READ)
        if train_file and test_file:
            train_df = read_file(train_file, train_file.filename).head(1000)
            test_df = read_file(test_file, test_file.filename).head(1000)

        # 🔥 CASE 2: Dropdown files (saved earlier)
        elif train_name and test_name:
            train_path = os.path.join(UPLOAD_FOLDER, train_name)
            test_path = os.path.join(UPLOAD_FOLDER, test_name)

            with open(train_path, "rb") as f:
                train_df = read_file(f, train_path).head(1000)

            with open(test_path, "rb") as f:
                test_df = read_file(f, test_path).head(1000)

        else:
            return jsonify({"error": "Upload OR select both files"}), 400

        # ✅ run ML
        result = run_all(train_df, test_df)

        # ✅ free memory (VERY IMPORTANT)
        del train_df
        del test_df
        gc.collect()

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ list saved files
@app.route("/files", methods=["GET"])
def list_files():
    files = os.listdir(UPLOAD_FOLDER)
    return jsonify({"files": files})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)