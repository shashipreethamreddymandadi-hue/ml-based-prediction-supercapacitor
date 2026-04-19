import os
import time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
from models import run_all

app = Flask(__name__)
CORS(app)

# ✅ upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ✅ file reader (CSV + Excel)
def read_file(file, filename):
    filename = filename.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(file)

    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(file)

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

        # 🔥 CASE 1: Uploaded files
        if train_file and test_file:
            train_filename = str(int(time.time())) + "_" + train_file.filename
            test_filename = str(int(time.time())) + "_" + test_file.filename

            train_path = os.path.join(UPLOAD_FOLDER, train_filename)
            test_path = os.path.join(UPLOAD_FOLDER, test_filename)

            train_file.save(train_path)
            test_file.save(test_path)

        # 🔥 CASE 2: Dropdown files
        elif train_name and test_name:
            train_path = os.path.join(UPLOAD_FOLDER, train_name)
            test_path = os.path.join(UPLOAD_FOLDER, test_name)

        else:
            return jsonify({"error": "Upload OR select both files"}), 400

        # ✅ READ FILES (THIS WAS MISSING)
        train_df = read_file(open(train_path, "rb"), train_path)
        test_df = read_file(open(test_path, "rb"), test_path)

        # run ML
        result = run_all(train_df, test_df)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ list saved files
@app.route("/files", methods=["GET"])
def list_files():
    files = os.listdir(UPLOAD_FOLDER)
    return jsonify({"files": files})


if __name__ == "__main__":
    app.run(debug=True)