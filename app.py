import os
import gc
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
from models import run_all

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ✅ Optimized file reader (LOW MEMORY)
def read_file(file, filename):
    filename = filename.lower()

    print("📂 Reading file:", filename)

    if filename.endswith(".csv"):
        # CSV is lighter
        return pd.read_csv(file)

    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        # limit rows at read time (IMPORTANT)
        return pd.read_excel(file, engine="openpyxl", nrows=300)

    else:
        raise ValueError("Unsupported file format")


# ✅ serve frontend
@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("🚀 Request received")

        train_file = request.files.get("train")
        test_file = request.files.get("test")

        train_name = request.form.get("train_name")
        test_name = request.form.get("test_name")

        # 🔥 CASE 1: Uploaded files
        if train_file and test_file:
            train_df = read_file(train_file, train_file.filename)
            test_df = read_file(test_file, test_file.filename)

        # 🔥 CASE 2: Saved files
        elif train_name and test_name:
            train_path = os.path.join(UPLOAD_FOLDER, train_name)
            test_path = os.path.join(UPLOAD_FOLDER, test_name)

            train_df = read_file(open(train_path, "rb"), train_path)
            test_df = read_file(open(test_path, "rb"), test_path)

        else:
            return jsonify({"error": "Upload OR select both files"}), 400

        # 🔥 EXTRA SAFETY LIMIT (in case CSV)
        train_df = train_df.head(300)
        test_df = test_df.head(300)

        print("📊 Data loaded")
        print("Train shape:", train_df.shape)
        print("Test shape:", test_df.shape)

        # ✅ Run ML
        print("⚙️ Running model...")
        result = run_all(train_df, test_df)

        print("✅ Model completed")

        # ✅ FORCE MEMORY CLEANUP
        del train_df
        del test_df
        gc.collect()

        return jsonify(result)

    except Exception as e:
        print("❌ ERROR:", str(e))

        # Cleanup even on failure
        gc.collect()

        return jsonify({"error": str(e)}), 500


# ✅ list saved files
@app.route("/files", methods=["GET"])
def list_files():
    files = os.listdir(UPLOAD_FOLDER)
    return jsonify({"files": files})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)