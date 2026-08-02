import os
import pickle

import pandas as pd
from flask import Flask, jsonify, request

from data_input import get_sample_payload, get_x_test

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_file.p")
DATA_PATH = os.path.join(BASE_DIR, "..", "salary_data_cleanedv2.csv")


with open(MODEL_PATH, "rb") as pickled:
    MODEL = pickle.load(pickled)["model"]

X_TEST = get_x_test(data_path=DATA_PATH)


@app.route("/sample-row", methods=["GET"])
def sample_row():
    index = int(request.args.get("index", 1))
    if index < 0 or index >= len(X_TEST):
        return jsonify({"error": "index out of range"}), 400
    return jsonify(get_sample_payload(index=index, data_path=DATA_PATH))


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    features = payload.get("features")
    if not isinstance(features, dict):
        return jsonify({"error": "Send JSON like {'features': {...}}"}), 400

    row_df = pd.DataFrame([features]).reindex(columns=X_TEST.columns, fill_value=0)
    prediction = float(MODEL.predict(row_df)[0])
    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(debug=True)