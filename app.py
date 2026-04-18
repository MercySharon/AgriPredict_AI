"""
AgriPredict AI — Flask Backend
Supports:
- Yield prediction (cy.py models)
- Price prediction (mp.py models)
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import joblib
import os
import requests
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# =========================
# PATHS
# =========================
YIELD_MODEL_FOLDER = os.path.join("saved_models", "yield")
PRICE_MODEL_FOLDER = os.path.join("saved_models", "price")

PRICE_CROPS = ["rice", "jowar", "bajra", "groundnut", "urad", "moong", "maize"]

# =========================
# GLOBAL STORAGE
# =========================
yield_model = None
yield_crop_enc = None
yield_state_enc = None
yield_season_enc = None

price_global_model = None
price_global_crop_enc = None
price_global_state_enc = None
price_global_season_enc = None

price_crop_models = {}
price_crop_state_enc = {}
price_crop_season_enc = {}
price_crop_crop_enc = {}   
price_crop_scores = {}

meta = {
    "yield_crops": [],
    "yield_states": [],
    "yield_seasons": [],
    "price_crops": [],
    "price_states": [],
    "price_seasons": ["Kharif", "Rabi", "Summer"],
}

# =========================
# LOAD MODELS
# =========================
def load_models():
    global yield_model, yield_crop_enc, yield_state_enc, yield_season_enc
    global price_global_model, price_global_crop_enc
    global price_global_state_enc, price_global_season_enc
    global price_crop_crop_enc  
    global meta

    print("\nLoading models...\n")

    # ================= YIELD =================
    try:
        yield_model = joblib.load(os.path.join(YIELD_MODEL_FOLDER, "global_yield.pkl"))
        yield_crop_enc = joblib.load(os.path.join(YIELD_MODEL_FOLDER, "crop_encoder.pkl"))
        yield_state_enc = joblib.load(os.path.join(YIELD_MODEL_FOLDER, "state_encoder.pkl"))
        yield_season_enc = joblib.load(os.path.join(YIELD_MODEL_FOLDER, "season_encoder.pkl"))

        meta["yield_crops"] = yield_crop_enc.classes_.tolist()
        meta["yield_states"] = yield_state_enc.classes_.tolist()
        meta["yield_seasons"] = yield_season_enc.classes_.tolist()

        print("Yield model loaded")
    except Exception as e:
        print("Yield model load failed:", e)

    # ================= PRICE CROP MODELS =================
    try:
        crop_models = joblib.load(os.path.join(PRICE_MODEL_FOLDER, "crop_models.pkl"))
        crop_state_enc = joblib.load(os.path.join(PRICE_MODEL_FOLDER, "crop_state_encoders.pkl"))
        crop_season_enc = joblib.load(os.path.join(PRICE_MODEL_FOLDER, "crop_season_encoders.pkl"))
        crop_scores_loaded = joblib.load(os.path.join(PRICE_MODEL_FOLDER, "crop_scores.pkl"))
        price_crop_crop_enc = joblib.load(os.path.join(PRICE_MODEL_FOLDER, "crop_crop_encoders.pkl"))

        for crop in crop_models:
            price_crop_models[crop] = crop_models[crop]
            price_crop_state_enc[crop] = crop_state_enc[crop]
            price_crop_season_enc[crop] = crop_season_enc[crop]
            price_crop_scores[crop] = crop_scores_loaded.get(crop, 0)

        print("Crop-specific price models loaded")
    except Exception as e:
        print("Crop-specific price models not found:", e)

    # ================= GLOBAL PRICE =================
    try:
        price_global_model = joblib.load(os.path.join(PRICE_MODEL_FOLDER, "global_price_model.pkl"))
        price_global_crop_enc = joblib.load(os.path.join(PRICE_MODEL_FOLDER, "global_crop_encoder.pkl"))
        price_global_state_enc = joblib.load(os.path.join(PRICE_MODEL_FOLDER, "global_state_encoder.pkl"))
        price_global_season_enc = joblib.load(os.path.join(PRICE_MODEL_FOLDER, "global_season_encoder.pkl"))

        meta["price_crops"] = price_global_crop_enc.classes_.tolist()
        meta["price_states"] = price_global_state_enc.classes_.tolist()

        print("Global price model loaded")
    except Exception as e:
        print("Global price model not found:", e)

    print("\nModel loading complete!\n")


# =========================
# HELPERS
# =========================
def safe_encode(encoder, value):
    classes = list(encoder.classes_)
    if value in classes:
        return int(encoder.transform([value])[0])

    lc_map = {c.lower(): c for c in classes}
    if value.lower() in lc_map:
        return int(encoder.transform([lc_map[value.lower()]])[0])

    return 0


STATE_COORDS = {
    "andhra pradesh": (15.9129, 79.7400),
    "assam": (26.2006, 92.9376),
    "bihar": (25.0961, 85.3131),
    "chhattisgarh": (21.2787, 81.8661),
    "gujarat": (22.2587, 71.1924),
    "haryana": (29.0588, 76.0856),
    "karnataka": (15.3173, 75.7139),
    "kerala": (10.8505, 76.2711),
    "madhya pradesh": (22.9734, 78.6569),
    "maharashtra": (19.7515, 75.7139),
    "odisha": (20.9517, 85.0985),
    "punjab": (31.1471, 75.3412),
    "rajasthan": (27.0238, 74.2179),
    "tamil nadu": (11.1271, 78.6569),
    "telangana": (18.1124, 79.0193),
    "uttar pradesh": (26.8467, 80.9462),
    "west bengal": (22.9868, 87.8550),
    "delhi": (28.7041, 77.1025),
    "jammu and kashmir": (33.7782, 76.5762),
    "puducherry": (11.9416, 79.8083)
}

def fetch_weather(state_name):
    key = state_name.lower().strip()
    lat, lon = STATE_COORDS.get(key, (20.5937, 78.9629))

    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m"
            f"&daily=precipitation_sum&forecast_days=7&timezone=auto"
        )

        resp = requests.get(url, timeout=6)
        resp.raise_for_status()
        d = resp.json()

        temp = d["current"]["temperature_2m"]
        humidity = d["current"]["relative_humidity_2m"]
        rainfall = sum(d["daily"]["precipitation_sum"])

        return {
            "temperature": round(temp, 1),
            "humidity": round(humidity, 1),
            "rainfall": round(rainfall, 2)
        }

    except:
        return {"temperature": 28.0, "humidity": 75.0, "rainfall": 85.0}


def match_price_crop(crop_name):
    key = crop_name.lower().replace(" ", "")
    for pc in PRICE_CROPS:
        if pc in key or key in pc:
            return pc
    return None


# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return "AgriPredict AI Backend Running"


@app.route("/api/health")
def health():
    return jsonify({
        "yield_loaded": yield_model is not None,
        "price_global_loaded": price_global_model is not None,
        "price_crop_models": list(price_crop_models.keys())
    })


@app.route("/api/meta")
def get_meta():
    return jsonify(meta)


# =========================
# YIELD PREDICTION
# =========================
@app.route("/api/predict_yield", methods=["POST"])
def predict_yield():
    if yield_model is None:
        return jsonify({"error": "Yield model not loaded"}), 503

    data = request.json or {}

    crop = data.get("crop", "")
    state = data.get("state", "")
    season = data.get("season", "")
    area = float(data.get("area", 1000))

    weather = fetch_weather(state)
    temperature = weather["temperature"]
    humidity = weather["humidity"]
    rainfall = weather["rainfall"]

    crop_enc = safe_encode(yield_crop_enc, crop)
    state_enc = safe_encode(yield_state_enc, state)
    season_enc = safe_encode(yield_season_enc, season)

    fertilizer = float(data.get("fertilizer", 100))
    pesticide = float(data.get("pesticide", 10))
    crop_year = int(data.get("crop_year", datetime.now().year))

    features = np.array([[crop_enc, state_enc, season_enc, area,
                          temperature, humidity, rainfall,
                          fertilizer, pesticide, crop_year]])

    prediction = float(yield_model.predict(features)[0])

    return jsonify({
        "predicted_yield": round(prediction, 4),
        "weather": weather
    })


# =========================
# PRICE PREDICTION
# =========================
@app.route("/api/predict_price", methods=["POST"])
def predict_price():
    if price_global_model is None and not price_crop_models:
        return jsonify({"error": "No price models loaded"}), 503

    data = request.json or {}

    crop = data.get("crop", "")
    state = data.get("state", "")
    season = data.get("season", "")

    arrival = float(data.get("yield_value", 1)) / 1000

    matched_crop = match_price_crop(crop)

    use_specific = (
        matched_crop is not None and   
        matched_crop in price_crop_models and
        price_crop_scores.get(matched_crop, 0) > 0.5
    )

    if use_specific:
        state_enc = price_crop_state_enc[matched_crop]
        season_enc = price_crop_season_enc[matched_crop]
        crop_enc = price_crop_crop_enc[matched_crop]

        state_val = safe_encode(state_enc, state)
        season_val = safe_encode(season_enc, season)
        crop_val = safe_encode(crop_enc, crop)

        features = np.array([[crop_val, state_val, season_val, arrival]])
        model = price_crop_models[matched_crop]
        model_used = f"{matched_crop}-specific"

    else:
        crop_val = safe_encode(price_global_crop_enc, crop)
        state_val = safe_encode(price_global_state_enc, state)
        season_val = safe_encode(price_global_season_enc, season)

        features = np.array([[crop_val, state_val, season_val, arrival]])
        model = price_global_model
        model_used = "global"

    prediction = float(model.predict(features)[0])

    return jsonify({
        "predicted_price": round(prediction, 2),
        "model_used": model_used,
        "arrival_proxy": arrival
    })


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    load_models()
    app.run(debug=True, port=5000)