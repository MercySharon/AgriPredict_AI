import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ==============================
# 1. LOAD DATA
# ==============================
df = pd.read_csv("datasets/crop_yield_withMonthlyCLimateData.csv")

# Drop leakage columns
df.drop(columns=["Production", "Annual_Rainfall"], inplace=True, errors="ignore")

# ==============================
# 2. ENCODING
# ==============================
le_crop = LabelEncoder()
le_state = LabelEncoder()
le_season = LabelEncoder()

df["Crop_enc"] = le_crop.fit_transform(df["Crop"])
df["State_enc"] = le_state.fit_transform(df["State"])
df["Season_enc"] = le_season.fit_transform(df["Season"])

# ==============================
# 3. SAVE META (IMPORTANT FOR BACKEND)
# ==============================
meta = {
    "fertilizer_mean": float(df["Fertilizer"].mean()),
    "pesticide_mean": float(df["Pesticide"].mean())
}

# ==============================
# 4. GLOBAL MODEL
# ==============================
FEATURES = [
    "Crop_enc",
    "State_enc",
    "Season_enc",
    "Area",
    "Temperature",
    "Humidity",
    "Rainfall",
    "Fertilizer",
    "Pesticide",
    "Crop_Year"
]

X = df[FEATURES]
y = df["Yield"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

global_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

global_model.fit(X_train, y_train)

# Evaluation
y_pred = global_model.predict(X_test)

print("\nGLOBAL MODEL")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# ==============================
# 5. CROP-SPECIFIC MODELS
# ==============================
crop_counts = df["Crop_enc"].value_counts()
valid_crop_labels = crop_counts[crop_counts >= 400].index

crop_models = {}
results = {}

for crop_label in valid_crop_labels:

    crop_name = le_crop.inverse_transform([crop_label])[0]
    crop_df = df[df["Crop_enc"] == crop_label]

    X_crop = crop_df[FEATURES]
    y_crop = np.log1p(crop_df["Yield"])  # log transform

    X_train, X_test, y_train, y_test = train_test_split(
        X_crop, y_crop, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    results[crop_name] = {"R2": r2, "MAE": mae}
    crop_models[crop_name] = model

print("\nCrop-wise Results:")
for k, v in results.items():
    print(k, v)

# ==============================
# 6. FILTER GOOD MODELS
# ==============================
good_crop_models = {}

for crop, metrics in results.items():
    if metrics["R2"] >= 0.5:
        good_crop_models[crop] = crop_models[crop]

print("\nUsing only good crop models:", len(good_crop_models))

# ==============================
# 7. SAVE EVERYTHING
# ==============================
os.makedirs("saved_models/yield", exist_ok=True)

# Models
joblib.dump(global_model, "saved_models/yield/global_yield.pkl")
joblib.dump(good_crop_models, "saved_models/yield/crop_models.pkl")

# Encoders
joblib.dump(le_crop, "saved_models/yield/crop_encoder.pkl")
joblib.dump(le_state, "saved_models/yield/state_encoder.pkl")
joblib.dump(le_season, "saved_models/yield/season_encoder.pkl")

# Feature order (VERY IMPORTANT)
joblib.dump(FEATURES, "saved_models/yield/feature_columns.pkl")

# Meta values (fertilizer/pesticide means)
joblib.dump(meta, "saved_models/yield/meta.pkl")

print("\n✅ Yield models + encoders + meta saved cleanly!")