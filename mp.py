import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


# CONFIG
DATA_FOLDER = "datasets"
MODEL_FOLDER = "saved_models/price"

os.makedirs(MODEL_FOLDER, exist_ok=True)

crops = ["rice", "jowar", "bajra", "groundnut", "urad", "moong", "maize"]

# SEASON FUNCTION
def get_season(month):
    if month in [6,7,8,9,10]:
        return "Kharif"
    elif month in [11,12,1,2,3]:
        return "Rabi"
    else:
        return "Summer"

print("\nSTARTING TRAINING...\n")

# CROP-SPECIFIC MODELS
crop_models = {}
crop_state_enc = {}
crop_season_enc = {}
crop_scores = {}
crop_crop_enc = {}

for crop in crops:
    print(f"\nProcessing {crop.upper()}...")

    file_path = os.path.join(DATA_FOLDER, f"{crop}.csv")

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    df = pd.read_csv(file_path)

    # Feature engineering
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month
    df["Season"] = df["Month"].apply(get_season)
    
    df.rename(columns={"Commodity": "Crop"}, inplace=True)
    # Encoders
    state_encoder = LabelEncoder()
    season_encoder = LabelEncoder()
    crop_encoder = LabelEncoder()

    df["State_enc"] = state_encoder.fit_transform(df["State"])
    df["Season_enc"] = season_encoder.fit_transform(df["Season"])
    df["Crop_enc"] = crop_encoder.fit_transform(df["Crop"])
    # Features
    X = df[["Crop_enc", "State_enc", "Season_enc", "Arrival"]]
    y = df["Price"]

    if len(df) > 100:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{crop.upper()} PERFORMANCE:")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.2f}")

    crop_models[crop] = model
    crop_state_enc[crop] = state_encoder
    crop_season_enc[crop] = season_encoder
    crop_crop_enc[crop] = crop_encoder
    crop_scores[crop] = r2

    print(f"Saved {crop} model")

# Save crop-specific artifacts
joblib.dump(crop_models, os.path.join(MODEL_FOLDER, "crop_models.pkl"))
joblib.dump(crop_state_enc, os.path.join(MODEL_FOLDER, "crop_state_encoders.pkl"))
joblib.dump(crop_season_enc, os.path.join(MODEL_FOLDER, "crop_season_encoders.pkl"))
joblib.dump(crop_scores, os.path.join(MODEL_FOLDER, "crop_scores.pkl"))
joblib.dump(crop_crop_enc, os.path.join(MODEL_FOLDER, "crop_crop_encoders.pkl"))

# =========================
# GLOBAL MODEL
print("\nTraining GLOBAL model...")

global_path = os.path.join(DATA_FOLDER, "global_dataset.csv")

if not os.path.exists(global_path):
    print("global_dataset.csv not found!")
    exit()

global_df = pd.read_csv(global_path)

# Rename
global_df.rename(columns={"Commodity": "Crop"}, inplace=True)

# Feature engineering
global_df["Date"] = pd.to_datetime(global_df["Date"])
global_df["Month"] = global_df["Date"].dt.month
global_df["Season"] = global_df["Month"].apply(get_season)

# Encoders
crop_encoder = LabelEncoder()
state_encoder = LabelEncoder()
season_encoder = LabelEncoder()

global_df["Crop_enc"] = crop_encoder.fit_transform(global_df["Crop"])
global_df["State_enc"] = state_encoder.fit_transform(global_df["State"])
global_df["Season_enc"] = season_encoder.fit_transform(global_df["Season"])

# Features
X_global = global_df[["Crop_enc", "State_enc", "Season_enc", "Arrival"]]
y_global = global_df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X_global, y_global, test_size=0.2, random_state=42
)

global_model = RandomForestRegressor(n_estimators=100, random_state=42)
global_model.fit(X_train, y_train)

y_pred = global_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nGLOBAL MODEL PERFORMANCE:")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")

# =========================
# SAVE GLOBAL ARTIFACTS
# =========================
joblib.dump(global_model, os.path.join(MODEL_FOLDER, "global_price_model.pkl"))
joblib.dump(crop_encoder, os.path.join(MODEL_FOLDER, "global_crop_encoder.pkl"))
joblib.dump(state_encoder, os.path.join(MODEL_FOLDER, "global_state_encoder.pkl"))
joblib.dump(season_encoder, os.path.join(MODEL_FOLDER, "global_season_encoder.pkl"))

print("\nALL PRICE MODELS TRAINED SUCCESSFULLY!")