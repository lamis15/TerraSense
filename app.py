from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import json

app = Flask(__name__)
CORS(app)


# ---------- Chargement des modèles Agriculture ----------
# On supporte à la fois "saved_models" et "models" (au cas où)
BASE_DIR = os.path.dirname(__file__)
PREFERRED_DIR = os.path.join(BASE_DIR, "saved_models")
FALLBACK_DIR = os.path.join(BASE_DIR, "models")

MODELS_DIR = PREFERRED_DIR if os.path.isdir(PREFERRED_DIR) else FALLBACK_DIR

rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

try:
    bounds = joblib.load(os.path.join(MODELS_DIR, "bounds.pkl"))
except Exception:
    bounds = None


@app.route("/")
def home():
    """Renvoie la page web principale TerraSense."""
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.route("/agriculture")
def agriculture_page():
    """Page dédiée au module Agriculture (recommandation de cultures)."""
    with open("agriculture.html", "r", encoding="utf-8") as f:
        return f.read()


@app.route("/images-agricultures/<path:filename>")
def agriculture_images(filename: str):
    """Servir les images locales utilisées dans le guide agriculture."""
    images_dir = os.path.join(BASE_DIR, "images-agricultures")
    return send_from_directory(images_dir, filename)


# ---------- Chargement du modèle Air Quality (Maher) ----------
AIR_DIR = os.path.join(BASE_DIR, "modeles_maha")
rf_air_model = joblib.load(os.path.join(AIR_DIR, "random_forest_maha.pkl"))
air_scaler = joblib.load(os.path.join(AIR_DIR, "scaler_maha.pkl"))
with open(os.path.join(AIR_DIR, "config_maha.json"), "r", encoding="utf-8") as f:
    air_config = json.load(f)
AIR_FEATURES = air_config["selected_features"]
with open(os.path.join(AIR_DIR, "imputation_stats_maha.json"), "r", encoding="utf-8") as f:
    air_impute_stats = json.load(f)


@app.route("/air")
def air_page():
    """Page dédiée au module Qualité de l'air (Maher)."""
    with open("air.html", "r", encoding="utf-8") as f:
        return f.read()


@app.route("/predict_air", methods=["POST"])
def predict_air():
    """Endpoint de prédiction pour la qualité de l'air."""
    try:
        data = request.get_json()

        # Construire un DataFrame avec toutes les features attendues par le scaler
        scaler_features = list(getattr(air_scaler, "feature_names_in_", AIR_FEATURES))
        row_full = {}
        for feat in scaler_features:
            if feat in data:
                row_full[feat] = float(data[feat])
            else:
                # Utiliser la moyenne d'imputation si disponible (CO(GT), T, RH, AH, etc.)
                if feat in air_impute_stats:
                    row_full[feat] = float(air_impute_stats[feat]["mean"])
                else:
                    # Fallback neutre si jamais une feature inattendue apparaît
                    row_full[feat] = 0.0

        df_full = pd.DataFrame([row_full])[scaler_features]

        # Normalisation sur toutes les colonnes du scaler
        df_scaled_full = air_scaler.transform(df_full)
        df_scaled_full = pd.DataFrame(df_scaled_full, columns=scaler_features)

        # Garder uniquement les features utilisées par le modèle
        df_model = df_scaled_full[AIR_FEATURES]

        # Prédictions
        pred_class = int(rf_air_model.predict(df_model)[0])
        proba = rf_air_model.predict_proba(df_model)[0]

        classes_labels = {0: "Bas", 1: "Moyen", 2: "Élevé"}

        return jsonify(
            {
                "success": True,
                "class": pred_class,
                "label": classes_labels.get(pred_class, str(pred_class)),
                "proba": {
                    "bas": float(proba[0]),
                    "moyen": float(proba[1]),
                    "eleve": float(proba[2]),
                },
            }
        )
    except Exception as e:
        import traceback

        print("=== ERREUR /predict_air ===")
        traceback.print_exc()
        print("=== FIN ERREUR /predict_air ===")
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint de prédiction pour la recommandation de cultures."""
    try:
        data = request.get_json()

        # Création du DataFrame à partir des valeurs envoyées
        input_data = pd.DataFrame(
            {
                "N": [float(data["N"])],
                "P": [float(data["P"])],
                "K": [float(data["K"])],
                "temperature": [float(data["temperature"])],
                "humidity": [float(data["humidity"])],
                "ph": [float(data["ph"])],
                "rainfall": [float(data["rainfall"])],
            }
        )

        # Feature engineering : indice de fertilité du sol
        input_data["Soil_Fertility_Index"] = (
            input_data["N"] + input_data["P"] + input_data["K"]
        ) / 3

        # Clipping avec les bornes si disponible (optionnel)
        if bounds is not None:
            for col in input_data.columns:
                if col in bounds:
                    input_data[col] = input_data[col].clip(
                        bounds[col]["lower"], bounds[col]["upper"]
                    )

        # Mise à l'échelle : utiliser exactement les mêmes colonnes que lors du fit du scaler
        feature_cols = list(getattr(scaler, "feature_names_in_", input_data.columns))
        input_for_scaler = input_data[feature_cols]
        scaled_values = scaler.transform(input_for_scaler)
        input_model = input_data.copy()
        input_model[feature_cols] = scaled_values

        # Prédictions
        prediction = rf_model.predict(input_model)[0]
        probabilities = rf_model.predict_proba(input_model)[0]

        crop = label_encoder.inverse_transform([prediction])[0]
        confidence = float(probabilities.max() * 100)

        return jsonify({
            "success": True,
            "crop": crop,
            "confidence": confidence,
            "soilFertility": float(input_data["Soil_Fertility_Index"].values[0]),
        })

    except Exception as e:
        # Log détaillé côté serveur pour le debug
        import traceback
        print("=== ERREUR /predict ===")
        traceback.print_exc()
        print("=== FIN ERREUR /predict ===")
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    # Pour le développement local
    app.run(debug=True, host="0.0.0.0", port=5000)


