from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
import pandas as pd
import joblib
import numpy as np
import logging
from typing import List, Dict, Any
import os
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------
# Cycle de vie : chargement du dataset, du modèle, des features
# --------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Démarrage de l'application...")

    required_files = ["donnemald3.xlsx", "model_final.pkl", "selected_cols.pkl"]
    for file in required_files:
        if not os.path.exists(file):
            logger.error(f"Fichier requis manquant: {file}")
            raise FileNotFoundError(f"Fichier requis manquant: {file}")

    try:
        app.state.df = pd.read_excel("donnemald3.xlsx")
        logger.info(f"Dataset chargé avec {len(app.state.df)} lignes")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du dataset: {e}")
        raise

    try:
        app.state.model = joblib.load("model_final.pkl")
        logger.info("Modèle ML chargé avec succès")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        raise

    try:
        app.state.selected_features = joblib.load("selected_cols.pkl")
        logger.info(
            f"Colonnes sélectionnées chargées: {len(app.state.selected_features)} features"
        )
    except Exception as e:
        logger.error(f"Erreur lors du chargement des colonnes: {e}")
        raise

    yield

    logger.info("Arrêt de l'application...")
    app.state.df = None
    app.state.model = None
    app.state.selected_features = None


# --------------------------------------------------------
# Création FastAPI + CORS
# --------------------------------------------------------
app = FastAPI(
    title="API de Prédiction Étudiante",
    description="API pour prédire la performance des étudiants et proposer des recommandations",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------
# Helpers : post-traitement binaire + recommandations
# --------------------------------------------------------
def proba_to_binary_output(
    proba: np.ndarray,
    row: pd.Series,
) -> Dict[str, Any]:
    """
    proba : array de shape (1, 2) avec [P(classe0), P(classe1)]
    On suppose 0 = non admis, 1 = admis.
    Retourne prediction 0/1 + proba en pourcentage + recommandations.
    """
    p_non_admis_raw = float(proba[0, 0])
    p_admis_raw = float(proba[0, 1])

    # Arrondi en pourcentage
    p_non_admis = round(p_non_admis_raw * 100, 2)
    p_admis = round(p_admis_raw * 100, 2)

    if p_admis_raw >= p_non_admis_raw:
        prediction = 1
        label = "Admis"
        confidence = p_admis
    else:
        prediction = 0
        label = "Non admis"
        confidence = p_non_admis

    # Générer les recommandations
    recos = build_recommendations(row, prediction, p_admis_raw)

    return {
        "prediction": prediction,
        "prediction_label": label,
        "probabilities": {
            "admis": p_admis,
            "non_admis": p_non_admis,
        },
        "confidence": confidence,
        "recommendations": recos,
    }

def get_float(row: pd.Series, col: str):
    v = row.get(col)
    return float(v) if pd.notna(v) else None


def build_recommendations(row: pd.Series, prediction: int, proba: float) -> List[str]:
    recos: List[str] = []

    # -----------------------------
    # 1) Niveau de risque dynamique
    # -----------------------------
    if proba >= 0.75:
        niveau = "faible"
    elif proba >= 0.5:
        niveau = "moyen"
    else:
        niveau = "élevé"

    # -----------------------------
    # 2) Message global
    # -----------------------------
    messages = {
        "élevé": "🔴 Risque élevé d’échec. Actions prioritaires recommandées :",
        "moyen": "🟠 Situation à améliorer. Ajustements nécessaires :",
        "faible": "🟢 Bonne progression. Conseils pour maintenir le niveau :",
    }

    recos.append(messages[niveau])
    # -----------------------------
    # 3) Définition des règles
    # -----------------------------
    rules = [
        {
            "col": "imp2",
            "condition": lambda v: v == 0,
            "messages": {
                "élevé": "🔴 Vérification quotidienne des devoirs par les parents.",
                "moyen": "🟠 Vérifier plus régulièrement les devoirs.",
                "faible": "🟢 Maintenir le suivi des devoirs.",
            },
            "seuil": 0.7,
        },
        {
            "col": "imp3",
            "condition": lambda v: v == 0,
            "messages": {
                "élevé": "🔴 Renforcer fortement la supervision des devoirs.",
                "moyen": "🟠 Améliorer l’accompagnement à la maison.",
                "faible": "🟢 Continuer l’encadrement actuel.",
            },
            "seuil": 0.65,
        },
        {
            "col": "imp5",
            "condition": lambda v: v == 0,
            "messages": {
                "élevé": "🔴 Clarifier urgemment les objectifs éducatifs.",
                "moyen": "🟠 Discuter des objectifs scolaires avec l’élève.",
                "faible": "🟢 Maintenir les échanges éducatifs.",
            },
            "seuil": 0.6,
        },
        {
            "col": "imp6",
            "condition": lambda v: v == 0,
            "messages": {
                "élevé": "🔴 Augmenter fortement les activités de lecture.",
                "moyen": "🟠 Encourager davantage la lecture.",
                "faible": "🟢 Maintenir les habitudes de lecture.",
            },
            "seuil": 0.6,
        },
        {
            "col": "imp10",
            "condition": lambda v: v == 0,
            "messages": {
                "élevé": "🔴 Structurer strictement le temps de travail.",
                "moyen": "🟠 Mieux organiser le temps d’étude.",
                "faible": "🟢 Continuer l’organisation actuelle.",
            },
            "seuil": 0.65,
        },
    ]

    # -----------------------------
    # 4) Application des règles
    # -----------------------------
    for rule in rules:
        v = get_float(row, rule["col"])

        if v is None:
            continue

        if rule["condition"](v) and proba < rule["seuil"]:
            recos.append(rule["messages"][niveau])

    # -----------------------------
    # 5) Variables spécifiques
    # -----------------------------
    rang = get_float(row, "Ran_TbB")
    if rang is not None and rang > 15 and niveau != "faible":
        recos.append("Réorganiser la position en classe et renforcer le suivi.")

    cs = get_float(row, "cour_supl")
    if cs is not None:
        if cs == 0 and proba < 0.7:
            recos.append("Encourager fortement les cours de soutien.")
        elif cs == 1 and niveau == "faible":
            recos.append("Continuer à exploiter les cours de soutien.")

    # -----------------------------
    # 6) Variables contextuelles
    # -----------------------------
    if str(row.get("mere_niv_ac", "")).upper() in ["0", "AUCUN", "PRIMAIRE"]:
        if niveau != "faible":
            recos.append("Adapter la communication avec la famille.")

    if "public" in str(row.get("etab_prim_stat", "")).lower():
        if niveau == "élevé":
            recos.append("Mobiliser les ressources scolaires disponibles.")

    # -----------------------------
    # 7) Limiter les recommandations
    # -----------------------------
    max_recos = {
        "élevé": 6,
        "moyen": 4,
        "faible": 3,
    }

    recos = recos[: max_recos[niveau]]

    return recos_uniques

# --------------------------------------------------------
# Modèle Pydantic dynamique (si tu veux l’utiliser plus tard)
# --------------------------------------------------------
def create_student_model(features: List[str]):
    """Crée dynamiquement un modèle Pydantic basé sur les features"""
    fields = {feat: (float, ...) for feat in features}
    return create_model("StudentData", **fields)


StudentData = None

# --------------------------------------------------------
# Endpoint de prédiction
# --------------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "OK", "service": "Prediction API", "version": "1.0.0"}

@app.get("/docs")
async def docs_redirect():
    return {"message": "API Prediction - Documentation", "endpoints": ["/health", "/predict"]}


@app.post("/predict")
async def predict(data: Dict[str, Any]):
    """
    Endpoint de prédiction binaire (0 = non admis, 1 = admis) + proba en % + recommandations.
    """
    try:
        if not hasattr(app.state, "model") or app.state.model is None:
            raise HTTPException(status_code=503, detail="Modèle non chargé")

        df_input = pd.DataFrame([data])

        # Vérifier les colonnes manquantes
        missing_cols = [
            col for col in app.state.selected_features if col not in df_input.columns
        ]
        if missing_cols:
            logger.warning(f"Colonnes manquantes: {missing_cols}")
            raise HTTPException(
                status_code=400,
                detail=f"Colonnes manquantes: {missing_cols}",
            )

        # Garder uniquement les features utiles
        df_input = df_input[app.state.selected_features]

        if df_input.isna().any().any():
            logger.warning("Données d'entrée contiennent des valeurs NaN")
            raise HTTPException(
                status_code=400,
                detail="Les données contiennent des valeurs manquantes",
            )

        # Prédiction brute
        prediction_array = app.state.model.predict(df_input)
        prediction = int(prediction_array[0])

        # Probabilités si dispo
        if hasattr(app.state.model, "predict_proba"):
            proba_array = app.state.model.predict_proba(df_input)
            # post-traitement binaire + recommandations
            row = df_input.iloc[0]
            resultat = proba_to_binary_output(proba_array, row)
        else:
            # Cas sans predict_proba : on fabrique une proba simple
            row = df_input.iloc[0]
            if prediction == 1:
                p_admis = 70.0
                p_non = 30.0
            else:
                p_admis = 30.0
                p_non = 70.0
            recos = build_recommendations(row, prediction)
            resultat = {
                "prediction": prediction,
                "prediction_label": "Admis" if prediction == 1 else "Non admis",
                "probabilities": {
                    "admis": p_admis,
                    "non_admis": p_non,
                },
                "confidence": p_admis if prediction == 1 else p_non,
                "recommendations": recos,
            }

        logger.info(
            f"Prédiction réussie: prediction={resultat['prediction']} label={resultat['prediction_label']}"
        )

        return {"success": True, **resultat}

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Erreur de validation des données: {e}")
        raise HTTPException(status_code=400, detail=f"Erreur de format: {str(e)}")
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


# --------------------------------------------------------
# Gestion 404
# --------------------------------------------------------
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "Endpoint non trouvé", "detail": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
