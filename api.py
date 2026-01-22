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


# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gestion du cycle de vie de l'application
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Démarrage de l'application...")
    
    # Vérifier l'existence des fichiers
    required_files = ['donnemald3.xlsx', 'model_final.pkl', 'selected_cols.pkl']
    for file in required_files:
        if not os.path.exists(file):
            logger.error(f"Fichier requis manquant: {file}")
            raise FileNotFoundError(f"Fichier requis manquant: {file}")
    
    # Charger les ressources
    try:
        app.state.df = pd.read_excel('donnemald3.xlsx')
        logger.info(f"Dataset chargé avec {len(app.state.df)} lignes")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du dataset: {e}")
        raise
    
    try:
        app.state.model = joblib.load('model_final.pkl')
        logger.info("Modèle ML chargé avec succès")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        raise
    
    try:
        app.state.selected_features = joblib.load('selected_cols.pkl')
        logger.info(f"Colonnes sélectionnées chargées: {len(app.state.selected_features)} features")
    except Exception as e:
        logger.error(f"Erreur lors du chargement des colonnes: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Arrêt de l'application...")
    app.state.df = None
    app.state.model = None
    app.state.selected_features = None

# Création de l'application FastAPI
app = FastAPI(
    title="API de Prédiction Étudiante",
    description="API pour prédire la performance des étudiants",
    version="1.0.0",
    lifespan=lifespan
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Modèles Pydantic dynamiques
# -------------------------
# Création dynamique du modèle pour la prédiction
def create_student_model(features: List[str]):
    """Crée dynamiquement un modèle Pydantic basé sur les features"""
    fields = {feat: (float, ...) for feat in features}
    return create_model("StudentData", **fields)

# Créer le modèle après le chargement des features
StudentData = None

@app.post("/predict")
async def predict(data: Dict[str, Any]):
    """
    Endpoint de prédiction
    
    Args:
        data: Dictionnaire contenant les features de l'étudiant
    
    Returns:
        Prédiction et probabilités
    """
    try:
        # Vérifier que le modèle est chargé
        if not hasattr(app.state, 'model') or app.state.model is None:
            raise HTTPException(status_code=503, detail="Modèle non chargé")
        
        # Convertir les données en DataFrame
        df_input = pd.DataFrame([data])
        
        # Vérifier les colonnes manquantes
        missing_cols = [col for col in app.state.selected_features if col not in df_input.columns]
        if missing_cols:
            logger.warning(f"Colonnes manquantes: {missing_cols}")
            raise HTTPException(
                status_code=400, 
                detail=f"Colonnes manquantes: {missing_cols}"
            )
        
        # Sélectionner uniquement les features nécessaires
        df_input = df_input[app.state.selected_features]
        
        # Vérifier les valeurs NaN
        if df_input.isna().any().any():
            logger.warning("Données d'entrée contiennent des valeurs NaN")
            raise HTTPException(
                status_code=400, 
                detail="Les données contiennent des valeurs manquantes"
            )
        
        # Faire la prédiction
        prediction = app.state.model.predict(df_input)[0]
        
        # Obtenir les probabilités si disponible
        probabilities = None
        if hasattr(app.state.model, "predict_proba"):
            probabilities = app.state.model.predict_proba(df_input).tolist()
        
        logger.info(f"Prédiction réussie: {prediction}")
        
        return {
            "prediction": int(prediction),
            "prediction_label": f"Classe {int(prediction)}",
            "probabilities": probabilities,
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Erreur de validation des données: {e}")
        raise HTTPException(status_code=400, detail=f"Erreur de format: {str(e)}")
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")


# -------------------------
# Exception handlers
# -------------------------
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "Endpoint non trouvé", "detail": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",  # Écoute sur toutes les interfaces
        port=8000,        # Port par défaut
        reload=True       # Redémarrage automatique en développement
    )