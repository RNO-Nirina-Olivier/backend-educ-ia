from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import List, Dict
import numpy as np
from pydantic import create_model

app = FastAPI()

# -------------------------
# Charger dataset
# -------------------------
df = pd.read_excel('donnemald3.xlsx')
# -------------------------
# Charger modèle + colonnes
# -------------------------
model = joblib.load('model_final.pkl')
selected_features = joblib.load('selected_cols.pkl')

# Création dynamique du modèle Pydantic
class StudentData(BaseModel):
    pass

# Ajout dynamique des champs

fields = {feat: (float, ...) for feat in selected_features}
StudentData = create_model("StudentData", **fields)

# -------------------------
# Route: prediction
# -------------------------
@app.post("/predict")
def predict(data: StudentData):
    df_input = pd.DataFrame([data.dict()])

    missing_cols = [c for c in selected_features if c not in df_input.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")

    df_input = df_input[selected_features]
    pred = model.predict(df_input)[0]
    probabilite = model.predict_proba(df_input)
    return {"prediction": int(pred),
            "probabilite": probabilite}

# -------------------------
# Route: liste des étudiants
# -------------------------
@app.get("/students")
async def get_students(limit: int = 100):
    students = (
        df.head(limit)
        .replace({np.nan: None})   # conversion NaN → null
        .to_dict(orient='records')
    )
    return {"students": students}


# -------------------------
# Route: statistiques
# -------------------------
@app.get("/statistics")
async def get_statistics():

    # Calcul du taux de réussite
    if df['réussite'].dtype == 'O':  # texte
        success_rate = (df['réussite'].str.lower() == 'admis').mean()
    else:
        success_rate = (df['réussite'] == 1).mean()

    # Vérifier si note_math existe
    if 'note_math' in df.columns:
        mean_math = float(df['note_math'].mean())
    else:
        mean_math = None  # ou tu peux mettre une autre colonne

    stats = {
        "mean_math": mean_math,
        "success_rate": float(success_rate),
        "total_students": int(len(df))
    }
    return stats
