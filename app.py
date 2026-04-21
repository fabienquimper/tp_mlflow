"""
API REST custom — enveloppe le modèle MLflow avec un schéma propre.

Avantages vs `mlflow models serve` :
  - Swagger UI fonctionnel avec champs de saisie (/docs)
  - Schéma de requête et réponse explicite (Pydantic)
  - Scores de confiance inclus dans la réponse
  - Prêt à être dockerisé / déployé sur un vrai serveur

Usage :
  python app.py
  Ouvrir http://localhost:5002/docs
"""
import mlflow.sklearn
import uvicorn
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from pydantic import BaseModel

# --- Schémas de l'API ---

class PredictRequest(BaseModel):
    texts: list[str]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"texts": ["I feel amazing!", "I am furious!", "I miss you so much"]}
            ]
        }
    }

class EmotionResult(BaseModel):
    text: str
    emotion: str
    confidence: dict[str, float]

class PredictResponse(BaseModel):
    model_version: str
    predictions: list[EmotionResult]

# --- Application ---

app = FastAPI(
    title="Emotion Classifier",
    description=(
        "Classifie un texte en `happy`, `sad` ou `angry`.\n\n"
        "Le modèle est chargé depuis le **MLflow Model Registry** (alias `@champion`).\n\n"
        "En production, cette API serait déployée dans un conteneur Docker "
        "derrière un reverse proxy avec authentification."
    ),
    version="1.0.0",
)

_model = None
_model_version = "unknown"

def get_model():
    global _model, _model_version
    if _model is None:
        client = MlflowClient()
        meta = client.get_registered_model("emotion-classifier")
        for v in meta.latest_versions:
            aliases = client.get_model_version(meta.name, v.version).aliases
            if "champion" in aliases:
                _model_version = f"v{v.version} (@champion)"
                break
        _model = mlflow.sklearn.load_model("models:/emotion-classifier@champion")
    return _model, _model_version

# --- Endpoints ---

@app.get("/health", tags=["Info"])
def health():
    """Vérifie que le serveur est disponible."""
    return {"status": "ok"}

@app.get("/model-info", tags=["Info"])
def model_info():
    """Retourne la version du modèle actuellement en service."""
    _, version = get_model()
    return {"model": "emotion-classifier", "version": version}

@app.post("/predict", response_model=PredictResponse, tags=["Prédiction"])
def predict(request: PredictRequest):
    """
    Classifie un ou plusieurs textes en émotions.

    - **texts** : liste de textes à analyser
    - Retourne l'émotion prédite + les scores de confiance pour chaque classe
    """
    if not request.texts:
        raise HTTPException(status_code=422, detail="La liste de textes est vide.")

    model, version = get_model()
    classes = model.classes_.tolist()
    probas = model.predict_proba(request.texts)

    results = []
    for text, proba_row in zip(request.texts, probas):
        confidence = {cls: round(float(p), 3) for cls, p in zip(classes, proba_row)}
        emotion = max(confidence, key=confidence.get)
        results.append(EmotionResult(text=text, emotion=emotion, confidence=confidence))

    return PredictResponse(model_version=version, predictions=results)

# --- Lancement ---

if __name__ == "__main__":
    print("\nDemarrage de l'API Emotion Classifier...")
    print("  Swagger UI : http://localhost:5002/docs")
    print("  Health     : http://localhost:5002/health\n")
    uvicorn.run(app, host="0.0.0.0", port=5002)
