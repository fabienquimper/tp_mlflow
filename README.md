# Emotion Classifier — TP MLflow

Classifieur d'émotions (texte → `happy` / `sad` / `angry`) pour illustrer un cycle MLOps complet.

```
  train.py  ──▶  MLflow UI  ──▶  Model Registry  ──▶  API REST :5001
  (Data Scientist)  (tracking)     (@champion)         (predict.py / curl)
```

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install mlflow scikit-learn datasets fastapi uvicorn
```

---

## Les 3 modes de données

C'est le cœur du TP : observer l'impact du volume de données sur la qualité du modèle.

| Mode | Commande | Données | Accuracy attendue |
|------|----------|---------|-------------------|
| 1 — Minuscule | `python train.py --mode 1` | 12 phrases | ~0% — modèle inutilisable |
| 2 — Moyen | `python train.py --mode 2` | 105 phrases | ~57% — médiocre |
| 3 — Complet | `python train.py --mode 3` | ~1600 phrases (+ HuggingFace) | ~68% — acceptable |

> Lance les 3 puis compare les runs dans l'UI : **http://localhost:5000 → Training runs**

---

## Cycle complet

### 1. Lancer l'interface MLflow (terminal dédié)

```bash
mlflow ui --port 5000
# Ouvrir http://localhost:5000
```

### 2. Entraîner et enregistrer le modèle

```bash
python train.py --mode 2    # ou --mode 1 / --mode 3
```

Le script entraîne un pipeline `TF-IDF + LogisticRegression`, logue les métriques dans MLflow,
enregistre le modèle dans le **Model Registry** et le promeut sous l'alias `@champion`.

### 3. Tester les prédictions

```bash
# Mode direct — charge le modèle depuis le registry local
python predict.py
python predict.py --text "I feel amazing!"
```

### 4. Déployer l'API avec interface graphique (recommandé)

```bash
python app.py
```

Ouvre **http://localhost:5002/docs** — Swagger UI complet avec champs de saisie.

Ou en ligne de commande :
```bash
curl -X POST http://localhost:5002/predict \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Best day ever!", "I am furious!", "I miss you"]}'
```

Réponse :
```json
{
  "model_version": "v9 (@champion)",
  "predictions": [
    {"text": "Best day ever!", "emotion": "happy", "confidence": {"angry": 0.27, "happy": 0.44, "sad": 0.29}},
    {"text": "I am furious!",  "emotion": "angry", "confidence": {"angry": 0.59, "happy": 0.21, "sad": 0.20}},
    {"text": "I miss you",     "emotion": "sad",   "confidence": {"angry": 0.24, "happy": 0.33, "sad": 0.43}}
  ]
}
```

> `mlflow models serve` reste disponible sur le port 5001 (voir `predict.py --api`),
> mais son Swagger UI ne permet pas de saisir les données — préférer `app.py`.

---

## L'interface MLflow

L'UI **ne prédit pas** — elle observe et compare. Onglets utiles pour ce TP :

| Onglet | Ce qu'on y voit |
|--------|-----------------|
| **Training runs** | Tous les runs, leurs métriques et paramètres. Clique pour voir le détail. |
| **Models** | Les versions de `emotion-classifier` et l'alias `@champion` |
| **Evaluation** | Comparaison côte à côte des versions (avancé) |
| Observability / Traces / Prompts… | Fonctionnalités LLM — hors-sujet pour ce TP |

---

## Simulation vs production réelle

| | Ce TP | Production |
|-|-------|------------|
| Entraînement | `python train.py` à la main | Pipeline automatisé (Airflow, CI/CD) |
| Tracking | MLflow local (SQLite) | MLflow sur serveur partagé (PostgreSQL + S3) |
| Validation | Alias `@champion` mis à jour automatiquement | Validation humaine + tests avant promotion |
| Déploiement | `mlflow models serve` (1 processus) | Docker + Kubernetes (scalable) |
| URL | `http://localhost:5001` | `https://api.entreprise.com/predict` + auth |
| Monitoring | Aucun | Prometheus + Grafana (latence, dérive du modèle) |

> Ce qui **ne change pas** : le Model Registry, les aliases, la séparation entraînement / déploiement.

---

## Structure

```
├── MLproject        # Entry points mlflow run
├── python_env.yaml  # Dépendances
├── train.py         # Entraînement — options : --mode 1 / 2 / 3
├── predict.py       # Prédiction CLI — options : --text, --api, --port
└── app.py           # API FastAPI custom — Swagger UI sur :5002/docs
```

## Pour aller plus loin

- Changer `C=1.0` dans `train.py` et relancer — observer la nouvelle version dans le registry
- `mlflow run . -e train` — exécution reproductible via MLproject
- Remplacer TF-IDF par un modèle pré-entraîné (BERT) pour dépasser 90% d'accuracy
