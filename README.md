# Emotion Classifier — TP MLflow

Classifieur d'émotions (texte → `happy` / `sad` / `angry`) pour illustrer un cycle MLOps complet,
du poste local jusqu'au déploiement en production.

```
  train.py  ──▶  MLflow UI  ──▶  Model Registry  ──▶  app.py (API REST)
  (entraîne)     (tracking)      (@champion)           (Swagger :5002/docs)
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

Observer l'impact du volume de données sur la qualité du modèle :


| Mode           | Commande                   | Données                      | Accuracy attendue           |
| -------------- | -------------------------- | ----------------------------- | --------------------------- |
| 1 — Minuscule | `python train.py --mode 1` | 12 phrases                    | ~0% — modèle inutilisable |
| 2 — Moyen     | `python train.py --mode 2` | 105 phrases                   | ~57% — médiocre           |
| 3 — Complet   | `python train.py --mode 3` | ~1600 phrases (+ HuggingFace) | ~68% — acceptable          |

> Lance les 3, puis compare les runs dans l'UI : **http://localhost:5000 → Training runs**

---

## Cycle complet (local)

### 1. Lancer l'interface MLflow

Terminal dédié, laisser tourner :

```bash
mlflow ui --port 5000
# Ouvrir http://localhost:5000
```

### 2. Entraîner et enregistrer le modèle

```bash
python train.py --mode 2    # ou --mode 1 / --mode 3
```

Entraîne `TF-IDF + LogisticRegression`, logue les métriques, enregistre le modèle
dans le **Model Registry** et le promeut sous l'alias `@champion`.

### 3. Tester les prédictions en ligne de commande

```bash
python predict.py
python predict.py --text "I feel amazing!"
```

### 4. Déployer l'API avec Swagger UI

```bash
python app.py
# Ouvrir http://localhost:5002/docs → Try it out → Execute
```

Ou via curl :

```bash
curl -X POST http://localhost:5002/predict \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Best day ever!", "I am furious!", "I miss you"]}'
```

Réponse avec scores de confiance :

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

---

## L'interface MLflow

L'UI **ne prédit pas** — elle observe et compare. Onglets utiles pour ce TP :


| Onglet                             | Ce qu'on y voit                                                         |
| ---------------------------------- | ----------------------------------------------------------------------- |
| **Training runs**                  | Tous les runs, métriques et paramètres — clique pour voir le détail |
| **Models**                         | Versions de`emotion-classifier` et alias `@champion`                    |
| **Evaluation**                     | Comparaison côte à côte des versions (avancé)                       |
| Observability / Traces / Prompts… | Fonctionnalités LLM — hors-sujet pour ce TP                           |

---

## Local vs Production

Ce TP tourne entièrement en local, mais l'architecture est déjà celle de la production.
**Le code ne change pas.** Seul l'endroit où les composants s'exécutent change.


|                 | Ce TP (local)                 | Production                                  |
| --------------- | ----------------------------- | ------------------------------------------- |
| Tracking MLflow | SQLite + fichiers locaux      | PostgreSQL + S3/MinIO sur serveur partagé  |
| Model Registry  | Accessible à vous seul       | Accessible à toute l'équipe               |
| API             | `http://localhost:5002`       | `https://api.entreprise.com/predict` + auth |
| Déploiement    | `python app.py` (1 processus) | Conteneur Docker sur Kubernetes             |
| Entraînement   | Manuel (`python train.py`)    | Pipeline automatisé (CI/CD, Airflow)       |
| Monitoring      | Aucun                         | Prometheus + Grafana (latence, dérive)     |

> Ce qui **ne change pas** : le Model Registry, les alias, la séparation data scientist / MLOps.

---

## Vers la production — 3 étapes

### Étape 1 — Ce que vous avez déjà ✓

En terminant ce TP, vous avez :

- Un modèle versionné dans un **Model Registry** avec un alias de déploiement
- Une **API REST documentée** (`app.py`) prête à recevoir des requêtes
- Un `Dockerfile` pour empaqueter cette API de façon portable

La seule variable qui change entre local et production :

```bash
# Local (défaut)
MLFLOW_TRACKING_URI=http://localhost:5000

# Production — pointe vers le serveur MLflow de l'équipe
export MLFLOW_TRACKING_URI=http://mlflow.mon-entreprise.com:5000
```

`train.py` et `app.py` n'ont pas besoin d'être modifiés.

---

### Étape 2 — Dockeriser et tester localement

Le `Dockerfile` du repo empaquette `app.py` en image portable :

```bash
# 1. Construire l'image
docker build -t emotion-classifier .

# 2. Tester le conteneur en local (même résultat qu'avec python app.py)
docker run -p 5002:5002 emotion-classifier

# 3. Pousser l'image sur Docker Hub (ou un registry privé)
docker tag emotion-classifier monuser/emotion-classifier:v1
docker push monuser/emotion-classifier:v1
```

À partir de là, n'importe quel serveur peut démarrer l'API avec une seule commande :

```bash
docker run -p 5002:5002 \
  -e MLFLOW_TRACKING_URI=http://mlflow.mon-entreprise.com:5000 \
  monuser/emotion-classifier:v1
```

---

### Étape 3 — MLflow hébergé et déploiement cloud

#### Option A — Services MLflow managés (recommandé pour débuter)


| Service                          | MLflow inclus  | Gratuit ?              | Idéal pour                |
| -------------------------------- | -------------- | ---------------------- | -------------------------- |
| **DagsHub**                      | Oui, hébergé | Oui (plan gratuit)     | Étudiants, petits projets |
| **Databricks Community Edition** | Oui, natif     | Oui                    | Formation, prototypage     |
| **Azure ML**                     | Oui, intégré | Payant (essai gratuit) | Entreprises Microsoft      |
| **AWS SageMaker**                | Oui, intégré | Payant (essai gratuit) | Entreprises AWS            |

Avec **DagsHub** par exemple, il suffit de :

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/monuser/mon-repo.mlflow
python train.py --mode 3   # les runs apparaissent sur DagsHub, pas en local
```

#### Option B — Héberger son propre serveur MLflow (cloud privé ou VM)

Sur n'importe quelle VM (OVH, Scaleway, serveur interne…) :

```bash
# Sur le serveur
mlflow server \
  --backend-store-uri postgresql://user:pass@localhost/mlflow \
  --artifacts-destination s3://mon-bucket/mlflow \
  --host 0.0.0.0 --port 5000

# Puis déployer l'API
docker run -p 5002:5002 \
  -e MLFLOW_TRACKING_URI=http://IP-DU-SERVEUR:5000 \
  monuser/emotion-classifier:v1
```

> **MLflow est cloud-agnostique** — il fonctionne de façon identique sur un cloud privé interne,
> AWS, Azure ou GCP. Vous n'êtes pas enfermé chez un fournisseur.

---

## Structure

```
├── Dockerfile       # Conteneurisation de app.py
├── MLproject        # Entry points pour mlflow run
├── python_env.yaml  # Dépendances (mlflow run)
├── requirements.txt # Dépendances (pip / Docker)
├── app.py           # API FastAPI — Swagger UI sur :5002/docs
├── predict.py       # Client CLI — --text, --api, --port
└── train.py         # Entraînement — --mode 1 / 2 / 3
```

---

## Pour aller plus loin

- Modifier `C=1.0` dans `train.py` et relancer — voir la nouvelle version apparaître dans le registry
- `mlflow run . -e train` — entraînement reproductible via MLproject
- Connecter `train.py` à **DagsHub** pour un registry partagé sans infrastructure
- Remplacer TF-IDF par un modèle pré-entraîné (BERT) pour dépasser 90% d'accuracy


## Screenshots (MLFlow et API, SwaggerUI)

![1776833365174](images/README/1776833365174.png)

![1776833395527](images/README/1776833395527.png)
