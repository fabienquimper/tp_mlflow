# Dockerfile — conteneurise app.py pour un déploiement réel
#
# Ce fichier montre que passer du local à la production ne change pas le code :
# on emballe l'application dans une image portable, déployable n'importe où.
#
# Build :  docker build -t emotion-classifier .
# Run   :  docker run -p 5002:5002 \
#            -e MLFLOW_TRACKING_URI=http://mon-serveur-mlflow:5000 \
#            emotion-classifier

FROM python:3.12-slim

WORKDIR /app

# Dépendances d'abord (layer mis en cache si requirements.txt ne change pas)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code de l'application
COPY app.py .

# Le modèle sera chargé depuis le MLflow Tracking Server au démarrage
# (configuré via la variable d'environnement MLFLOW_TRACKING_URI)
ENV MLFLOW_TRACKING_URI=http://localhost:5000

EXPOSE 5002

CMD ["python", "app.py"]
