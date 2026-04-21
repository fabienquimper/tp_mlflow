"""
ETAPE 2 — Consommation du modèle déployé

Deux modes d'utilisation :
  A) Direct (registry)  →  python predict.py --text "I feel great"
  B) Via API REST       →  démarrer le serveur d'abord, puis python predict.py --api --text "..."

Pour démarrer le serveur REST (mode B) :
  mlflow models serve -m "models:/emotion-classifier@champion" -p 5001 --no-conda
"""
import argparse

EXAMPLES = [
    "I am absolutely thrilled about this!",
    "This is the worst day of my life",
    "How dare they do this to me!",
    "I love spending time with my friends",
    "I feel so lost and empty inside",
]

CONFIDENCE_BARS = {
    (0,  20): "          ",
    (20, 40): "##        ",
    (40, 60): "#####     ",
    (60, 80): "#######   ",
    (80, 101): "##########",
}

def confidence_bar(pct: float) -> str:
    for (low, high), bar in CONFIDENCE_BARS.items():
        if low <= pct < high:
            return bar
    return "##########"

def confidence_comment(pct: float) -> str:
    if pct >= 85:
        return "tres confiant"
    if pct >= 65:
        return "assez confiant"
    if pct >= 45:
        return "incertain"
    return "tres incertain — le modele hesite beaucoup"

def format_prediction(text: str, emotion: str, proba: dict) -> str:
    lines = []
    lines.append(f'\n  Texte   : "{text}"')
    lines.append(f"  Resultat: [{emotion.upper():>6}]")
    lines.append(f"  Confiance par emotion :")
    for label in ["happy", "sad", "angry"]:
        pct = proba.get(label, 0) * 100
        marker = "<-- PREDICTION" if label == emotion else ""
        lines.append(
            f"    {label:>6} : [{confidence_bar(pct)}] {pct:5.1f}%  {marker}"
        )
    top_pct = proba.get(emotion, 0) * 100
    lines.append(f"  → Le modele est {confidence_comment(top_pct)} ({top_pct:.0f}%)")
    return "\n".join(lines)


def predict_direct(texts: list[str]) -> list[tuple[str, dict]]:
    """Charge le modèle depuis le registry MLflow (sans serveur HTTP)."""
    import mlflow.sklearn
    from mlflow import MlflowClient

    print("\n  Connexion au Model Registry MLflow...")
    client = MlflowClient()
    model_meta = client.get_registered_model("emotion-classifier")
    versions = model_meta.latest_versions
    champion_version = None
    for v in versions:
        aliases = client.get_model_version(model_meta.name, v.version).aliases
        if "champion" in aliases:
            champion_version = v.version
            break
    print(f"  Modele charge  : emotion-classifier v{champion_version} (@champion)")
    print(f"  Source         : Model Registry local (mlruns/)")
    print(f"  Equivalent prod: requete a https://api.entreprise.com/predict")

    model = mlflow.sklearn.load_model("models:/emotion-classifier@champion")
    classes = model.classes_.tolist()
    probas = model.predict_proba(texts)
    results = []
    for text, proba_row in zip(texts, probas):
        proba_dict = dict(zip(classes, proba_row))
        emotion = max(proba_dict, key=proba_dict.get)
        results.append((emotion, proba_dict))
    return results


def predict_api(texts: list[str], port: int = 5001) -> list[tuple[str, dict]]:
    """Appelle le modèle via l'API REST (mlflow models serve)."""
    import requests

    url = f"http://localhost:{port}/invocations"
    print(f"\n  Appel API REST : {url}")
    print(f"  Equivalent prod: requete a https://api.entreprise.com/predict")

    payload = {"inputs": texts}
    response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
    response.raise_for_status()

    predictions = response.json()["predictions"]
    return [(pred, {}) for pred in predictions]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Teste le classifieur d'emotions deploye",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python predict.py
  python predict.py --text "I feel amazing!"
  python predict.py --text "I'm so furious right now"

  # Apres avoir lance le serveur REST :
  mlflow models serve -m "models:/emotion-classifier@champion" -p 5001 --no-conda
  python predict.py --api --text "This makes me so angry"
        """,
    )
    parser.add_argument("--text",  type=str, default=None, help="Texte a classifier")
    parser.add_argument("--api",   action="store_true",    help="Utiliser l'API REST au lieu du registry direct")
    parser.add_argument("--port",  type=int, default=5001, help="Port du serveur REST (defaut: 5001)")
    args = parser.parse_args()

    texts_to_test = [args.text] if args.text else EXAMPLES
    mode_label = f"API REST (port {args.port})" if args.api else "Registry direct (local)"

    print()
    print("=" * 60)
    print(f"  PREDICTION — {mode_label}")
    print("=" * 60)

    if args.api:
        print("\n  [!] Mode API REST : le modele est appele via HTTP,")
        print("      comme depuis une vraie application en production.")
        print("      Le script ne sait pas comment le modele fonctionne,")
        print("      il envoie juste du texte et recoit une emotion.")
        results = predict_api(texts_to_test, args.port)
    else:
        print("\n  [i] Mode direct : le modele est charge depuis le registry")
        print("      MLflow local. Pas de serveur HTTP necessaire.")
        print("      En production, ce serait un appel API authentifie.")
        results = predict_direct(texts_to_test)

    if args.api:
        print(f"\n  {len(results)} prediction(s) recues :")
        for text, (emotion, _) in zip(texts_to_test, results):
            print(f'\n  Texte   : "{text}"')
            print(f"  Resultat: [{emotion.upper():>6}]")
            print("  (scores de confiance non disponibles via l'API REST basique)")
    else:
        print(f"\n  {len(results)} prediction(s) :")
        for text, (emotion, proba) in zip(texts_to_test, results):
            print(format_prediction(text, emotion, proba))

    print()
    print("=" * 60)
    print()
