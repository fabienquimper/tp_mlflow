"""
ETAPE 1 — Entraînement et enregistrement du modèle

Trois modes de données pour observer l'impact sur la qualité du modèle :

  python train.py --mode 1   # ~12 phrases  → modèle très médiocre (intentionnel)
  python train.py --mode 2   # ~105 phrases → modèle passable
  python train.py --mode 3   # ~1600 phrases (HuggingFace inclus) → meilleur modèle

Conseil : lance les 3 modes puis compare les runs dans l'UI MLflow (http://localhost:5000)
"""
import argparse
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ==============================================================================
# DATASETS — trois niveaux de volume
# ==============================================================================

# MODE 1 — Intentionnellement minuscule pour montrer l'échec
TINY_HAPPY = [
    "I am happy", "This is great", "I feel wonderful", "Amazing day",
]
TINY_SAD = [
    "I am sad", "This is awful", "I feel terrible", "What a bad day",
]
TINY_ANGRY = [
    "I am angry", "How dare you", "I'm furious", "This is outrageous",
]

# MODE 2 — Ensemble moyen, entièrement hardcodé (pas de réseau nécessaire)
MEDIUM_HAPPY = [
    "I am so happy today!", "This is wonderful!", "I feel great!", "Amazing news!",
    "I love this so much", "Best day ever!", "Feeling fantastic!", "Super excited!",
    "What a beautiful day", "I'm thrilled about this", "I just got promoted!",
    "This made my day", "I'm over the moon", "Life is beautiful", "Pure joy!",
    "I'm smiling so much", "Couldn't be happier", "Everything is going well",
    "I feel blessed", "What great news!", "I'm so grateful", "This is delightful",
    "Wonderful surprise today", "I feel on top of the world", "So much happiness",
    "I'm laughing and smiling", "Great things happening", "I feel alive and joyful",
    "Today is a happy day", "I love everything right now", "Brilliant day ahead",
    "I feel cheerful and bright", "Such a lovely moment", "I'm ecstatic",
    "Feeling positive and light",
]
MEDIUM_SAD = [
    "I am so sad", "This is the worst day of my life", "I feel awful", "What a bad day",
    "I miss you so much", "This hurts deeply", "I'm devastated", "Feeling hopeless",
    "I can't stop crying", "Everything feels wrong", "I feel so lonely",
    "Nothing makes sense anymore", "I lost something precious", "I'm heartbroken",
    "Life feels empty", "I don't see the point", "I feel lost", "So much grief",
    "This is so painful", "I'm deeply depressed", "I feel empty inside",
    "There is no hope left", "Everything hurts", "I'm broken and sad",
    "I just want to cry", "My heart is aching", "I feel so down today",
    "Nothing brings me joy anymore", "I'm miserable and lonely",
    "Sadness overwhelms me", "I feel defeated", "This loss is unbearable",
    "I'm drowning in sadness", "The pain never stops", "I feel worthless and sad",
]
MEDIUM_ANGRY = [
    "I'm furious!", "This makes me so angry", "How dare you!", "I'm outraged!",
    "So frustrated with this", "This is infuriating", "I can't stand this",
    "Unacceptable behavior", "Stop doing that!", "I'm really annoyed",
    "I want to scream", "This is absolutely ridiculous", "They crossed a line",
    "I'm boiling with rage", "How could they do this", "I'm so mad right now",
    "This is complete nonsense", "I hate when this happens",
    "They make me so angry", "I can't believe this", "I am furious about this",
    "This enrages me completely", "I'm filled with anger", "Such an angry feeling",
    "Rage is building inside me", "I'm livid and furious",
    "This makes my blood boil", "I could explode with anger",
    "How infuriating can this be", "I'm seething with rage",
    "This is maddening behavior", "I'm disgusted and angry",
    "Totally unacceptable and infuriating", "I despise this so much",
    "My anger knows no bounds",
]

# ==============================================================================
# CHARGEMENT DES DONNÉES selon le mode
# ==============================================================================

def load_data(mode: int) -> tuple[list[str], list[str]]:
    if mode == 1:
        texts = TINY_HAPPY + TINY_SAD + TINY_ANGRY
        labels = ["happy"] * len(TINY_HAPPY) + ["sad"] * len(TINY_SAD) + ["angry"] * len(TINY_ANGRY)
        return texts, labels

    texts = MEDIUM_HAPPY + MEDIUM_SAD + MEDIUM_ANGRY
    labels = (
        ["happy"] * len(MEDIUM_HAPPY)
        + ["sad"] * len(MEDIUM_SAD)
        + ["angry"] * len(MEDIUM_ANGRY)
    )

    if mode == 2:
        return texts, labels

    # MODE 3 — Ajout des données HuggingFace (dair-ai/emotion)
    # labels HF : 0=sadness, 1=joy, 3=anger
    HF_MAP = {0: "sad", 1: "happy", 3: "angry"}
    HF_LIMIT = 500

    try:
        from datasets import load_dataset
        print("  Téléchargement du dataset HuggingFace 'dair-ai/emotion'...")
        ds = load_dataset("dair-ai/emotion", split="train")
        counts = {"happy": 0, "sad": 0, "angry": 0}
        for item in ds:
            emotion = HF_MAP.get(item["label"])
            if emotion and counts[emotion] < HF_LIMIT:
                texts.append(item["text"])
                labels.append(emotion)
                counts[emotion] += 1
        hf_total = sum(counts.values())
        print(f"  OK : {hf_total} phrases HuggingFace ajoutees ({counts})")
    except ImportError:
        print("  ATTENTION : `datasets` non installe — passage en mode 2")
        print("  Installe-le avec : pip install datasets")

    return texts, labels


# ==============================================================================
# AFFICHAGE VERBOSE
# ==============================================================================

MODE_DESCRIPTIONS = {
    1: ("mode-1-tiny",   "~12 phrases — modele tres mediocre (intentionnel)"),
    2: ("mode-2-medium", "~105 phrases — modele passable"),
    3: ("mode-3-full",   "~1600 phrases avec HuggingFace — meilleur modele"),
}

ACCURACY_COMMENTS = {
    (0,  40): "Tres mauvais — le modele ne generalise pas du tout. Normal avec si peu de donnees !",
    (40, 60): "Mediocre — le modele devine parfois correctement, mais reste peu fiable.",
    (60, 75): "Passable — resultats corrects mais beaucoup d'erreurs sur les cas ambigus.",
    (75, 88): "Bon — le modele est utilisable pour une demo ou un prototype.",
    (88, 101): "Tres bon — proche de la limite de TF-IDF+LogReg sur ce type de texte.",
}

def accuracy_comment(acc_pct: float) -> str:
    for (low, high), comment in ACCURACY_COMMENTS.items():
        if low <= acc_pct < high:
            return comment
    return ""


# ==============================================================================
# SCRIPT PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entraine un classifieur d'emotions avec 3 niveaux de donnees",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python train.py --mode 1   # Peu de donnees  → modele nul
  python train.py --mode 2   # Donnees moyennes → modele acceptable
  python train.py --mode 3   # Beaucoup de donnees → meilleur modele

Ensuite, compare les 3 runs dans l'UI : http://localhost:5000
        """,
    )
    parser.add_argument(
        "--mode", type=int, choices=[1, 2, 3], default=2,
        help="Volume de donnees : 1=tiny (~12), 2=medium (~105), 3=full (~1600)",
    )
    args = parser.parse_args()

    run_name, mode_desc = MODE_DESCRIPTIONS[args.mode]

    print()
    print("=" * 60)
    print(f"  ENTRAINEMENT — MODE {args.mode}")
    print(f"  {mode_desc}")
    print("=" * 60)

    # --- Chargement des données ---
    print(f"\n[1/5] Chargement des donnees (mode {args.mode})...")
    texts, labels = load_data(args.mode)

    from collections import Counter
    dist = Counter(labels)
    print(f"  Total   : {len(texts)} phrases")
    print(f"  happy   : {dist['happy']} exemples")
    print(f"  sad     : {dist['sad']} exemples")
    print(f"  angry   : {dist['angry']} exemples")

    if args.mode == 1:
        print()
        print("  [!] ATTENTION : seulement 4 exemples par classe !")
        print("      Le modele va probablement etre tres mauvais.")
        print("      C'est voulu — pour illustrer l'importance des donnees.")

    # --- Split train/test ---
    print(f"\n[2/5] Decoupage train / test (80% / 20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42,
        stratify=labels if args.mode > 1 else None,
    )
    print(f"  Entrainement : {len(X_train)} phrases")
    print(f"  Test         : {len(X_test)} phrases")

    # --- Modèle ---
    print(f"\n[3/5] Construction du pipeline TF-IDF + LogisticRegression...")
    print("  TfidfVectorizer : transforme le texte en vecteurs de mots")
    print("  LogisticRegression : classifie selon ces vecteurs")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf",   LogisticRegression(max_iter=1000, C=1.0)),
    ])

    # --- Entraînement ---
    print(f"\n[4/5] Entrainement en cours...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    acc_pct = acc * 100

    print(f"\n  ACCURACY : {acc_pct:.1f}%")
    print(f"  → {accuracy_comment(acc_pct)}")
    print()
    print(classification_report(y_test, y_pred, zero_division=0))

    # --- MLflow ---
    print(f"[5/5] Enregistrement dans MLflow...")
    mlflow.set_experiment("emotion-classifier")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "mode":        args.mode,
            "model":       "TF-IDF + LogisticRegression",
            "ngram_range": "(1,2)",
            "C":           1.0,
            "n_samples":   len(texts),
            "n_train":     len(X_train),
            "n_test":      len(X_test),
        })
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("n_happy",  dist["happy"])
        mlflow.log_metric("n_sad",    dist["sad"])
        mlflow.log_metric("n_angry",  dist["angry"])

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="emotion-model",
            registered_model_name="emotion-classifier",
            input_example=X_train[:1],
        )

    client = MlflowClient()
    latest = client.get_registered_model("emotion-classifier").latest_versions[-1]
    client.set_registered_model_alias("emotion-classifier", "champion", latest.version)

    print(f"  Run     : '{run_name}'")
    print(f"  Modele  : version {latest.version}, alias @champion")
    print(f"  Samples : {len(texts)} phrases entrainees")

    print()
    print("=" * 60)
    print("  TERMINE !")
    print()
    print("  Pour tester les predictions :")
    print('    python predict.py --text "I feel great!"')
    print()
    print("  Pour voir ce run dans l'interface MLflow :")
    print("    http://localhost:5000  →  onglet 'Training runs'")
    print()
    if args.mode < 3:
        print(f"  Conseil : essaie aussi --mode {args.mode + 1} pour comparer !")
    else:
        print("  Tu as lance tous les modes ! Compare les 3 runs dans l'UI.")
    print("=" * 60)
    print()
