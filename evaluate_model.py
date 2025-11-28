import argparse
import pickle
import re
from pathlib import Path

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# Reuse the same preprocessing choices used by the API
STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


def preprocess_text(text: str) -> str:
    """Normalize, clean, and stem a review just like the API does."""
    review = re.sub(r"[^a-zA-Z]", " ", str(text))
    tokens = review.lower().split()
    tokens = [STEMMER.stem(word) for word in tokens if word not in STOPWORDS]
    return " ".join(tokens)


def load_artifacts(models_dir: Path):
    """Load model artifacts that power the Flask API."""
    with open(models_dir / "model_xgb.pkl", "rb") as fh:
        predictor = pickle.load(fh)
    with open(models_dir / "scaler.pkl", "rb") as fh:
        scaler = pickle.load(fh)
    with open(models_dir / "countVectorizer.pkl", "rb") as fh:
        vectorizer = pickle.load(fh)

    return predictor, scaler, vectorizer


def infer(texts, predictor, scaler, vectorizer):
    processed = [preprocess_text(text) for text in texts]
    features = vectorizer.transform(processed).toarray()
    scaled = scaler.transform(features)
    proba = predictor.predict_proba(scaled)
    preds = proba.argmax(axis=1)
    return preds


def read_dataset(data_path: Path, text_column: str, label_column: str) -> pd.DataFrame:
    if data_path.suffix == ".tsv":
        df = pd.read_csv(data_path, sep="\t")
    else:
        df = pd.read_csv(data_path)

    missing_cols = {text_column, label_column} - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Dataset at {data_path} is missing columns: {', '.join(missing_cols)}"
        )

    df = df[[text_column, label_column]].dropna()
    df[text_column] = df[text_column].astype(str)
    df[label_column] = df[label_column].astype(int)
    return df


def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1
    )
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Negative (0)", "Positive (1)"],
    )

    return accuracy, precision, recall, f1, cm, report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the pickled sentiment model on a labeled dataset."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("Data/amazon_alexa.tsv"),
        help="Dataset with text and binary labels (default: Data/amazon_alexa.tsv).",
    )
    parser.add_argument(
        "--text-column",
        default="verified_reviews",
        help="Name of the column that contains the review text.",
    )
    parser.add_argument(
        "--label-column",
        default="feedback",
        help="Name of the column that contains the binary label (1=Positive, 0=Negative).",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("Models"),
        help="Directory containing the trained model artifacts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    predictor, scaler, vectorizer = load_artifacts(args.models_dir)
    dataset = read_dataset(args.data_path, args.text_column, args.label_column)

    y_true = dataset[args.label_column].values
    y_pred = infer(dataset[args.text_column].values, predictor, scaler, vectorizer)

    accuracy, precision, recall, f1, cm, report = compute_metrics(y_true, y_pred)

    print("=== Aggregate Metrics ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}\n")

    print("=== Classification Report ===")
    print(report)

    print("=== Confusion Matrix ===")
    print(cm)


if __name__ == "__main__":
    main()

