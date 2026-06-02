"""End-to-end sentiment analysis pipeline for Sriwijaya Air app reviews.

The project uses Google Play rating scores as weak sentiment labels:
1-2 = negative, 3 = neutral, 4-5 = positive.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "week02_dataprocessing" / "SriwijayaAir_Preprocessed.csv"
OUTPUT_DIR = ROOT_DIR / "outputs"
PROCESSED_DIR = OUTPUT_DIR / "processed"
REPORT_DIR = OUTPUT_DIR / "reports"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = ROOT_DIR / "models"

LABEL_ORDER = ["negative", "neutral", "positive"]
LABEL_COLORS = {
    "negative": "#d1495b",
    "neutral": "#6c757d",
    "positive": "#2a9d8f",
}
NUMERIC_FEATURES = ["word_count", "char_count", "avg_word_length"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate sentiment classifiers for Sriwijaya Air reviews."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the preprocessed CSV file.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data used for the test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible train/test split.",
    )
    return parser.parse_args()


def ensure_directories() -> None:
    for path in [PROCESSED_DIR, REPORT_DIR, FIGURE_DIR, MODEL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def score_to_sentiment(score: int) -> str:
    """Convert Google Play rating score into a weak sentiment label."""
    score = int(score)
    if score <= 2:
        return "negative"
    if score == 3:
        return "neutral"
    return "positive"


def add_sentiment_and_features(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"content", "score", "clean_text"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    prepared = df.copy()
    prepared["content"] = prepared["content"].fillna("").astype(str)
    prepared["clean_text"] = prepared["clean_text"].fillna("").astype(str).str.strip()
    prepared = prepared[prepared["clean_text"].str.len() > 0].copy()

    prepared["score"] = pd.to_numeric(prepared["score"], errors="coerce")
    prepared = prepared.dropna(subset=["score"]).copy()
    prepared["score"] = prepared["score"].astype(int)
    prepared["sentiment"] = prepared["score"].apply(score_to_sentiment)

    tokens = prepared["clean_text"].str.split()
    prepared["word_count"] = tokens.apply(len)
    prepared["char_count"] = prepared["clean_text"].str.len()
    prepared["avg_word_length"] = prepared.apply(
        lambda row: row["char_count"] / row["word_count"] if row["word_count"] else 0,
        axis=1,
    )

    return prepared.reset_index(drop=True)


def build_feature_transformer() -> ColumnTransformer:
    text_vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        max_features=8000,
        sublinear_tf=True,
    )

    return ColumnTransformer(
        transformers=[
            ("tfidf", text_vectorizer, "clean_text"),
            ("numeric", MinMaxScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def build_models(random_state: int) -> Dict[str, Pipeline]:
    classifiers = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "naive_bayes": MultinomialNB(alpha=0.7),
        "linear_svm": LinearSVC(
            class_weight="balanced",
            random_state=random_state,
        ),
    }

    return {
        name: Pipeline(
            steps=[
                ("features", build_feature_transformer()),
                ("classifier", classifier),
            ]
        )
        for name, classifier in classifiers.items()
    }


def balance_training_data(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Randomly oversample minority labels on the training split only."""
    train_df = pd.concat(
        [
            x_train.reset_index(drop=True),
            y_train.reset_index(drop=True).rename("sentiment"),
        ],
        axis=1,
    )
    target_size = train_df["sentiment"].value_counts().max()

    balanced_parts = []
    for sentiment, group in train_df.groupby("sentiment"):
        balanced_parts.append(
            group.sample(target_size, replace=True, random_state=random_state)
        )

    balanced = (
        pd.concat(balanced_parts)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
    return balanced[["clean_text", *NUMERIC_FEATURES]], balanced["sentiment"]


def make_feature_tables(df: pd.DataFrame) -> None:
    texts = df["clean_text"].tolist()

    bow = CountVectorizer(ngram_range=(1, 1), min_df=2, max_features=1000)
    bow_matrix = bow.fit_transform(texts)
    bow_counts = np.asarray(bow_matrix.sum(axis=0)).ravel()
    bow_terms = pd.DataFrame(
        {"term": bow.get_feature_names_out(), "frequency": bow_counts}
    ).sort_values("frequency", ascending=False)
    bow_terms.head(50).to_csv(REPORT_DIR / "top_bow_terms.csv", index=False)

    ngrams = CountVectorizer(ngram_range=(2, 3), min_df=2, max_features=1000)
    ngram_matrix = ngrams.fit_transform(texts)
    ngram_counts = np.asarray(ngram_matrix.sum(axis=0)).ravel()
    ngram_terms = pd.DataFrame(
        {"term": ngrams.get_feature_names_out(), "frequency": ngram_counts}
    ).sort_values("frequency", ascending=False)
    ngram_terms.head(50).to_csv(REPORT_DIR / "top_ngram_terms.csv", index=False)

    tfidf = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=3000)
    tfidf_matrix = tfidf.fit_transform(texts)
    tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    tfidf_terms = pd.DataFrame(
        {"term": tfidf.get_feature_names_out(), "mean_tfidf": tfidf_scores}
    ).sort_values("mean_tfidf", ascending=False)
    tfidf_terms.head(50).to_csv(REPORT_DIR / "top_tfidf_terms.csv", index=False)

    rows = []
    for sentiment in LABEL_ORDER:
        sentiment_tokens = " ".join(
            df.loc[df["sentiment"] == sentiment, "clean_text"]
        ).split()
        for term, count in Counter(sentiment_tokens).most_common(20):
            rows.append(
                {
                    "sentiment": sentiment,
                    "term": term,
                    "frequency": count,
                }
            )
    pd.DataFrame(rows).to_csv(REPORT_DIR / "top_terms_by_sentiment.csv", index=False)


def plot_sentiment_distribution(df: pd.DataFrame) -> None:
    counts = df["sentiment"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    colors = [LABEL_COLORS[label] for label in LABEL_ORDER]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=colors)
    ax.set_title("Distribusi Sentimen Ulasan Sriwijaya Air Mobile")
    ax.set_xlabel("Sentimen")
    ax.set_ylabel("Jumlah ulasan")
    ax.bar_label(bars, padding=3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "sentiment_distribution.png", dpi=180)
    plt.close(fig)


def plot_model_comparison(metrics: pd.DataFrame) -> None:
    plot_df = metrics.set_index("model")[["accuracy", "macro_f1", "weighted_f1"]]
    plot_df.index = [model.replace("_", " ").title() for model in plot_df.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df.plot(kind="bar", ax=ax, color=["#457b9d", "#f4a261", "#2a9d8f"])
    ax.set_title("Perbandingan Performa Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(f"Confusion Matrix - {model_name.replace('_', ' ').title()}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(LABEL_ORDER)), labels=LABEL_ORDER)
    ax.set_yticks(np.arange(len(LABEL_ORDER)), labels=LABEL_ORDER)

    threshold = cm.max() / 2 if cm.size else 0
    for row_idx in range(cm.shape[0]):
        for col_idx in range(cm.shape[1]):
            color = "white" if cm[row_idx, col_idx] > threshold else "black"
            ax.text(
                col_idx,
                row_idx,
                int(cm[row_idx, col_idx]),
                ha="center",
                va="center",
                color=color,
            )

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "confusion_matrix_best_model.png", dpi=180)
    plt.close(fig)


def plot_top_terms_by_sentiment() -> None:
    top_terms = pd.read_csv(REPORT_DIR / "top_terms_by_sentiment.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharex=False)

    for ax, sentiment in zip(axes, LABEL_ORDER):
        subset = top_terms[top_terms["sentiment"] == sentiment].head(10)
        subset = subset.sort_values("frequency", ascending=True)
        ax.barh(
            subset["term"],
            subset["frequency"],
            color=LABEL_COLORS[sentiment],
        )
        ax.set_title(sentiment.title())
        ax.set_xlabel("Frequency")

    fig.suptitle("Top Terms by Sentiment")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "top_terms_by_sentiment.png", dpi=180)
    plt.close(fig)


def save_classification_report(
    report: Dict[str, Dict[str, float]], model_name: str
) -> None:
    rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            row = {"label": label}
            row.update(values)
            rows.append(row)
        else:
            rows.append({"label": label, "score": values})
    pd.DataFrame(rows).to_csv(
        REPORT_DIR / f"classification_report_{model_name}.csv", index=False
    )


def train_and_evaluate(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> Tuple[Dict[str, Pipeline], pd.DataFrame, str, np.ndarray]:
    feature_columns = ["clean_text", *NUMERIC_FEATURES]
    x_train, x_test, y_train, y_test = train_test_split(
        df[feature_columns],
        df["sentiment"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["sentiment"],
    )
    x_train_balanced, y_train_balanced = balance_training_data(
        x_train,
        y_train,
        random_state,
    )

    trained_models = {}
    metric_rows = []
    best_model_name = ""
    best_rank = (-1.0, -1.0)
    best_cm = np.zeros((len(LABEL_ORDER), len(LABEL_ORDER)), dtype=int)

    for model_name, model in build_models(random_state).items():
        model.fit(x_train_balanced, y_train_balanced)
        predictions = model.predict(x_test)

        report = classification_report(
            y_test,
            predictions,
            labels=LABEL_ORDER,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_test, predictions, labels=LABEL_ORDER)

        metrics = {
            "model": model_name,
            "accuracy": accuracy_score(y_test, predictions),
            "macro_f1": f1_score(
                y_test,
                predictions,
                labels=LABEL_ORDER,
                average="macro",
                zero_division=0,
            ),
            "weighted_f1": f1_score(
                y_test,
                predictions,
                labels=LABEL_ORDER,
                average="weighted",
                zero_division=0,
            ),
            "negative_f1": report["negative"]["f1-score"],
            "neutral_f1": report["neutral"]["f1-score"],
            "positive_f1": report["positive"]["f1-score"],
        }

        trained_models[model_name] = model
        metric_rows.append(metrics)
        save_classification_report(report, model_name)
        pd.DataFrame(cm, index=LABEL_ORDER, columns=LABEL_ORDER).to_csv(
            REPORT_DIR / f"confusion_matrix_{model_name}.csv"
        )

        ranking_score = (metrics["macro_f1"], metrics["accuracy"])
        if ranking_score > best_rank:
            best_rank = ranking_score
            best_model_name = model_name
            best_cm = cm

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["macro_f1", "accuracy"], ascending=False
    )
    return trained_models, metrics_df, best_model_name, best_cm


def save_run_summary(
    df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    best_model_name: str,
    args: argparse.Namespace,
) -> None:
    best_metrics = metrics_df.iloc[0].to_dict()
    class_counts = df["sentiment"].value_counts().reindex(LABEL_ORDER, fill_value=0)

    summary = {
        "input_file": str(args.input),
        "rows_used": int(len(df)),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "label_mapping": {
            "1-2": "negative",
            "3": "neutral",
            "4-5": "positive",
        },
        "class_distribution": {
            label: int(count) for label, count in class_counts.items()
        },
        "feature_engineering": {
            "text": "TF-IDF with unigram, bigram, and trigram features",
            "numeric": NUMERIC_FEATURES,
            "training_balance": "Random oversampling is applied to the training split only.",
            "exploration_tables": [
                "top_bow_terms.csv",
                "top_ngram_terms.csv",
                "top_tfidf_terms.csv",
                "top_terms_by_sentiment.csv",
            ],
        },
        "best_model": best_model_name,
        "best_metrics": {
            key: float(value)
            for key, value in best_metrics.items()
            if key != "model"
        },
    }

    (REPORT_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    markdown = [
        "# Run Summary",
        "",
        f"- Rows used: {len(df)}",
        f"- Label mapping: 1-2 negative, 3 neutral, 4-5 positive",
        f"- Class distribution: "
        + ", ".join(f"{label}={count}" for label, count in class_counts.items()),
        f"- Best model: {best_model_name}",
        f"- Accuracy: {best_metrics['accuracy']:.4f}",
        f"- Macro F1: {best_metrics['macro_f1']:.4f}",
        f"- Weighted F1: {best_metrics['weighted_f1']:.4f}",
    ]
    (REPORT_DIR / "run_summary.md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    ensure_directories()

    df = pd.read_csv(args.input)
    prepared = add_sentiment_and_features(df)
    prepared.to_csv(
        PROCESSED_DIR / "SriwijayaAir_Sentiment_Labeled.csv",
        index=False,
    )

    make_feature_tables(prepared)
    plot_sentiment_distribution(prepared)
    plot_top_terms_by_sentiment()

    trained_models, metrics_df, best_model_name, best_cm = train_and_evaluate(
        prepared,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    metrics_df.to_csv(REPORT_DIR / "model_metrics.csv", index=False)
    (REPORT_DIR / "model_metrics.json").write_text(
        metrics_df.to_json(orient="records", indent=2),
        encoding="utf-8",
    )
    plot_model_comparison(metrics_df)
    plot_confusion_matrix(best_cm, best_model_name)

    best_model = trained_models[best_model_name]
    joblib.dump(best_model, MODEL_DIR / "best_sentiment_model.joblib")

    predictions = prepared.copy()
    predictions["predicted_sentiment"] = best_model.predict(
        predictions[["clean_text", *NUMERIC_FEATURES]]
    )
    predictions.to_csv(
        PROCESSED_DIR / "SriwijayaAir_Sentiment_Predictions.csv",
        index=False,
    )

    save_run_summary(prepared, metrics_df, best_model_name, args)

    print("Sentiment pipeline finished.")
    print(f"Rows used: {len(prepared)}")
    print(f"Best model: {best_model_name}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
