"""Generate portfolio-ready evidence visuals for the sentiment project."""

from __future__ import annotations

import json
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_FILE = ROOT_DIR / "outputs" / "processed" / "SriwijayaAir_Sentiment_Predictions.csv"
REPORT_DIR = ROOT_DIR / "outputs" / "reports"
PORTFOLIO_DIR = ROOT_DIR / "outputs" / "portfolio"

LABEL_ORDER = ["negative", "neutral", "positive"]
LABEL_COLORS = {
    "negative": "#d1495b",
    "neutral": "#6c757d",
    "positive": "#2a9d8f",
}
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "naive_bayes": "Naive Bayes",
}
GENERIC_TERMS = {
    "aplikasi",
    "app",
    "apps",
    "nya",
    "ya",
    "aja",
    "sriwijaya",
    "air",
    "mobile",
    "banget",
    "gak",
    "ga",
    "yg",
    "yang",
    "dan",
    "di",
    "ke",
    "ini",
}


def ensure_dir() -> None:
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = pd.read_csv(PROCESSED_FILE)
    df["at"] = pd.to_datetime(df["at"], errors="coerce")
    df["content"] = df["content"].fillna("").astype(str)
    df["clean_text"] = df["clean_text"].fillna("").astype(str)

    metrics = pd.read_csv(REPORT_DIR / "model_metrics.csv")
    summary = json.loads((REPORT_DIR / "run_summary.json").read_text(encoding="utf-8"))
    return df, metrics, summary


def add_caption(fig: plt.Figure, caption: str) -> None:
    fig.subplots_adjust(bottom=0.15)
    fig.text(
        0.02,
        0.03,
        caption,
        fontsize=10,
        color="#495057",
        ha="left",
        va="bottom",
    )


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.savefig(PORTFOLIO_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_case_study_overview(
    df: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: dict,
) -> None:
    best = metrics.iloc[0]
    counts = df["sentiment"].value_counts().reindex(LABEL_ORDER, fill_value=0)

    fig = plt.figure(figsize=(14, 9), facecolor="white")
    grid = fig.add_gridspec(3, 4, height_ratios=[0.75, 1.4, 1.25], hspace=0.45, wspace=0.45)

    title_ax = fig.add_subplot(grid[0, :])
    title_ax.axis("off")
    title_ax.text(
        0,
        0.78,
        "Sriwijaya Air Mobile Review Sentiment Analysis",
        fontsize=25,
        fontweight="bold",
        color="#212529",
        ha="left",
    )
    title_ax.text(
        0,
        0.38,
        "End-to-end NLP pipeline from Google Play reviews to model evaluation and user insight.",
        fontsize=13,
        color="#495057",
        ha="left",
    )

    cards = [
        ("Reviews Used", f"{len(df):,}", "Clean review rows after preprocessing", 22),
        ("Best Model", "Logistic\nRegression", "Selected by macro F1-score", 17),
        ("Accuracy", format_pct(float(best["accuracy"])), "Held-out test set", 22),
        ("Macro F1", format_pct(float(best["macro_f1"])), "Balances all sentiment classes", 22),
    ]
    for idx, (label, value, subtext, value_size) in enumerate(cards):
        ax = fig.add_subplot(grid[1, idx])
        ax.set_facecolor("#f8f9fa")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#dee2e6")
        ax.text(0.08, 0.72, label, transform=ax.transAxes, fontsize=11, color="#6c757d")
        ax.text(
            0.08,
            0.43,
            value,
            transform=ax.transAxes,
            fontsize=value_size,
            fontweight="bold",
            color="#212529",
            linespacing=0.95,
        )
        ax.text(0.08, 0.18, subtext, transform=ax.transAxes, fontsize=9.5, color="#6c757d")

    dist_ax = fig.add_subplot(grid[2, :2])
    bars = dist_ax.bar(
        counts.index,
        counts.values,
        color=[LABEL_COLORS[label] for label in LABEL_ORDER],
    )
    dist_ax.set_title("Sentiment Distribution", fontsize=13, fontweight="bold")
    dist_ax.set_ylabel("Reviews")
    dist_ax.bar_label(bars, padding=3)
    dist_ax.spines[["top", "right"]].set_visible(False)

    perf_ax = fig.add_subplot(grid[2, 2:])
    values = [best["accuracy"], best["macro_f1"], best["weighted_f1"]]
    bars = perf_ax.barh(["Accuracy", "Macro F1", "Weighted F1"], values, color=["#457b9d", "#f4a261", "#2a9d8f"])
    perf_ax.set_title("Best Model Scores", fontsize=13, fontweight="bold")
    perf_ax.set_xlim(0, 1)
    perf_ax.bar_label(bars, labels=[format_pct(v) for v in values], padding=4)
    perf_ax.spines[["top", "right"]].set_visible(False)

    add_caption(
        fig,
        "Evidence: rating-derived sentiment labels, TF-IDF n-gram features, oversampled training split, and held-out evaluation.",
    )
    save_fig(fig, "01_case_study_overview.png")


def plot_data_evidence(df: pd.DataFrame) -> None:
    rating_counts = df["score"].value_counts().sort_index()
    sentiment_counts = df["sentiment"].value_counts().reindex(LABEL_ORDER, fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")

    axes[0].bar(rating_counts.index.astype(str), rating_counts.values, color="#457b9d")
    axes[0].set_title("Google Play Rating Distribution", fontweight="bold")
    axes[0].set_xlabel("Rating score")
    axes[0].set_ylabel("Reviews")
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].pie(
        sentiment_counts.values,
        labels=[label.title() for label in LABEL_ORDER],
        autopct="%1.1f%%",
        colors=[LABEL_COLORS[label] for label in LABEL_ORDER],
        startangle=90,
        textprops={"fontsize": 10},
    )
    axes[1].set_title("Weak Sentiment Labels from Rating", fontweight="bold")

    add_caption(
        fig,
        "Evidence: labels are mapped from rating scores: 1-2 negative, 3 neutral, and 4-5 positive.",
    )
    save_fig(fig, "02_data_label_evidence.png")


def plot_model_evidence(metrics: pd.DataFrame) -> None:
    metrics = metrics.copy()
    metrics["model_label"] = metrics["model"].map(MODEL_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), facecolor="white")

    x = np.arange(len(metrics))
    width = 0.25
    axes[0].bar(x - width, metrics["accuracy"], width, label="Accuracy", color="#457b9d")
    axes[0].bar(x, metrics["macro_f1"], width, label="Macro F1", color="#f4a261")
    axes[0].bar(x + width, metrics["weighted_f1"], width, label="Weighted F1", color="#2a9d8f")
    axes[0].set_title("Model Comparison", fontweight="bold")
    axes[0].set_xticks(x, metrics["model_label"], rotation=0)
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc="lower right")
    axes[0].spines[["top", "right"]].set_visible(False)

    best = metrics.iloc[0]
    class_scores = pd.Series(
        {
            "Negative": best["negative_f1"],
            "Neutral": best["neutral_f1"],
            "Positive": best["positive_f1"],
        }
    )
    colors = [LABEL_COLORS["negative"], LABEL_COLORS["neutral"], LABEL_COLORS["positive"]]
    bars = axes[1].bar(class_scores.index, class_scores.values, color=colors)
    axes[1].set_title("Class-level F1 of Best Model", fontweight="bold")
    axes[1].set_ylim(0, 1)
    axes[1].bar_label(bars, labels=[format_pct(v) for v in class_scores.values], padding=3)
    axes[1].spines[["top", "right"]].set_visible(False)

    add_caption(
        fig,
        "Evidence: Logistic Regression performs best overall, while the neutral class remains difficult because it has fewer and more ambiguous examples.",
    )
    save_fig(fig, "03_model_performance_evidence.png")


def plot_confusion_matrix_evidence() -> None:
    cm = pd.read_csv(REPORT_DIR / "confusion_matrix_logistic_regression.csv", index_col=0)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")
    image = ax.imshow(cm.values, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Best Model Confusion Matrix", fontsize=15, fontweight="bold")
    ax.set_xlabel("Predicted sentiment")
    ax.set_ylabel("Actual sentiment")
    ax.set_xticks(np.arange(len(LABEL_ORDER)), [label.title() for label in LABEL_ORDER])
    ax.set_yticks(np.arange(len(LABEL_ORDER)), [label.title() for label in LABEL_ORDER])

    threshold = cm.values.max() / 2
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            color = "white" if cm.values[row, col] > threshold else "#212529"
            ax.text(col, row, int(cm.values[row, col]), ha="center", va="center", color=color, fontsize=12)

    add_caption(
        fig,
        "Evidence: the model identifies negative and positive reviews well; neutral reviews are often close to complaint or praise language.",
    )
    save_fig(fig, "04_confusion_matrix_evidence.png")


def top_negative_terms(df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    tokens: list[str] = []
    negative_text = df.loc[df["sentiment"] == "negative", "clean_text"].str.cat(sep=" ")
    for token in negative_text.split():
        token = token.strip().lower()
        if len(token) > 2 and token not in GENERIC_TERMS:
            tokens.append(token)

    return pd.DataFrame(Counter(tokens).most_common(top_n), columns=["term", "frequency"])


def plot_user_pain_points(df: pd.DataFrame) -> None:
    terms = top_negative_terms(df)
    terms = terms.sort_values("frequency")

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    bars = ax.barh(terms["term"], terms["frequency"], color="#d1495b")
    ax.set_title("Most Frequent Negative Review Signals", fontsize=15, fontweight="bold")
    ax.set_xlabel("Term frequency in negative reviews")
    ax.bar_label(bars, padding=4)
    ax.spines[["top", "right"]].set_visible(False)

    add_caption(
        fig,
        "Evidence: negative reviews repeatedly mention promo, ticketing, member features, loading, opening the app, and check-in flow.",
    )
    save_fig(fig, "05_user_pain_points.png")


def plot_sentiment_trend(df: pd.DataFrame) -> None:
    trend_df = df.dropna(subset=["at"]).copy()
    trend_df["month"] = trend_df["at"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        trend_df.groupby(["month", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=LABEL_ORDER, fill_value=0)
        .sort_index()
    )
    monthly = monthly[monthly.sum(axis=1) >= 3]

    fig, ax = plt.subplots(figsize=(12, 5.5), facecolor="white")
    for label in LABEL_ORDER:
        ax.plot(
            monthly.index,
            monthly[label],
            marker="o",
            linewidth=2,
            label=label.title(),
            color=LABEL_COLORS[label],
        )
    ax.set_title("Sentiment Trend by Review Month", fontsize=15, fontweight="bold")
    ax.set_xlabel("Review month")
    ax.set_ylabel("Reviews")
    ax.legend(loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate(rotation=35)

    add_caption(
        fig,
        "Evidence: timeline view helps recruiters see that the analysis can be extended into monitoring product sentiment over time.",
    )
    save_fig(fig, "06_sentiment_trend_by_month.png")


def select_review_examples(df: pd.DataFrame) -> pd.DataFrame:
    query_plan = {
        "negative": ["aplikasi terburuk", "detail penerbangan error"],
        "neutral": ["mohon ditambahkan", "lumayan bisa buat check in"],
        "positive": ["mudah digunakan", "sangat membantu sekali"],
    }

    selected_rows = []
    used_indexes = set()
    for sentiment, queries in query_plan.items():
        subset = df[
            (df["sentiment"] == sentiment)
            & (df["predicted_sentiment"] == sentiment)
        ].copy()
        for query in queries:
            matches = subset[
                subset["content"].str.contains(query, case=False, na=False)
                | subset["clean_text"].str.contains(query, case=False, na=False)
            ]
            if not matches.empty:
                row = matches.iloc[0]
                if row.name not in used_indexes:
                    selected_rows.append(row)
                    used_indexes.add(row.name)

        if len([r for r in selected_rows if r["sentiment"] == sentiment]) < 2:
            fallback = subset[subset["clean_text"].str.len().between(12, 130)].head(2)
            for _, row in fallback.iterrows():
                if row.name not in used_indexes:
                    selected_rows.append(row)
                    used_indexes.add(row.name)
                if len([r for r in selected_rows if r["sentiment"] == sentiment]) >= 2:
                    break

    examples = pd.DataFrame(selected_rows)
    return examples[["score", "sentiment", "predicted_sentiment", "content", "clean_text"]]


def sanitize_review_text(text: str) -> str:
    text = " ".join(str(text).split())
    return "".join(ch for ch in text if 32 <= ord(ch) <= 126)


def plot_review_examples(df: pd.DataFrame) -> None:
    examples = select_review_examples(df)
    fig, ax = plt.subplots(figsize=(14, 6.8), facecolor="white")
    ax.axis("off")
    ax.set_title("Example Reviews Correctly Classified by the Best Model", fontsize=15, fontweight="bold", pad=18)

    rows = []
    for _, row in examples.iterrows():
        review_text = sanitize_review_text(row["content"]) or row["clean_text"]
        rows.append(
            [
                int(row["score"]),
                row["sentiment"].title(),
                row["predicted_sentiment"].title(),
                "\n".join(textwrap.wrap(review_text, width=76)),
            ]
        )

    table = ax.table(
        cellText=rows,
        colLabels=["Rating", "Label", "Prediction", "Review Evidence"],
        cellLoc="left",
        colLoc="left",
        colWidths=[0.08, 0.12, 0.15, 0.65],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 2.7)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dee2e6")
        if row == 0:
            cell.set_facecolor("#212529")
            cell.set_text_props(color="white", weight="bold")
        else:
            sentiment = rows[row - 1][1].lower()
            if col in {1, 2}:
                cell.set_facecolor(LABEL_COLORS[sentiment])
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor("#ffffff")

    add_caption(
        fig,
        "Evidence: sample-level outputs show how review text becomes sentiment predictions that can be inspected by humans.",
    )
    save_fig(fig, "07_review_examples.png")


def write_portfolio_markdown(summary: dict, metrics: pd.DataFrame) -> None:
    best = metrics.iloc[0]
    text = f"""# Portfolio Evidence Pack

Gunakan file PNG di folder ini sebagai visual evidence untuk website portfolio.

## Recommended Website Order

1. `01_case_study_overview.png` - ringkasan problem, dataset, model terbaik, dan skor utama.
2. `02_data_label_evidence.png` - bukti sumber label dari rating Google Play.
3. `03_model_performance_evidence.png` - perbandingan model dan F1-score per kelas.
4. `04_confusion_matrix_evidence.png` - bukti evaluasi prediksi pada test set.
5. `05_user_pain_points.png` - insight kata/isu yang sering muncul di review negatif.
6. `06_sentiment_trend_by_month.png` - gambaran potensi monitoring sentimen dari waktu ke waktu.
7. `07_review_examples.png` - contoh review yang diklasifikasikan model.

## Copy for Portfolio Website

Saya membangun pipeline NLP end-to-end untuk menganalisis ulasan aplikasi Sriwijaya Air Mobile di Google Play Store. Data review diproses melalui cleaning dan feature engineering berbasis TF-IDF n-gram, lalu dibandingkan menggunakan Logistic Regression, Linear SVM, dan Naive Bayes. Model terbaik adalah {MODEL_LABELS.get(summary["best_model"], summary["best_model"])} dengan accuracy {best["accuracy"]:.4f}, macro F1 {best["macro_f1"]:.4f}, dan weighted F1 {best["weighted_f1"]:.4f}.

Project ini juga menghasilkan insight produk: review negatif banyak menyinggung promo, tiket, fitur member, loading, aplikasi sulit dibuka, dan proses check-in. Kelas neutral masih menjadi tantangan karena jumlah datanya kecil dan rating 3 sering berisi campuran keluhan serta apresiasi.
"""
    (PORTFOLIO_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dir()
    df, metrics, summary = load_data()
    plot_case_study_overview(df, metrics, summary)
    plot_data_evidence(df)
    plot_model_evidence(metrics)
    plot_confusion_matrix_evidence()
    plot_user_pain_points(df)
    plot_sentiment_trend(df)
    plot_review_examples(df)
    write_portfolio_markdown(summary, metrics)
    print(f"Portfolio evidence generated in {PORTFOLIO_DIR}")


if __name__ == "__main__":
    main()
