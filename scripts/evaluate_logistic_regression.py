"""Evaluate saved logistic regression validation scores."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (  # type: ignore[import-untyped]
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCORE_PATH = PROJECT_ROOT / "data" / "processed" / "logistic_validation_scores.parquet"

THRESHOLDS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
]

TOP_K_VALUES = [
    100,
    500,
    1000,
    5000,
    10000,
]


def evaluate_thresholds(
    scores: pd.DataFrame,
) -> None:
    """Compare precision, recall, and F1 across thresholds."""
    y_true = scores["label_purchase"].to_numpy(dtype=np.int8)

    probabilities = scores["prediction_probability"].to_numpy(dtype=np.float32)

    print()
    print("=" * 75)
    print("THRESHOLD COMPARISON")
    print("=" * 75)

    header = (
        f"{'Threshold':<12}"
        f"{'Predicted +':>14}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
    )

    print(header)
    print("-" * len(header))

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(np.int8)

        predicted_positive = int(predictions.sum())

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        print(
            f"{threshold:<12.2f}"
            f"{predicted_positive:>14,}"
            f"{precision:>12.4f}"
            f"{recall:>12.4f}"
            f"{f1:>12.4f}"
        )


def evaluate_top_k(
    scores: pd.DataFrame,
) -> None:
    """Evaluate global Top-K ranking performance."""
    total_positives = int(scores["label_purchase"].sum())

    total_rows = len(scores)

    baseline_positive_rate = total_positives / total_rows if total_rows else 0.0

    ranked_scores = scores.sort_values(
        by="prediction_probability",
        ascending=False,
    ).reset_index(drop=True)

    print()
    print("=" * 90)
    print("GLOBAL TOP-K EVALUATION")
    print("=" * 90)

    print(f"Total validation rows: " f"{total_rows:,}")

    print(f"Total positives: " f"{total_positives:,}")

    print(f"Baseline positive rate: " f"{baseline_positive_rate:.6f}")

    print()

    header = (
        f"{'Top-K':<10}"
        f"{'Hits':>10}"
        f"{'Precision@K':>16}"
        f"{'Recall@K':>14}"
        f"{'Lift@K':>12}"
    )

    print(header)
    print("-" * len(header))

    for top_k in TOP_K_VALUES:
        actual_k = min(
            top_k,
            len(ranked_scores),
        )

        top_k_rows = ranked_scores.iloc[:actual_k]

        hits = int(top_k_rows["label_purchase"].sum())

        precision_at_k = hits / actual_k if actual_k else 0.0

        recall_at_k = hits / total_positives if total_positives else 0.0

        lift_at_k = (
            precision_at_k / baseline_positive_rate if baseline_positive_rate else 0.0
        )

        print(
            f"{actual_k:<10,}"
            f"{hits:>10,}"
            f"{precision_at_k:>16.4f}"
            f"{recall_at_k:>14.4f}"
            f"{lift_at_k:>12.2f}x"
        )


def show_top_predictions(
    scores: pd.DataFrame,
) -> None:
    """Show highest-scoring user-item pairs."""
    top_predictions = (
        scores.sort_values(
            by="prediction_probability",
            ascending=False,
        )
        .head(20)
        .reset_index(drop=True)
    )

    print()
    print("=" * 75)
    print("TOP 20 PREDICTIONS")
    print("=" * 75)

    print(
        top_predictions[
            [
                "user_id",
                "item_id",
                "label_purchase",
                "prediction_probability",
            ]
        ].to_string(index=False)
    )


def main() -> None:
    """Evaluate saved logistic regression predictions."""
    print(f"Loading validation scores from: " f"{SCORE_PATH}")

    scores = pd.read_parquet(SCORE_PATH)

    print(f"Loaded rows: " f"{len(scores):,}")

    required_columns = {
        "user_id",
        "item_id",
        "label_purchase",
        "prediction_probability",
    }

    missing_columns = required_columns - set(scores.columns)

    if missing_columns:
        raise ValueError("Missing required columns: " f"{sorted(missing_columns)}")

    evaluate_thresholds(scores)

    evaluate_top_k(scores)

    show_top_predictions(scores)

    print()
    print("Logistic regression evaluation completed.")


if __name__ == "__main__":
    main()
