"""Evaluate saved HistGradientBoosting validation scores."""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCORE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hist_gradient_boosting_validation_scores.parquet"
)

THRESHOLDS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
]

GLOBAL_TOP_K_VALUES = [
    100,
    500,
    1000,
    5000,
    10000,
]

USER_TOP_K_VALUES = [
    1,
    5,
    10,
]


def evaluate_thresholds(
    scores: pd.DataFrame,
) -> None:
    """Compare threshold-based classification performance."""
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

        true_positive = int(((predictions == 1) & (y_true == 1)).sum())

        false_positive = int(((predictions == 1) & (y_true == 0)).sum())

        false_negative = int(((predictions == 0) & (y_true == 1)).sum())

        predicted_positive = true_positive + false_positive

        precision = true_positive / predicted_positive if predicted_positive else 0.0

        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative)
            else 0.0
        )

        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        print(
            f"{threshold:<12.2f}"
            f"{predicted_positive:>14,}"
            f"{precision:>12.4f}"
            f"{recall:>12.4f}"
            f"{f1:>12.4f}"
        )


def evaluate_global_top_k(
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

    for top_k in GLOBAL_TOP_K_VALUES:
        actual_k = min(
            top_k,
            len(ranked_scores),
        )

        top_rows = ranked_scores.iloc[:actual_k]

        hits = int(top_rows["label_purchase"].sum())

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


def evaluate_per_user_top_k(
    scores: pd.DataFrame,
) -> None:
    """Evaluate recommendation performance separately for each user."""
    total_users = int(scores["user_id"].nunique())

    users_with_purchase = int(
        scores.loc[
            scores["label_purchase"] == 1,
            "user_id",
        ].nunique()
    )

    total_positive_pairs = int(scores["label_purchase"].sum())

    print()
    print("Sorting predictions by user...")

    ranked = scores.sort_values(
        by=[
            "user_id",
            "prediction_probability",
        ],
        ascending=[
            True,
            False,
        ],
    )

    print()
    print("=" * 90)
    print("PER-USER TOP-K EVALUATION")
    print("=" * 90)

    print(f"Total users: " f"{total_users:,}")

    print("Users with at least one purchase: " f"{users_with_purchase:,}")

    print("Total positive user-item pairs: " f"{total_positive_pairs:,}")

    print()

    header = (
        f"{'Top-K':<10}"
        f"{'Hit users':>12}"
        f"{'Precision@K':>16}"
        f"{'Recall@K':>14}"
        f"{'Hit Rate@K':>16}"
    )

    print(header)
    print("-" * len(header))

    for top_k in USER_TOP_K_VALUES:
        top_rows = ranked.groupby(
            "user_id",
            sort=False,
        ).head(top_k)

        hits = int(top_rows["label_purchase"].sum())

        recommended_rows = len(top_rows)

        precision_at_k = hits / recommended_rows if recommended_rows else 0.0

        recall_at_k = hits / total_positive_pairs if total_positive_pairs else 0.0

        hit_users = int(
            top_rows.loc[
                top_rows["label_purchase"] == 1,
                "user_id",
            ].nunique()
        )

        hit_rate_at_k = hit_users / users_with_purchase if users_with_purchase else 0.0

        print(
            f"{top_k:<10}"
            f"{hit_users:>12,}"
            f"{precision_at_k:>16.4f}"
            f"{recall_at_k:>14.4f}"
            f"{hit_rate_at_k:>16.4f}"
        )


def main() -> None:
    """Evaluate saved HistGradientBoosting predictions."""
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

    evaluate_global_top_k(scores)

    evaluate_per_user_top_k(scores)

    print()
    print("HistGradientBoosting evaluation completed.")


if __name__ == "__main__":
    main()
