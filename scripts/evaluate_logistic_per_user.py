"""Evaluate per-user Top-K recommendation performance."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCORE_PATH = PROJECT_ROOT / "data" / "processed" / "logistic_validation_scores.parquet"

TOP_K_VALUES = [
    1,
    5,
    10,
]


def evaluate_per_user_top_k(
    scores: pd.DataFrame,
    top_k: int,
) -> tuple[int, int, float, float, float]:
    """Evaluate per-user Top-K ranking performance."""
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

    top_k_rows = (
        ranked.groupby(
            "user_id",
            sort=False,
        )
        .head(top_k)
        .copy()
    )

    total_users = int(scores["user_id"].nunique())

    users_with_purchase = int(
        scores.loc[
            scores["label_purchase"] == 1,
            "user_id",
        ].nunique()
    )

    hits = int(top_k_rows["label_purchase"].sum())

    recommended_rows = len(top_k_rows)

    precision_at_k = hits / recommended_rows if recommended_rows else 0.0

    total_positive_pairs = int(scores["label_purchase"].sum())

    recall_at_k = hits / total_positive_pairs if total_positive_pairs else 0.0

    hit_users = int(
        top_k_rows.loc[
            top_k_rows["label_purchase"] == 1,
            "user_id",
        ].nunique()
    )

    hit_rate_at_k = hit_users / users_with_purchase if users_with_purchase else 0.0

    return (
        total_users,
        hit_users,
        precision_at_k,
        recall_at_k,
        hit_rate_at_k,
    )


def main() -> None:
    """Run per-user Top-K evaluation."""
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

    total_users = int(scores["user_id"].nunique())

    users_with_purchase = int(
        scores.loc[
            scores["label_purchase"] == 1,
            "user_id",
        ].nunique()
    )

    total_positive_pairs = int(scores["label_purchase"].sum())

    print()
    print("=" * 80)
    print("PER-USER TOP-K EVALUATION")
    print("=" * 80)

    print(f"Total users: " f"{total_users:,}")

    print(f"Users with at least one purchase: " f"{users_with_purchase:,}")

    print(f"Total positive user-item pairs: " f"{total_positive_pairs:,}")

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

    for top_k in TOP_K_VALUES:
        (
            _,
            hit_users,
            precision_at_k,
            recall_at_k,
            hit_rate_at_k,
        ) = evaluate_per_user_top_k(
            scores,
            top_k,
        )

        print(
            f"{top_k:<10}"
            f"{hit_users:>12,}"
            f"{precision_at_k:>16.4f}"
            f"{recall_at_k:>14.4f}"
            f"{hit_rate_at_k:>16.4f}"
        )

    print()
    print("Per-user Top-K evaluation completed.")


if __name__ == "__main__":
    main()
