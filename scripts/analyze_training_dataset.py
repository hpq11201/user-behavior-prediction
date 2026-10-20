"""Analyze coverage and label distribution of the training dataset."""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = PROJECT_ROOT / "data" / "processed" / "user_behavior.db"

FEATURE_START = "2025-11-18 00:00:00"
FEATURE_END = "2025-12-10 23:59:59"

LABEL_START = "2025-12-11 00:00:00"
LABEL_END = "2025-12-18 23:59:59"


def main() -> None:
    """Analyze training dataset diagnostics."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        cursor = connection.cursor()

        future_purchase_pairs_query = f"""
        SELECT COUNT(*)
        FROM (
            SELECT DISTINCT
                user_id,
                item_id
            FROM user_behavior
            WHERE behavior_type = 4
              AND timestamp BETWEEN
                  '{LABEL_START}'
                  AND '{LABEL_END}'
        )
        """

        cursor.execute(future_purchase_pairs_query)
        future_purchase_pairs = cursor.fetchone()[0]

        covered_future_purchase_pairs_query = f"""
        SELECT COUNT(*)
        FROM (
            SELECT DISTINCT
                future.user_id,
                future.item_id
            FROM user_behavior AS future

            INNER JOIN (
                SELECT DISTINCT
                    user_id,
                    item_id
                FROM user_behavior
                WHERE timestamp BETWEEN
                    '{FEATURE_START}'
                    AND '{FEATURE_END}'
            ) AS historical
                ON historical.user_id = future.user_id
                AND historical.item_id = future.item_id

            WHERE future.behavior_type = 4
              AND future.timestamp BETWEEN
                  '{LABEL_START}'
                  AND '{LABEL_END}'
        )
        """

        cursor.execute(covered_future_purchase_pairs_query)
        covered_future_purchase_pairs = cursor.fetchone()[0]

        candidate_coverage = (
            100.0 * covered_future_purchase_pairs / future_purchase_pairs
            if future_purchase_pairs
            else 0.0
        )

        daily_positive_query = f"""
        SELECT
            DATE(future.timestamp) AS purchase_date,
            COUNT(
                DISTINCT
                future.user_id || '-' || future.item_id
            ) AS positive_pairs

        FROM user_behavior AS future

        INNER JOIN (
            SELECT DISTINCT
                user_id,
                item_id
            FROM user_behavior
            WHERE timestamp BETWEEN
                '{FEATURE_START}'
                AND '{FEATURE_END}'
        ) AS historical
            ON historical.user_id = future.user_id
            AND historical.item_id = future.item_id

        WHERE future.behavior_type = 4
          AND future.timestamp BETWEEN
              '{LABEL_START}'
              AND '{LABEL_END}'

        GROUP BY DATE(future.timestamp)
        ORDER BY purchase_date
        """

        cursor.execute(daily_positive_query)
        daily_positive_results = cursor.fetchall()

        total_training_positive_query = """
        SELECT COUNT(*)
        FROM training_dataset
        WHERE label_purchase = 1
        """

        cursor.execute(total_training_positive_query)
        training_positive_samples = cursor.fetchone()[0]

    print()
    print("=" * 60)
    print("TRAINING DATASET DIAGNOSTICS")
    print("=" * 60)

    print(f"Future purchase pairs: " f"{future_purchase_pairs:,}")

    print("Covered future purchase pairs: " f"{covered_future_purchase_pairs:,}")

    print(f"Candidate coverage: " f"{candidate_coverage:.2f}%")

    print("Training positive samples: " f"{training_positive_samples:,}")

    print()
    print("Daily covered positive pairs:")

    for (
        purchase_date,
        positive_pairs,
    ) in daily_positive_results:
        print(f"{purchase_date}: " f"{positive_pairs:,}")


if __name__ == "__main__":
    main()
