"""Build an earlier modeling dataset for model training."""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = PROJECT_ROOT / "data" / "processed" / "user_behavior.db"

FEATURE_START = "2025-11-18 00:00:00"
FEATURE_END = "2025-12-03 23:59:59"

LABEL_START = "2025-12-04 00:00:00"
LABEL_END = "2025-12-10 23:59:59"

TOP_ITEMS_PER_CATEGORY = 10

OUTPUT_TABLE = "training_modeling_dataset"


def build_training_dataset(
    connection: sqlite3.Connection,
) -> None:
    """Build the earlier training modeling dataset."""
    cursor = connection.cursor()

    print("Step 1: Building training historical pairs...")

    cursor.execute("""
        DROP TABLE IF EXISTS train_historical_pairs
        """)

    cursor.execute(f"""
        CREATE TABLE train_historical_pairs AS
        SELECT DISTINCT
            user_id,
            item_id
        FROM user_behavior
        WHERE timestamp BETWEEN
            '{FEATURE_START}'
            AND '{FEATURE_END}'
        """)

    connection.commit()

    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_train_historical_pairs
        ON train_historical_pairs(user_id, item_id)
        """)

    connection.commit()

    print("Step 2: Building training user categories...")

    cursor.execute("""
        DROP TABLE IF EXISTS train_user_categories
        """)

    cursor.execute(f"""
        CREATE TABLE train_user_categories AS
        SELECT DISTINCT
            user_id,
            category_id
        FROM user_behavior
        WHERE timestamp BETWEEN
            '{FEATURE_START}'
            AND '{FEATURE_END}'
        """)

    connection.commit()

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS
        idx_train_user_categories
        ON train_user_categories(category_id, user_id)
        """)

    connection.commit()

    print("Step 3: Building training item popularity...")

    cursor.execute("""
        DROP TABLE IF EXISTS train_item_popularity
        """)

    cursor.execute(f"""
        CREATE TABLE train_item_popularity AS
        SELECT
            category_id,
            item_id,
            COUNT(*) AS interaction_count,
            COUNT(DISTINCT user_id) AS user_count
        FROM user_behavior
        WHERE timestamp BETWEEN
            '{FEATURE_START}'
            AND '{FEATURE_END}'
        GROUP BY
            category_id,
            item_id
        """)

    connection.commit()

    print("Step 4: Ranking training items...")

    cursor.execute("""
        DROP TABLE IF EXISTS train_ranked_items
        """)

    cursor.execute("""
        CREATE TABLE train_ranked_items AS
        SELECT
            category_id,
            item_id,
            interaction_count,
            user_count,
            ROW_NUMBER() OVER (
                PARTITION BY category_id
                ORDER BY
                    user_count DESC,
                    interaction_count DESC,
                    item_id
            ) AS item_rank
        FROM train_item_popularity
        """)

    connection.commit()

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS
        idx_train_ranked_items
        ON train_ranked_items(category_id, item_rank)
        """)

    connection.commit()

    print("Step 5: Building Top-10 training candidate set...")

    cursor.execute("""
        DROP TABLE IF EXISTS train_candidate_set
        """)

    cursor.execute(f"""
        CREATE TABLE train_candidate_set AS

        SELECT
            user_id,
            item_id
        FROM train_historical_pairs

        UNION

        SELECT
            train_user_categories.user_id,
            train_ranked_items.item_id
        FROM train_user_categories
        INNER JOIN train_ranked_items
            ON train_ranked_items.category_id
                = train_user_categories.category_id
        WHERE train_ranked_items.item_rank
            <= {TOP_ITEMS_PER_CATEGORY}
        """)

    connection.commit()

    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_train_candidate_set
        ON train_candidate_set(user_id, item_id)
        """)

    connection.commit()

    print("Step 6: Building training historical features...")

    cursor.execute("""
        DROP TABLE IF EXISTS train_modeling_features
        """)

    cursor.execute(f"""
        CREATE TABLE train_modeling_features AS
        SELECT
            user_id,
            item_id,
            COUNT(*) AS total_behaviors,

            SUM(
                CASE
                    WHEN behavior_type = 1
                    THEN 1
                    ELSE 0
                END
            ) AS view_count,

            SUM(
                CASE
                    WHEN behavior_type = 2
                    THEN 1
                    ELSE 0
                END
            ) AS favorite_count,

            SUM(
                CASE
                    WHEN behavior_type = 3
                    THEN 1
                    ELSE 0
                END
            ) AS cart_count,

            COUNT(
                DISTINCT DATE(timestamp)
            ) AS active_days,

            MAX(timestamp) AS last_behavior_time

        FROM user_behavior

        WHERE timestamp BETWEEN
            '{FEATURE_START}'
            AND '{FEATURE_END}'

        GROUP BY
            user_id,
            item_id
        """)

    connection.commit()

    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_train_modeling_features
        ON train_modeling_features(user_id, item_id)
        """)

    connection.commit()

    print("Step 7: Building training category recall pairs...")

    cursor.execute("""
        DROP TABLE IF EXISTS train_category_recall
        """)

    cursor.execute(f"""
        CREATE TABLE train_category_recall AS
        SELECT DISTINCT
            train_user_categories.user_id,
            train_ranked_items.item_id
        FROM train_user_categories
        INNER JOIN train_ranked_items
            ON train_ranked_items.category_id
                = train_user_categories.category_id
        WHERE train_ranked_items.item_rank
            <= {TOP_ITEMS_PER_CATEGORY}
        """)

    connection.commit()

    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_train_category_recall
        ON train_category_recall(user_id, item_id)
        """)

    connection.commit()

    print("Step 8: Building training future labels...")

    cursor.execute("""
        DROP TABLE IF EXISTS train_future_purchase_labels
        """)

    cursor.execute(f"""
        CREATE TABLE train_future_purchase_labels AS
        SELECT DISTINCT
            user_id,
            item_id,
            1 AS label_purchase
        FROM user_behavior
        WHERE behavior_type = 4
          AND timestamp BETWEEN
              '{LABEL_START}'
              AND '{LABEL_END}'
        """)

    connection.commit()

    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_train_future_purchase_labels
        ON train_future_purchase_labels(user_id, item_id)
        """)

    connection.commit()

    print("Step 9: Building final training modeling dataset...")

    cursor.execute(f"""
        DROP TABLE IF EXISTS {OUTPUT_TABLE}
        """)

    cursor.execute(f"""
        CREATE TABLE {OUTPUT_TABLE} AS

        SELECT
            candidates.user_id,
            candidates.item_id,

            COALESCE(
                features.total_behaviors,
                0
            ) AS total_behaviors,

            COALESCE(
                features.view_count,
                0
            ) AS view_count,

            COALESCE(
                features.favorite_count,
                0
            ) AS favorite_count,

            COALESCE(
                features.cart_count,
                0
            ) AS cart_count,

            COALESCE(
                features.active_days,
                0
            ) AS active_days,

            CASE
                WHEN features.last_behavior_time IS NOT NULL
                THEN CAST(
                    julianday('{FEATURE_END}')
                    - julianday(features.last_behavior_time)
                    AS INTEGER
                )
                ELSE -1
            END AS days_since_last_behavior,

            CASE
                WHEN historical.user_id IS NOT NULL
                THEN 1
                ELSE 0
            END AS seen_historically,

            CASE
                WHEN category_recall.user_id IS NOT NULL
                THEN 1
                ELSE 0
            END AS from_category_recall,

            COALESCE(
                labels.label_purchase,
                0
            ) AS label_purchase

        FROM train_candidate_set AS candidates

        LEFT JOIN train_modeling_features AS features
            ON features.user_id = candidates.user_id
            AND features.item_id = candidates.item_id

        LEFT JOIN train_historical_pairs AS historical
            ON historical.user_id = candidates.user_id
            AND historical.item_id = candidates.item_id

        LEFT JOIN train_category_recall AS category_recall
            ON category_recall.user_id = candidates.user_id
            AND category_recall.item_id = candidates.item_id

        LEFT JOIN train_future_purchase_labels AS labels
            ON labels.user_id = candidates.user_id
            AND labels.item_id = candidates.item_id
        """)

    connection.commit()

    print("Creating training dataset indexes...")

    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS
        idx_training_modeling_dataset_label
        ON {OUTPUT_TABLE}(label_purchase)
        """)

    connection.commit()


def validate_training_dataset(
    connection: sqlite3.Connection,
) -> None:
    """Validate the earlier training dataset."""
    cursor = connection.cursor()

    cursor.execute(f"""
        SELECT COUNT(*)
        FROM {OUTPUT_TABLE}
        """)
    total_samples = int(cursor.fetchone()[0])

    cursor.execute(f"""
        SELECT COUNT(*)
        FROM {OUTPUT_TABLE}
        WHERE label_purchase = 1
        """)
    positive_samples = int(cursor.fetchone()[0])

    negative_samples = total_samples - positive_samples

    positive_rate = 100.0 * positive_samples / total_samples if total_samples else 0.0

    cursor.execute(f"""
        SELECT COUNT(*)
        FROM {OUTPUT_TABLE}
        WHERE seen_historically = 0
          AND from_category_recall = 1
          AND label_purchase = 1
        """)
    category_only_positive = int(cursor.fetchone()[0])

    print()
    print("=" * 60)
    print("TRAINING MODELING DATASET VALIDATION")
    print("=" * 60)

    print(f"Feature window: " f"{FEATURE_START} -> {FEATURE_END}")

    print(f"Label window: " f"{LABEL_START} -> {LABEL_END}")

    print()

    print(f"Total samples: " f"{total_samples:,}")

    print(f"Positive samples: " f"{positive_samples:,}")

    print(f"Negative samples: " f"{negative_samples:,}")

    print(f"Positive rate: " f"{positive_rate:.4f}%")

    print("Category-only positive samples: " f"{category_only_positive:,}")


def main() -> None:
    """Build and validate the earlier training dataset."""
    print("Opening SQLite database...")

    with sqlite3.connect(DATABASE_PATH) as connection:
        build_training_dataset(connection)

        validate_training_dataset(connection)

    print()
    print("Training modeling dataset completed.")


if __name__ == "__main__":
    main()
