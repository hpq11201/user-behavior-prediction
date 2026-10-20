"""Build the final modeling dataset from Top-10 candidates."""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = PROJECT_ROOT / "data" / "processed" / "user_behavior.db"

FEATURE_START = "2025-11-18 00:00:00"
FEATURE_END = "2025-12-10 23:59:59"

LABEL_START = "2025-12-11 00:00:00"
LABEL_END = "2025-12-18 23:59:59"

CANDIDATE_TABLE = "candidate_set_k10"
OUTPUT_TABLE = "modeling_dataset"


def build_modeling_dataset(
    connection: sqlite3.Connection,
) -> None:
    """Build the final modeling dataset."""
    cursor = connection.cursor()

    print("Dropping old modeling dataset...")

    cursor.execute(f"""
        DROP TABLE IF EXISTS {OUTPUT_TABLE}
        """)

    connection.commit()

    print("Building historical user-item features...")

    cursor.execute("""
        DROP TABLE IF EXISTS modeling_features
        """)

    cursor.execute(f"""
        CREATE TABLE modeling_features AS
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

    print("Creating feature index...")

    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_modeling_features_user_item
        ON modeling_features(user_id, item_id)
        """)

    connection.commit()

    print("Building category recall pairs...")

    cursor.execute("""
        DROP TABLE IF EXISTS category_recall_k10
        """)

    cursor.execute("""
        CREATE TABLE category_recall_k10 AS
        SELECT DISTINCT
            user_categories.user_id,
            ranked_items.item_id

        FROM user_categories

        INNER JOIN ranked_items
            ON ranked_items.category_id
                = user_categories.category_id

        WHERE ranked_items.item_rank <= 10
        """)

    connection.commit()

    print("Creating category recall index...")

    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_category_recall_k10_user_item
        ON category_recall_k10(user_id, item_id)
        """)

    connection.commit()

    print("Building future purchase labels...")

    cursor.execute("""
        DROP TABLE IF EXISTS future_purchase_labels
        """)

    cursor.execute(f"""
        CREATE TABLE future_purchase_labels AS
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

    print("Creating label index...")

    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_future_purchase_labels_user_item
        ON future_purchase_labels(user_id, item_id)
        """)

    connection.commit()

    print("Building final modeling dataset...")

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

        FROM {CANDIDATE_TABLE} AS candidates

        LEFT JOIN modeling_features AS features
            ON features.user_id = candidates.user_id
            AND features.item_id = candidates.item_id

        LEFT JOIN historical_pairs AS historical
            ON historical.user_id = candidates.user_id
            AND historical.item_id = candidates.item_id

        LEFT JOIN category_recall_k10 AS category_recall
            ON category_recall.user_id = candidates.user_id
            AND category_recall.item_id = candidates.item_id

        LEFT JOIN future_purchase_labels AS labels
            ON labels.user_id = candidates.user_id
            AND labels.item_id = candidates.item_id
        """)

    connection.commit()

    print("Creating modeling dataset indexes...")

    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS
        idx_modeling_dataset_user
        ON {OUTPUT_TABLE}(user_id)
        """)

    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS
        idx_modeling_dataset_item
        ON {OUTPUT_TABLE}(item_id)
        """)

    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS
        idx_modeling_dataset_label
        ON {OUTPUT_TABLE}(label_purchase)
        """)

    connection.commit()


def validate_modeling_dataset(
    connection: sqlite3.Connection,
) -> None:
    """Validate the final modeling dataset."""
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
        WHERE seen_historically = 1
        """)
    historical_candidates = int(cursor.fetchone()[0])

    cursor.execute(f"""
        SELECT COUNT(*)
        FROM {OUTPUT_TABLE}
        WHERE from_category_recall = 1
        """)
    category_candidates = int(cursor.fetchone()[0])

    cursor.execute(f"""
        SELECT COUNT(*)
        FROM {OUTPUT_TABLE}
        WHERE seen_historically = 0
          AND from_category_recall = 1
        """)
    category_only_candidates = int(cursor.fetchone()[0])

    cursor.execute(f"""
        SELECT COUNT(*)
        FROM {OUTPUT_TABLE}
        WHERE label_purchase = 1
          AND seen_historically = 0
          AND from_category_recall = 1
        """)
    category_only_positive = int(cursor.fetchone()[0])

    print()
    print("=" * 60)
    print("MODELING DATASET VALIDATION")
    print("=" * 60)

    print(f"Total samples: " f"{total_samples:,}")

    print(f"Positive samples: " f"{positive_samples:,}")

    print(f"Negative samples: " f"{negative_samples:,}")

    print(f"Positive rate: " f"{positive_rate:.4f}%")

    print()

    print(f"Historical candidates: " f"{historical_candidates:,}")

    print(f"Category recall candidates: " f"{category_candidates:,}")

    print("Category-only candidates: " f"{category_only_candidates:,}")

    print("Category-only positive samples: " f"{category_only_positive:,}")

    print()

    print("Sample positive rows:")

    cursor.execute(f"""
        SELECT
            user_id,
            item_id,
            total_behaviors,
            view_count,
            favorite_count,
            cart_count,
            active_days,
            days_since_last_behavior,
            seen_historically,
            from_category_recall,
            label_purchase

        FROM {OUTPUT_TABLE}

        WHERE label_purchase = 1

        ORDER BY
            total_behaviors DESC

        LIMIT 10
        """)

    for row in cursor.fetchall():
        print(row)


def main() -> None:
    """Build and validate the modeling dataset."""
    print("Opening SQLite database...")

    with sqlite3.connect(DATABASE_PATH) as connection:
        build_modeling_dataset(connection)

        validate_modeling_dataset(connection)

    print()
    print("Modeling dataset completed.")


if __name__ == "__main__":
    main()
