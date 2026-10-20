"""Build a leakage-safe training dataset for purchase prediction."""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = PROJECT_ROOT / "data" / "processed" / "user_behavior.db"

FEATURE_START = "2025-11-18 00:00:00"
FEATURE_END = "2025-12-10 23:59:59"

LABEL_START = "2025-12-11 00:00:00"
LABEL_END = "2025-12-18 23:59:59"

OUTPUT_TABLE = "training_dataset"


def build_training_dataset(
    connection: sqlite3.Connection,
) -> None:
    """Build training samples using past behavior and future labels.

    Args:
        connection: SQLite database connection.
    """
    cursor = connection.cursor()

    cursor.execute(f"""
        DROP TABLE IF EXISTS {OUTPUT_TABLE}
        """)

    cursor.execute(f"""
        CREATE TABLE {OUTPUT_TABLE} AS

        WITH historical_features AS (
            SELECT
                user_id,
                item_id,

                COUNT(*) AS total_behaviors,

                SUM(
                    CASE
                        WHEN behavior_type = 1 THEN 1
                        ELSE 0
                    END
                ) AS view_count,

                SUM(
                    CASE
                        WHEN behavior_type = 2 THEN 1
                        ELSE 0
                    END
                ) AS favorite_count,

                SUM(
                    CASE
                        WHEN behavior_type = 3 THEN 1
                        ELSE 0
                    END
                ) AS cart_count,

                COUNT(
                    DISTINCT DATE(timestamp)
                ) AS active_days,

                MIN(timestamp) AS first_behavior_time,
                MAX(timestamp) AS last_behavior_time,

                MIN(
                    CASE
                        WHEN behavior_type = 1 THEN timestamp
                    END
                ) AS first_view_time,

                MAX(
                    CASE
                        WHEN behavior_type = 1 THEN timestamp
                    END
                ) AS last_view_time,

                MIN(
                    CASE
                        WHEN behavior_type = 3 THEN timestamp
                    END
                ) AS first_cart_time,

                MAX(
                    CASE
                        WHEN behavior_type = 3 THEN timestamp
                    END
                ) AS last_cart_time

            FROM user_behavior

            WHERE timestamp
                BETWEEN '{FEATURE_START}'
                AND '{FEATURE_END}'

            GROUP BY
                user_id,
                item_id
        ),

        future_purchase_labels AS (
            SELECT DISTINCT
                user_id,
                item_id,
                1 AS label_purchase

            FROM user_behavior

            WHERE timestamp
                BETWEEN '{LABEL_START}'
                AND '{LABEL_END}'

            AND behavior_type = 4
        )

        SELECT
            features.user_id,
            features.item_id,

            features.total_behaviors,
            features.view_count,
            features.favorite_count,
            features.cart_count,
            features.active_days,

            features.first_behavior_time,
            features.last_behavior_time,
            features.first_view_time,
            features.last_view_time,
            features.first_cart_time,
            features.last_cart_time,

            CAST(
                julianday('{FEATURE_END}')
                - julianday(
                    features.last_behavior_time
                )
                AS INTEGER
            ) AS days_since_last_behavior,

            CAST(
                julianday('{FEATURE_END}')
                - julianday(
                    features.first_behavior_time
                )
                AS INTEGER
            ) AS behavior_span_days,

            COALESCE(
                labels.label_purchase,
                0
            ) AS label_purchase

        FROM historical_features AS features

        LEFT JOIN future_purchase_labels AS labels
            ON labels.user_id = features.user_id
            AND labels.item_id = features.item_id
        """)

    connection.commit()


def create_indexes(
    connection: sqlite3.Connection,
) -> None:
    """Create indexes for training dataset access.

    Args:
        connection: SQLite database connection.
    """
    cursor = connection.cursor()

    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS
        idx_training_user
        ON {OUTPUT_TABLE}(user_id)
        """)

    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS
        idx_training_item
        ON {OUTPUT_TABLE}(item_id)
        """)

    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS
        idx_training_label
        ON {OUTPUT_TABLE}(label_purchase)
        """)

    connection.commit()


def validate_training_dataset(
    connection: sqlite3.Connection,
) -> None:
    """Validate training sample counts and label balance.

    Args:
        connection: SQLite database connection.
    """
    cursor = connection.cursor()

    cursor.execute(f"""
        SELECT COUNT(*)
        FROM {OUTPUT_TABLE}
        """)

    total_samples = cursor.fetchone()[0]

    cursor.execute(f"""
        SELECT COUNT(*)
        FROM {OUTPUT_TABLE}
        WHERE label_purchase = 1
        """)

    positive_samples = cursor.fetchone()[0]

    negative_samples = total_samples - positive_samples

    positive_rate = 100.0 * positive_samples / total_samples if total_samples else 0.0

    cursor.execute(f"""
        SELECT
            user_id,
            item_id,
            view_count,
            favorite_count,
            cart_count,
            active_days,
            days_since_last_behavior,
            label_purchase
        FROM {OUTPUT_TABLE}
        ORDER BY label_purchase DESC,
                 total_behaviors DESC
        LIMIT 10
        """)

    sample_rows = cursor.fetchall()

    print()
    print("=" * 60)
    print("TRAINING DATASET VALIDATION")
    print("=" * 60)

    print(f"Feature window: " f"{FEATURE_START} -> {FEATURE_END}")

    print(f"Label window: " f"{LABEL_START} -> {LABEL_END}")

    print()

    print(f"Total samples: " f"{total_samples:,}")

    print(f"Positive samples: " f"{positive_samples:,}")

    print(f"Negative samples: " f"{negative_samples:,}")

    print(f"Positive rate: " f"{positive_rate:.4f}%")

    print()
    print("Sample rows:")

    for row in sample_rows:
        print(row)


def main() -> None:
    """Build and validate the purchase prediction dataset."""
    print("Opening SQLite database...")

    with sqlite3.connect(DATABASE_PATH) as connection:
        print("Building training dataset...")

        build_training_dataset(connection)

        print("Creating training indexes...")

        create_indexes(connection)

        validate_training_dataset(connection)

    print()
    print("Training dataset completed.")


if __name__ == "__main__":
    main()
