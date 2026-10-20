"""Build user-item level behavioral features from SQLite."""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = PROJECT_ROOT / "data" / "processed" / "user_behavior.db"

OUTPUT_TABLE = "user_item_features"


def build_user_item_features(
    connection: sqlite3.Connection,
) -> None:
    """Build user-item level feature table.

    Args:
        connection: SQLite database connection.
    """
    cursor = connection.cursor()

    cursor.execute(f"""
        DROP TABLE IF EXISTS {OUTPUT_TABLE}
        """)

    cursor.execute(f"""
        CREATE TABLE {OUTPUT_TABLE} AS
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

            SUM(
                CASE
                    WHEN behavior_type = 4 THEN 1
                    ELSE 0
                END
            ) AS purchase_count,

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

            MIN(
                CASE
                    WHEN behavior_type = 4 THEN timestamp
                END
            ) AS first_purchase_time,

            COUNT(
                DISTINCT DATE(timestamp)
            ) AS active_days

        FROM user_behavior

        GROUP BY
            user_id,
            item_id
        """)

    connection.commit()


def create_feature_indexes(
    connection: sqlite3.Connection,
) -> None:
    """Create indexes for user-item feature analysis.

    Args:
        connection: SQLite database connection.
    """
    cursor = connection.cursor()

    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS
        idx_features_user
        ON {OUTPUT_TABLE}(user_id)
        """)

    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS
        idx_features_item
        ON {OUTPUT_TABLE}(item_id)
        """)

    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS
        idx_features_user_item
        ON {OUTPUT_TABLE}(user_id, item_id)
        """)

    connection.commit()


def validate_features(
    connection: sqlite3.Connection,
) -> None:
    """Validate the generated feature table.

    Args:
        connection: SQLite database connection.
    """
    cursor = connection.cursor()

    cursor.execute(f"""
        SELECT COUNT(*)
        FROM {OUTPUT_TABLE}
        """)

    feature_rows = cursor.fetchone()[0]

    cursor.execute(f"""
        SELECT
            SUM(total_behaviors),
            SUM(view_count),
            SUM(favorite_count),
            SUM(cart_count),
            SUM(purchase_count)
        FROM {OUTPUT_TABLE}
        """)

    (
        total_behaviors,
        total_views,
        total_favorites,
        total_carts,
        total_purchases,
    ) = cursor.fetchone()

    cursor.execute(f"""
        SELECT *
        FROM {OUTPUT_TABLE}
        ORDER BY total_behaviors DESC
        LIMIT 5
        """)

    top_rows = cursor.fetchall()

    print()
    print("=" * 60)
    print("USER-ITEM FEATURE VALIDATION")
    print("=" * 60)

    print(f"Feature rows: " f"{feature_rows:,}")

    print(f"Total behaviors: " f"{total_behaviors:,}")

    print(f"Views: " f"{total_views:,}")

    print(f"Favorites: " f"{total_favorites:,}")

    print(f"Carts: " f"{total_carts:,}")

    print(f"Purchases: " f"{total_purchases:,}")

    print()
    print("Top 5 user-item pairs:")

    for row in top_rows:
        print(row)


def main() -> None:
    """Build and validate user-item feature table."""
    print("Opening SQLite database...")

    with sqlite3.connect(DATABASE_PATH) as connection:
        print("Building user-item feature table...")

        build_user_item_features(connection)

        print("Creating feature indexes...")

        create_feature_indexes(connection)

        validate_features(connection)

    print()
    print("User-item feature table completed.")


if __name__ == "__main__":
    main()
