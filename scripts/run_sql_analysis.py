"""Run SQL analysis on the user behavior SQLite database."""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = PROJECT_ROOT / "data" / "processed" / "user_behavior.db"


def main() -> None:
    """Run SQL analysis."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        cursor = connection.cursor()

        print("Preparing index for strict funnel analysis...")

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_behavior_user_item_type_time
            ON user_behavior(
                user_id,
                item_id,
                behavior_type,
                timestamp
            )
            """)

        connection.commit()

        basic_summary_query = """
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT user_id) AS unique_users,
            COUNT(DISTINCT item_id) AS unique_items,
            SUM(
                CASE
                    WHEN behavior_type = 4 THEN 1
                    ELSE 0
                END
            ) AS purchase_rows
        FROM user_behavior
        """

        cursor.execute(basic_summary_query)
        basic_summary = cursor.fetchone()

        behavior_distribution_query = """
        SELECT
            behavior_type,
            COUNT(*) AS behavior_count
        FROM user_behavior
        GROUP BY behavior_type
        ORDER BY behavior_type
        """

        cursor.execute(behavior_distribution_query)
        behavior_distribution = cursor.fetchall()

        strict_funnel_query = """
        WITH view_stage AS (
            SELECT
                user_id,
                item_id,
                MIN(timestamp) AS view_time
            FROM user_behavior
            WHERE behavior_type = 1
            GROUP BY
                user_id,
                item_id
        ),

        cart_stage AS (
            SELECT
                view_stage.user_id,
                view_stage.item_id,
                view_stage.view_time,
                MIN(behavior.timestamp) AS cart_time
            FROM view_stage
            INNER JOIN user_behavior AS behavior
                ON behavior.user_id = view_stage.user_id
                AND behavior.item_id = view_stage.item_id
                AND behavior.behavior_type = 3
                AND behavior.timestamp >= view_stage.view_time
            GROUP BY
                view_stage.user_id,
                view_stage.item_id,
                view_stage.view_time
        ),

        purchase_stage AS (
            SELECT
                cart_stage.user_id,
                cart_stage.item_id,
                cart_stage.view_time,
                cart_stage.cart_time,
                MIN(behavior.timestamp) AS purchase_time
            FROM cart_stage
            INNER JOIN user_behavior AS behavior
                ON behavior.user_id = cart_stage.user_id
                AND behavior.item_id = cart_stage.item_id
                AND behavior.behavior_type = 4
                AND behavior.timestamp >= cart_stage.cart_time
            GROUP BY
                cart_stage.user_id,
                cart_stage.item_id,
                cart_stage.view_time,
                cart_stage.cart_time
        )

        SELECT
            (
                SELECT COUNT(*)
                FROM view_stage
            ) AS view_pairs,

            (
                SELECT COUNT(*)
                FROM cart_stage
            ) AS view_cart_pairs,

            (
                SELECT COUNT(*)
                FROM purchase_stage
            ) AS full_funnel_pairs,

            (
                SELECT COUNT(DISTINCT user_id)
                FROM view_stage
            ) AS view_users,

            (
                SELECT COUNT(DISTINCT user_id)
                FROM cart_stage
            ) AS view_cart_users,

            (
                SELECT COUNT(DISTINCT user_id)
                FROM purchase_stage
            ) AS full_funnel_users
        """

        print("Running strict funnel query...")

        cursor.execute(strict_funnel_query)
        strict_funnel = cursor.fetchone()

    (
        total_rows,
        unique_users,
        unique_items,
        purchase_rows,
    ) = basic_summary

    (
        view_pairs,
        view_cart_pairs,
        full_funnel_pairs,
        view_users,
        view_cart_users,
        full_funnel_users,
    ) = strict_funnel

    pair_view_to_cart_rate = 100.0 * view_cart_pairs / view_pairs if view_pairs else 0.0

    pair_cart_to_purchase_rate = (
        100.0 * full_funnel_pairs / view_cart_pairs if view_cart_pairs else 0.0
    )

    pair_view_to_purchase_rate = (
        100.0 * full_funnel_pairs / view_pairs if view_pairs else 0.0
    )

    user_view_to_cart_rate = 100.0 * view_cart_users / view_users if view_users else 0.0

    user_cart_to_purchase_rate = (
        100.0 * full_funnel_users / view_cart_users if view_cart_users else 0.0
    )

    user_view_to_purchase_rate = (
        100.0 * full_funnel_users / view_users if view_users else 0.0
    )

    print()
    print("=" * 60)
    print("BASIC SUMMARY")
    print("=" * 60)

    print(f"Total rows: {total_rows:,}")
    print(f"Unique users: {unique_users:,}")
    print(f"Unique items: {unique_items:,}")
    print(f"Purchase rows: {purchase_rows:,}")

    print()
    print("Behavior distribution:")

    for behavior_type, behavior_count in behavior_distribution:
        print(f"Behavior type {behavior_type}: " f"{behavior_count:,}")

    print()
    print("=" * 60)
    print("STRICT USER-ITEM FUNNEL")
    print("=" * 60)

    print(f"View pairs: {view_pairs:,}")
    print("View -> Cart pairs: " f"{view_cart_pairs:,}")
    print("View -> Cart -> Purchase pairs: " f"{full_funnel_pairs:,}")

    print()
    print("User-item pair conversion rates:")

    print("View -> Cart: " f"{pair_view_to_cart_rate:.2f}%")

    print("Cart -> Purchase: " f"{pair_cart_to_purchase_rate:.2f}%")

    print("View -> Purchase: " f"{pair_view_to_purchase_rate:.2f}%")

    print()
    print("=" * 60)
    print("STRICT USER-LEVEL FUNNEL")
    print("=" * 60)

    print(f"View users: {view_users:,}")

    print("View -> Cart users: " f"{view_cart_users:,}")

    print("View -> Cart -> Purchase users: " f"{full_funnel_users:,}")

    print()
    print("Strict user conversion rates:")

    print("View -> Cart: " f"{user_view_to_cart_rate:.2f}%")

    print("Cart -> Purchase: " f"{user_cart_to_purchase_rate:.2f}%")

    print("View -> Purchase: " f"{user_view_to_purchase_rate:.2f}%")


if __name__ == "__main__":
    main()
