"""Reusable SQL analysis queries for the user behavior project."""

import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = PROJECT_ROOT / "data" / "processed" / "user_behavior.db"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "sql_analysis"


def run_query(
    connection: sqlite3.Connection,
    query: str,
) -> pd.DataFrame:
    """Run SQL query and return result as DataFrame."""
    return pd.read_sql_query(
        query,
        connection,
    )


def save_result(
    dataframe: pd.DataFrame,
    filename: str,
) -> None:
    """Save query result to CSV."""
    output_path = OUTPUT_DIR / filename

    dataframe.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved: " f"{output_path}")


def basic_summary(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """Return basic dataset statistics."""
    query = """
    SELECT
        COUNT(*) AS total_behaviors,
        COUNT(DISTINCT user_id) AS unique_users,
        COUNT(DISTINCT item_id) AS unique_items,
        COUNT(DISTINCT category_id) AS unique_categories,
        MIN(timestamp) AS first_timestamp,
        MAX(timestamp) AS last_timestamp
    FROM user_behavior
    """

    return run_query(
        connection,
        query,
    )


def behavior_distribution(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """Return behavior counts by behavior type."""
    query = """
    SELECT
        behavior_type,
        COUNT(*) AS behavior_count
    FROM user_behavior
    GROUP BY behavior_type
    ORDER BY behavior_type
    """

    return run_query(
        connection,
        query,
    )


def daily_active_users(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """Return daily active users and total behavior volume."""
    query = """
    SELECT
        DATE(timestamp) AS event_date,
        COUNT(DISTINCT user_id) AS dau,
        COUNT(*) AS total_behaviors,
        ROUND(
            COUNT(*) * 1.0
            / COUNT(DISTINCT user_id),
            2
        ) AS behaviors_per_user
    FROM user_behavior
    GROUP BY DATE(timestamp)
    ORDER BY event_date
    """

    return run_query(
        connection,
        query,
    )


def daily_behavior_breakdown(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """Return daily counts for all four behavior types."""
    query = """
    SELECT
        DATE(timestamp) AS event_date,

        SUM(
            CASE
                WHEN behavior_type = 1
                THEN 1
                ELSE 0
            END
        ) AS views,

        SUM(
            CASE
                WHEN behavior_type = 2
                THEN 1
                ELSE 0
            END
        ) AS favorites,

        SUM(
            CASE
                WHEN behavior_type = 3
                THEN 1
                ELSE 0
            END
        ) AS carts,

        SUM(
            CASE
                WHEN behavior_type = 4
                THEN 1
                ELSE 0
            END
        ) AS purchases

    FROM user_behavior

    GROUP BY DATE(timestamp)

    ORDER BY event_date
    """

    return run_query(
        connection,
        query,
    )


def daily_conversion_rates(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """Return daily behavior-count conversion rates."""
    query = """
    SELECT
        event_date,
        views,
        carts,
        purchases,

        ROUND(
            purchases * 100.0
            / NULLIF(views, 0),
            4
        ) AS view_to_purchase_pct,

        ROUND(
            purchases * 100.0
            / NULLIF(carts, 0),
            4
        ) AS cart_to_purchase_pct

    FROM (
        SELECT
            DATE(timestamp) AS event_date,

            SUM(
                CASE
                    WHEN behavior_type = 1
                    THEN 1
                    ELSE 0
                END
            ) AS views,

            SUM(
                CASE
                    WHEN behavior_type = 3
                    THEN 1
                    ELSE 0
                END
            ) AS carts,

            SUM(
                CASE
                    WHEN behavior_type = 4
                    THEN 1
                    ELSE 0
                END
            ) AS purchases

        FROM user_behavior

        GROUP BY DATE(timestamp)
    )

    ORDER BY event_date
    """

    return run_query(
        connection,
        query,
    )


def strict_user_item_funnel(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """Return strict user-item chronological funnel counts."""
    query = """
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
            MIN(cart.timestamp) AS cart_time
        FROM view_stage

        INNER JOIN user_behavior AS cart
            ON cart.user_id = view_stage.user_id
            AND cart.item_id = view_stage.item_id

        WHERE cart.behavior_type = 3
          AND cart.timestamp >= view_stage.view_time

        GROUP BY
            view_stage.user_id,
            view_stage.item_id
    ),

    purchase_stage AS (
        SELECT
            cart_stage.user_id,
            cart_stage.item_id,
            MIN(purchase.timestamp) AS purchase_time
        FROM cart_stage

        INNER JOIN user_behavior AS purchase
            ON purchase.user_id = cart_stage.user_id
            AND purchase.item_id = cart_stage.item_id

        WHERE purchase.behavior_type = 4
          AND purchase.timestamp >= cart_stage.cart_time

        GROUP BY
            cart_stage.user_id,
            cart_stage.item_id
    )

    SELECT
        'View' AS funnel_stage,
        COUNT(*) AS pair_count
    FROM view_stage

    UNION ALL

    SELECT
        'View -> Cart' AS funnel_stage,
        COUNT(*) AS pair_count
    FROM cart_stage

    UNION ALL

    SELECT
        'View -> Cart -> Purchase' AS funnel_stage,
        COUNT(*) AS pair_count
    FROM purchase_stage
    """

    return run_query(
        connection,
        query,
    )


def candidate_coverage_summary(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """Return candidate coverage for the selected Top-10 strategy."""
    query = """
    WITH future_pairs AS (
        SELECT DISTINCT
            user_id,
            item_id
        FROM user_behavior
        WHERE behavior_type = 4
          AND timestamp BETWEEN
              '2025-12-11 00:00:00'
              AND '2025-12-18 23:59:59'
    ),

    historical_covered AS (
        SELECT
            COUNT(*) AS covered_pairs
        FROM future_pairs

        INNER JOIN historical_pairs
            ON historical_pairs.user_id
                = future_pairs.user_id
            AND historical_pairs.item_id
                = future_pairs.item_id
    ),

    candidate_covered AS (
        SELECT
            COUNT(*) AS covered_pairs
        FROM future_pairs

        INNER JOIN candidate_set_k10
            ON candidate_set_k10.user_id
                = future_pairs.user_id
            AND candidate_set_k10.item_id
                = future_pairs.item_id
    )

    SELECT
        (
            SELECT COUNT(*)
            FROM future_pairs
        ) AS future_purchase_pairs,

        (
            SELECT covered_pairs
            FROM historical_covered
        ) AS historical_covered_pairs,

        ROUND(
            (
                SELECT covered_pairs
                FROM historical_covered
            ) * 100.0
            /
            (
                SELECT COUNT(*)
                FROM future_pairs
            ),
            2
        ) AS historical_coverage_pct,

        (
            SELECT covered_pairs
            FROM candidate_covered
        ) AS top10_covered_pairs,

        ROUND(
            (
                SELECT covered_pairs
                FROM candidate_covered
            ) * 100.0
            /
            (
                SELECT COUNT(*)
                FROM future_pairs
            ),
            2
        ) AS top10_coverage_pct
    """

    return run_query(
        connection,
        query,
    )


def main() -> None:
    """Run and save all formal SQL analysis outputs."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Opening SQLite database...")

    with sqlite3.connect(DATABASE_PATH) as connection:
        analyses = [
            (
                "01_basic_summary.csv",
                basic_summary(connection),
            ),
            (
                "02_behavior_distribution.csv",
                behavior_distribution(connection),
            ),
            (
                "03_daily_active_users.csv",
                daily_active_users(connection),
            ),
            (
                "04_daily_behavior_breakdown.csv",
                daily_behavior_breakdown(connection),
            ),
            (
                "05_daily_conversion_rates.csv",
                daily_conversion_rates(connection),
            ),
            (
                "06_strict_user_item_funnel.csv",
                strict_user_item_funnel(connection),
            ),
            (
                "07_candidate_coverage_summary.csv",
                candidate_coverage_summary(connection),
            ),
        ]

    print()
    print("=" * 60)
    print("SQL ANALYSIS OUTPUT")
    print("=" * 60)

    for filename, dataframe in analyses:
        save_result(
            dataframe,
            filename,
        )

        print()
        print(filename)
        print(dataframe.head(10).to_string(index=False))

    print()
    print("SQL analysis query set completed.")


if __name__ == "__main__":
    main()
