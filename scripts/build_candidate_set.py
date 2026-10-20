"""Build and evaluate candidate items for purchase prediction."""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = PROJECT_ROOT / "data" / "processed" / "user_behavior.db"

FEATURE_START = "2025-11-18 00:00:00"
FEATURE_END = "2025-12-10 23:59:59"

LABEL_START = "2025-12-11 00:00:00"
LABEL_END = "2025-12-18 23:59:59"

TOP_ITEMS_PER_CATEGORY = 20


def execute_step(
    connection: sqlite3.Connection,
    message: str,
    sql: str,
) -> None:
    """Execute one SQL step and print progress."""
    if message:
        print(message)

    cursor = connection.cursor()
    cursor.execute(sql)
    connection.commit()


def build_intermediate_tables(
    connection: sqlite3.Connection,
) -> None:
    """Build materialized candidate-generation tables."""

    execute_step(
        connection,
        "Step 1/6: Building historical_pairs...",
        """
        DROP TABLE IF EXISTS historical_pairs
        """,
    )

    execute_step(
        connection,
        "",
        f"""
        CREATE TABLE historical_pairs AS
        SELECT DISTINCT
            user_id,
            item_id
        FROM user_behavior
        WHERE timestamp BETWEEN
            '{FEATURE_START}'
            AND '{FEATURE_END}'
        """,
    )

    execute_step(
        connection,
        "Creating index for historical_pairs...",
        """
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_historical_pairs
        ON historical_pairs(user_id, item_id)
        """,
    )

    execute_step(
        connection,
        "Step 2/6: Building user_categories...",
        """
        DROP TABLE IF EXISTS user_categories
        """,
    )

    execute_step(
        connection,
        "",
        f"""
        CREATE TABLE user_categories AS
        SELECT DISTINCT
            user_id,
            category_id
        FROM user_behavior
        WHERE timestamp BETWEEN
            '{FEATURE_START}'
            AND '{FEATURE_END}'
        """,
    )

    execute_step(
        connection,
        "Creating index for user_categories...",
        """
        CREATE INDEX IF NOT EXISTS
        idx_user_categories_category
        ON user_categories(category_id, user_id)
        """,
    )

    execute_step(
        connection,
        "Step 3/6: Building item_popularity...",
        """
        DROP TABLE IF EXISTS item_popularity
        """,
    )

    execute_step(
        connection,
        "",
        f"""
        CREATE TABLE item_popularity AS
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
        """,
    )

    execute_step(
        connection,
        "Creating index for item_popularity...",
        """
        CREATE INDEX IF NOT EXISTS
        idx_item_popularity_category
        ON item_popularity(
            category_id,
            user_count DESC,
            interaction_count DESC
        )
        """,
    )

    execute_step(
        connection,
        "Step 4/6: Building ranked_items...",
        """
        DROP TABLE IF EXISTS ranked_items
        """,
    )

    execute_step(
        connection,
        "",
        """
        CREATE TABLE ranked_items AS
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
        FROM item_popularity
        """,
    )

    execute_step(
        connection,
        "Creating index for ranked_items...",
        """
        CREATE INDEX IF NOT EXISTS
        idx_ranked_items_category_rank
        ON ranked_items(category_id, item_rank)
        """,
    )

    execute_step(
        connection,
        "Step 5/6: Building category_candidates...",
        """
        DROP TABLE IF EXISTS category_candidates
        """,
    )

    execute_step(
        connection,
        "",
        f"""
        CREATE TABLE category_candidates AS
        SELECT
            user_categories.user_id,
            ranked_items.item_id
        FROM user_categories
        INNER JOIN ranked_items
            ON ranked_items.category_id
                = user_categories.category_id
        WHERE ranked_items.item_rank
            <= {TOP_ITEMS_PER_CATEGORY}
        """,
    )

    execute_step(
        connection,
        "Creating index for category_candidates...",
        """
        CREATE INDEX IF NOT EXISTS
        idx_category_candidates_user_item
        ON category_candidates(user_id, item_id)
        """,
    )

    execute_step(
        connection,
        "Step 6/6: Building candidate_set...",
        """
        DROP TABLE IF EXISTS candidate_set
        """,
    )

    execute_step(
        connection,
        "",
        """
        CREATE TABLE candidate_set AS
        SELECT
            user_id,
            item_id
        FROM historical_pairs

        UNION

        SELECT
            user_id,
            item_id
        FROM category_candidates
        """,
    )

    execute_step(
        connection,
        "Creating unique index for candidate_set...",
        """
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_candidate_set_user_item
        ON candidate_set(user_id, item_id)
        """,
    )


def evaluate_candidates(
    connection: sqlite3.Connection,
) -> None:
    """Evaluate candidate size and future purchase coverage."""

    cursor = connection.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM candidate_set
        """)
    candidate_count = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(DISTINCT user_id)
        FROM candidate_set
        """)
    candidate_users = cursor.fetchone()[0]

    average_candidates_per_user = (
        candidate_count / candidate_users if candidate_users else 0.0
    )

    cursor.execute(f"""
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
        """)
    future_purchase_pairs = cursor.fetchone()[0]

    cursor.execute(f"""
        SELECT COUNT(*)
        FROM (
            SELECT DISTINCT
                future.user_id,
                future.item_id
            FROM user_behavior AS future
            INNER JOIN candidate_set AS candidates
                ON candidates.user_id = future.user_id
                AND candidates.item_id = future.item_id
            WHERE future.behavior_type = 4
              AND future.timestamp BETWEEN
                  '{LABEL_START}'
                  AND '{LABEL_END}'
        )
        """)
    covered_purchase_pairs = cursor.fetchone()[0]

    cursor.execute(f"""
        SELECT COUNT(*)
        FROM (
            SELECT DISTINCT
                future.user_id,
                future.item_id
            FROM user_behavior AS future
            INNER JOIN historical_pairs AS historical
                ON historical.user_id = future.user_id
                AND historical.item_id = future.item_id
            WHERE future.behavior_type = 4
              AND future.timestamp BETWEEN
                  '{LABEL_START}'
                  AND '{LABEL_END}'
        )
        """)
    historical_covered_pairs = cursor.fetchone()[0]

    historical_coverage = (
        100.0 * historical_covered_pairs / future_purchase_pairs
        if future_purchase_pairs
        else 0.0
    )

    candidate_coverage = (
        100.0 * covered_purchase_pairs / future_purchase_pairs
        if future_purchase_pairs
        else 0.0
    )

    coverage_gain = candidate_coverage - historical_coverage

    print()
    print("=" * 60)
    print("CANDIDATE SET VALIDATION")
    print("=" * 60)

    print(f"Top items per category: " f"{TOP_ITEMS_PER_CATEGORY}")

    print(f"Candidate pairs: " f"{candidate_count:,}")

    print(f"Candidate users: " f"{candidate_users:,}")

    print("Average candidates per user: " f"{average_candidates_per_user:,.2f}")

    print()

    print(f"Future purchase pairs: " f"{future_purchase_pairs:,}")

    print("Historical-only covered pairs: " f"{historical_covered_pairs:,}")

    print("Historical-only coverage: " f"{historical_coverage:.2f}%")

    print()

    print("New candidate covered pairs: " f"{covered_purchase_pairs:,}")

    print("New candidate coverage: " f"{candidate_coverage:.2f}%")

    print("Coverage improvement: " f"{coverage_gain:+.2f} percentage points")


def main() -> None:
    """Build and evaluate candidate generation tables."""

    print("Opening SQLite database...")

    with sqlite3.connect(DATABASE_PATH) as connection:
        build_intermediate_tables(connection)

        print()
        print("Evaluating candidate coverage...")

        evaluate_candidates(connection)

    print()
    print("Candidate generation completed.")


if __name__ == "__main__":
    main()
