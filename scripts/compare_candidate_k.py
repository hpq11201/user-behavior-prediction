"""Compare different Top-K category candidate strategies."""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = PROJECT_ROOT / "data" / "processed" / "user_behavior.db"

LABEL_START = "2025-12-11 00:00:00"
LABEL_END = "2025-12-18 23:59:59"

TOP_K_VALUES = [5, 10, 20]


def get_future_purchase_pairs(
    connection: sqlite3.Connection,
) -> int:
    """Return total distinct future purchase user-item pairs."""
    cursor = connection.cursor()

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

    return int(cursor.fetchone()[0])


def get_historical_coverage(
    connection: sqlite3.Connection,
    future_purchase_pairs: int,
) -> tuple[int, float]:
    """Return historical-only covered pairs and coverage."""
    cursor = connection.cursor()

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

    covered_pairs = int(cursor.fetchone()[0])

    coverage = (
        100.0 * covered_pairs / future_purchase_pairs if future_purchase_pairs else 0.0
    )

    return covered_pairs, coverage


def evaluate_top_k(
    connection: sqlite3.Connection,
    top_k: int,
    future_purchase_pairs: int,
    historical_coverage: float,
) -> tuple[int, int, float, int, float, float]:
    """Build and evaluate one Top-K candidate strategy."""
    cursor = connection.cursor()

    table_name = f"candidate_set_k{top_k}"

    print()
    print(f"Building Top-{top_k} candidate set...")

    cursor.execute(f"""
        DROP TABLE IF EXISTS {table_name}
        """)

    cursor.execute(f"""
        CREATE TABLE {table_name} AS

        SELECT
            user_id,
            item_id
        FROM historical_pairs

        UNION

        SELECT
            user_categories.user_id,
            ranked_items.item_id
        FROM user_categories
        INNER JOIN ranked_items
            ON ranked_items.category_id
                = user_categories.category_id
        WHERE ranked_items.item_rank <= {top_k}
        """)

    connection.commit()

    print(f"Creating index for Top-{top_k}...")

    cursor.execute(f"""
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_candidate_set_k{top_k}_user_item
        ON {table_name}(user_id, item_id)
        """)

    connection.commit()

    cursor.execute(f"""
        SELECT COUNT(*)
        FROM {table_name}
        """)
    candidate_pairs = int(cursor.fetchone()[0])

    cursor.execute(f"""
        SELECT COUNT(DISTINCT user_id)
        FROM {table_name}
        """)
    candidate_users = int(cursor.fetchone()[0])

    average_candidates = candidate_pairs / candidate_users if candidate_users else 0.0

    cursor.execute(f"""
        SELECT COUNT(*)
        FROM (
            SELECT DISTINCT
                future.user_id,
                future.item_id
            FROM user_behavior AS future
            INNER JOIN {table_name} AS candidates
                ON candidates.user_id = future.user_id
                AND candidates.item_id = future.item_id
            WHERE future.behavior_type = 4
              AND future.timestamp BETWEEN
                  '{LABEL_START}'
                  AND '{LABEL_END}'
        )
        """)

    covered_pairs = int(cursor.fetchone()[0])

    coverage = (
        100.0 * covered_pairs / future_purchase_pairs if future_purchase_pairs else 0.0
    )

    coverage_gain = coverage - historical_coverage

    return (
        candidate_pairs,
        candidate_users,
        average_candidates,
        covered_pairs,
        coverage,
        coverage_gain,
    )


def main() -> None:
    """Compare Top-K category candidate strategies."""
    print("Opening SQLite database...")

    with sqlite3.connect(DATABASE_PATH) as connection:
        future_purchase_pairs = get_future_purchase_pairs(connection)

        (
            historical_covered_pairs,
            historical_coverage,
        ) = get_historical_coverage(
            connection,
            future_purchase_pairs,
        )

        results = []

        for top_k in TOP_K_VALUES:
            result = evaluate_top_k(
                connection,
                top_k,
                future_purchase_pairs,
                historical_coverage,
            )

            results.append(
                (
                    top_k,
                    *result,
                )
            )

    print()
    print("=" * 90)
    print("CANDIDATE TOP-K COMPARISON")
    print("=" * 90)

    print(f"Future purchase pairs: " f"{future_purchase_pairs:,}")

    print(f"Historical covered pairs: " f"{historical_covered_pairs:,}")

    print(f"Historical-only coverage: " f"{historical_coverage:.2f}%")

    print()

    header = (
        f"{'Top-K':<8}"
        f"{'Candidate pairs':>18}"
        f"{'Avg/user':>14}"
        f"{'Covered':>12}"
        f"{'Coverage':>12}"
        f"{'Gain':>12}"
    )

    print(header)
    print("-" * len(header))

    for (
        top_k,
        candidate_pairs,
        candidate_users,
        average_candidates,
        covered_pairs,
        coverage,
        coverage_gain,
    ) in results:
        del candidate_users

        print(
            f"{top_k:<8}"
            f"{candidate_pairs:>18,}"
            f"{average_candidates:>14.2f}"
            f"{covered_pairs:>12,}"
            f"{coverage:>11.2f}%"
            f"{coverage_gain:>+11.2f}"
        )

    print()
    print("Comparison completed.")


if __name__ == "__main__":
    main()
