"""Build interpretable user behavior segments."""

import sqlite3
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = PROJECT_ROOT / "data" / "processed" / "user_behavior.db"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "user_segmentation"


def load_user_behavior_summary(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """Aggregate behavior statistics to user level."""
    query = """
    SELECT
        user_id,

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

        SUM(
            CASE
                WHEN behavior_type = 4
                THEN 1
                ELSE 0
            END
        ) AS purchase_count,

        COUNT(
            DISTINCT DATE(timestamp)
        ) AS active_days

    FROM user_behavior

    GROUP BY user_id

    ORDER BY user_id
    """

    result = pd.read_sql_query(
        query,
        connection,
    )

    return cast(
        pd.DataFrame,
        result,
    )


def calculate_high_value_threshold(
    users: pd.DataFrame,
) -> int:
    """Calculate high-value purchase threshold."""
    purchasers = users.loc[
        users["purchase_count"] > 0,
        "purchase_count",
    ]

    if purchasers.empty:
        return 1

    threshold = int(purchasers.quantile(0.75))

    return max(
        threshold,
        1,
    )


def assign_segments(
    users: pd.DataFrame,
    high_value_threshold: int,
) -> pd.DataFrame:
    """Assign mutually exclusive behavioral segments."""
    result = users.copy()

    result["segment"] = "Browsing-oriented"

    interested_mask = (
        (result["purchase_count"] == 0)
        & (result["cart_count"] == 0)
        & (result["favorite_count"] > 0)
    )

    result.loc[
        interested_mask,
        "segment",
    ] = "Interested"

    high_intent_mask = (result["purchase_count"] == 0) & (result["cart_count"] > 0)

    result.loc[
        high_intent_mask,
        "segment",
    ] = "High-intent"

    buyer_mask = (result["purchase_count"] > 0) & (
        result["purchase_count"] < high_value_threshold
    )

    result.loc[
        buyer_mask,
        "segment",
    ] = "Buyer"

    high_value_mask = result["purchase_count"] >= high_value_threshold

    result.loc[
        high_value_mask,
        "segment",
    ] = "High-value buyer"

    return cast(
        pd.DataFrame,
        result,
    )


def build_segment_summary(
    users: pd.DataFrame,
) -> pd.DataFrame:
    """Build summary statistics for each segment."""
    summary = (
        users.groupby("segment")
        .agg(
            users=(
                "user_id",
                "count",
            ),
            avg_total_behaviors=(
                "total_behaviors",
                "mean",
            ),
            avg_views=(
                "view_count",
                "mean",
            ),
            avg_favorites=(
                "favorite_count",
                "mean",
            ),
            avg_carts=(
                "cart_count",
                "mean",
            ),
            avg_purchases=(
                "purchase_count",
                "mean",
            ),
            avg_active_days=(
                "active_days",
                "mean",
            ),
        )
        .reset_index()
    )

    total_users = int(summary["users"].sum())

    summary["user_share_pct"] = summary["users"] / total_users * 100.0

    summary = summary.sort_values(
        by="users",
        ascending=False,
    ).reset_index(drop=True)

    numeric_columns = [
        "avg_total_behaviors",
        "avg_views",
        "avg_favorites",
        "avg_carts",
        "avg_purchases",
        "avg_active_days",
        "user_share_pct",
    ]

    summary[numeric_columns] = summary[numeric_columns].round(2)

    return cast(
        pd.DataFrame,
        summary,
    )


def save_outputs(
    users: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """Save segmentation outputs."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    user_output_path = OUTPUT_DIR / "user_segments.csv"

    summary_output_path = OUTPUT_DIR / "user_segment_summary.csv"

    users.to_csv(
        user_output_path,
        index=False,
    )

    summary.to_csv(
        summary_output_path,
        index=False,
    )

    print(f"Saved user segments: " f"{user_output_path}")

    print(f"Saved segment summary: " f"{summary_output_path}")


def create_segment_chart(
    summary: pd.DataFrame,
) -> None:
    """Create user segment distribution chart."""
    figure_path = OUTPUT_DIR / "user_segment_distribution.png"

    plt.figure(figsize=(10, 6))

    plt.bar(
        summary["segment"],
        summary["users"],
    )

    plt.title("User Behavior Segment Distribution")

    plt.xlabel("User Segment")

    plt.ylabel("Number of Users")

    plt.xticks(
        rotation=25,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=200,
    )

    plt.close()

    print(f"Saved segment chart: " f"{figure_path}")


def main() -> None:
    """Run user segmentation analysis."""
    print("Opening SQLite database...")

    with sqlite3.connect(DATABASE_PATH) as connection:
        users = load_user_behavior_summary(connection)

    print(f"Loaded users: " f"{len(users):,}")

    high_value_threshold = calculate_high_value_threshold(users)

    print("High-value purchase threshold: " f"{high_value_threshold}")

    segmented_users = assign_segments(
        users,
        high_value_threshold,
    )

    summary = build_segment_summary(segmented_users)

    save_outputs(
        segmented_users,
        summary,
    )

    create_segment_chart(summary)

    print()
    print("=" * 90)
    print("USER SEGMENTATION SUMMARY")
    print("=" * 90)

    print(summary.to_string(index=False))

    print()
    print("Segment definitions:")
    print("Browsing-oriented: " "no purchase, no cart, no favorite")
    print("Interested: " "favorite behavior but no cart or purchase")
    print("High-intent: " "cart behavior but no purchase")
    print("Buyer: " "purchase behavior below the high-value threshold")
    print(
        "High-value buyer: " "purchase count at or above the purchaser 75th percentile"
    )

    print()
    print("User segmentation completed.")


if __name__ == "__main__":
    main()
