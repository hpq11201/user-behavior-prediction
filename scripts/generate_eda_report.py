from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "user_behavior.db"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "eda"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def load_daily_behavior(conn: sqlite3.Connection) -> pd.DataFrame:
    query = """
    SELECT
        DATE(timestamp) AS event_date,
        SUM(CASE WHEN behavior_type = 1 THEN 1 ELSE 0 END) AS views,
        SUM(CASE WHEN behavior_type = 2 THEN 1 ELSE 0 END) AS favorites,
        SUM(CASE WHEN behavior_type = 3 THEN 1 ELSE 0 END) AS carts,
        SUM(CASE WHEN behavior_type = 4 THEN 1 ELSE 0 END) AS purchases,
        COUNT(*) AS total_behaviors,
        COUNT(DISTINCT user_id) AS dau
    FROM user_behavior
    GROUP BY DATE(timestamp)
    ORDER BY DATE(timestamp)
    """
    return pd.read_sql_query(query, conn)


def load_behavior_distribution(conn: sqlite3.Connection) -> pd.DataFrame:
    query = """
    SELECT
        behavior_type,
        COUNT(*) AS cnt
    FROM user_behavior
    GROUP BY behavior_type
    ORDER BY behavior_type
    """
    return pd.read_sql_query(query, conn)


def load_strict_funnel(conn: sqlite3.Connection) -> pd.DataFrame:
    query = """
    WITH view_pairs AS (
        SELECT DISTINCT user_id, item_id
        FROM user_behavior
        WHERE behavior_type = 1
    ),
    cart_pairs AS (
        SELECT DISTINCT v.user_id, v.item_id
        FROM view_pairs v
        INNER JOIN user_behavior c
            ON v.user_id = c.user_id
           AND v.item_id = c.item_id
        WHERE c.behavior_type = 3
    ),
    purchase_pairs AS (
        SELECT DISTINCT c.user_id, c.item_id
        FROM cart_pairs c
        INNER JOIN user_behavior p
            ON c.user_id = p.user_id
           AND c.item_id = p.item_id
        WHERE p.behavior_type = 4
    )
    SELECT 'View' AS stage, COUNT(*) AS cnt FROM view_pairs
    UNION ALL
    SELECT 'View->Cart' AS stage, COUNT(*) AS cnt FROM cart_pairs
    UNION ALL
    SELECT 'View->Cart->Purchase' AS stage, COUNT(*) AS cnt FROM purchase_pairs
    """
    return pd.read_sql_query(query, conn)


def save_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(df[x_col]), df[y_col], marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()


def save_multi_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
) -> None:
    plt.figure(figsize=(12, 6))
    x = pd.to_datetime(df[x_col])
    for col in y_cols:
        plt.plot(x, df[col], marker="o", label=col)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()


def save_bar_chart(
    categories: list[str],
    values: list[float],
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.bar(categories, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()


def main() -> None:
    print("Opening SQLite database...")
    ensure_output_dir()

    conn = get_connection()

    print("Loading daily behavior data...")
    daily = load_daily_behavior(conn)
    daily["view_to_purchase_rate"] = (daily["purchases"] / daily["views"] * 100).fillna(
        0
    )
    daily["cart_to_purchase_rate"] = (daily["purchases"] / daily["carts"] * 100).fillna(
        0
    )

    print("Loading behavior distribution...")
    behavior_dist = load_behavior_distribution(conn)

    print("Loading strict funnel...")
    funnel = load_strict_funnel(conn)

    conn.close()

    print("Generating chart 1/8...")
    save_line_chart(
        daily,
        "event_date",
        "total_behaviors",
        "Daily Total Behaviors",
        "Date",
        "Count",
        "01_daily_behavior_trend.png",
    )

    print("Generating chart 2/8...")
    save_line_chart(
        daily,
        "event_date",
        "dau",
        "Daily Active Users (DAU)",
        "Date",
        "Users",
        "02_daily_active_users.png",
    )

    print("Generating chart 3/8...")
    behavior_name_map = {
        1: "View",
        2: "Favorite",
        3: "Cart",
        4: "Purchase",
    }
    categories = [
        behavior_name_map.get(x, str(x)) for x in behavior_dist["behavior_type"]
    ]
    values = behavior_dist["cnt"].tolist()
    save_bar_chart(
        categories,
        values,
        "Behavior Distribution",
        "Behavior Type",
        "Count",
        "03_behavior_distribution.png",
    )

    print("Generating chart 4/8...")
    save_line_chart(
        daily,
        "event_date",
        "purchases",
        "Daily Purchase Trend",
        "Date",
        "Purchases",
        "04_daily_purchase_trend.png",
    )

    print("Generating chart 5/8...")
    save_multi_line_chart(
        daily,
        "event_date",
        ["views", "favorites", "carts", "purchases"],
        "Daily Behavior Breakdown",
        "Date",
        "Count",
        "05_daily_behavior_breakdown.png",
    )

    print("Generating chart 6/8...")
    save_line_chart(
        daily,
        "event_date",
        "view_to_purchase_rate",
        "Daily View to Purchase Conversion Rate",
        "Date",
        "Conversion Rate (%)",
        "06_view_purchase_conversion.png",
    )

    print("Generating chart 7/8...")
    save_line_chart(
        daily,
        "event_date",
        "cart_to_purchase_rate",
        "Daily Cart to Purchase Conversion Rate",
        "Date",
        "Conversion Rate (%)",
        "07_cart_purchase_conversion.png",
    )

    print("Generating chart 8/8...")
    save_bar_chart(
        funnel["stage"].tolist(),
        funnel["cnt"].tolist(),
        "Strict User-Item Funnel",
        "Stage",
        "Count",
        "08_strict_funnel.png",
    )

    print("EDA report charts generated successfully.")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
