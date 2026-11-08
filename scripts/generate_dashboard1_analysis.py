"""Generate extended EDA charts for Dashboard 1."""

import sqlite3
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = PROJECT_ROOT / "data" / "processed" / "user_behavior.db"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "dashboard1"


def read_sql(
    connection: sqlite3.Connection,
    query: str,
) -> pd.DataFrame:
    """Run SQL and return a DataFrame."""
    result = pd.read_sql_query(
        query,
        connection,
    )

    return cast(
        pd.DataFrame,
        result,
    )


def save_csv(
    dataframe: pd.DataFrame,
    filename: str,
) -> None:
    """Save DataFrame to CSV."""
    output_path = OUTPUT_DIR / filename

    dataframe.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved: {output_path}")


def save_current_figure(
    filename: str,
) -> None:
    """Save current matplotlib figure."""
    output_path = OUTPUT_DIR / filename

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


def chart_hourly_activity_heatmap(
    connection: sqlite3.Connection,
) -> None:
    """Create weekday-hour activity heatmap."""
    print("Generating 01 hourly activity heatmap...")

    query = """
    SELECT
        CAST(
            strftime('%w', timestamp)
            AS INTEGER
        ) AS weekday_num,

        CAST(
            strftime('%H', timestamp)
            AS INTEGER
        ) AS hour,

        COUNT(*) AS behavior_count

    FROM user_behavior

    GROUP BY
        weekday_num,
        hour

    ORDER BY
        weekday_num,
        hour
    """

    data = read_sql(
        connection,
        query,
    )

    pivot = data.pivot(
        index="weekday_num",
        columns="hour",
        values="behavior_count",
    ).fillna(0)

    weekday_order = [
        1,
        2,
        3,
        4,
        5,
        6,
        0,
    ]

    pivot = pivot.reindex(weekday_order)

    weekday_labels = [
        "Mon",
        "Tue",
        "Wed",
        "Thu",
        "Fri",
        "Sat",
        "Sun",
    ]

    plt.figure(figsize=(14, 6))

    image = plt.imshow(
        pivot.values,
        aspect="auto",
    )

    plt.colorbar(
        image,
        label="Behavior count",
    )

    plt.title("User Activity Heatmap by Weekday and Hour")

    plt.xlabel("Hour of Day")

    plt.ylabel("Weekday")

    plt.xticks(
        range(24),
        [str(hour) for hour in range(24)],
    )

    plt.yticks(
        range(7),
        weekday_labels,
    )

    save_current_figure("01_hourly_activity_heatmap.png")

    save_csv(
        data,
        "01_hourly_activity_heatmap.csv",
    )


def chart_hourly_conversion_heatmap(
    connection: sqlite3.Connection,
) -> None:
    """Create weekday-hour view-to-purchase heatmap."""
    print("Generating 02 hourly conversion heatmap...")

    query = """
    SELECT
        CAST(
            strftime('%w', timestamp)
            AS INTEGER
        ) AS weekday_num,

        CAST(
            strftime('%H', timestamp)
            AS INTEGER
        ) AS hour,

        SUM(
            CASE
                WHEN behavior_type = 1
                THEN 1
                ELSE 0
            END
        ) AS views,

        SUM(
            CASE
                WHEN behavior_type = 4
                THEN 1
                ELSE 0
            END
        ) AS purchases

    FROM user_behavior

    GROUP BY
        weekday_num,
        hour

    ORDER BY
        weekday_num,
        hour
    """

    data = read_sql(
        connection,
        query,
    )

    data["view_to_purchase_pct"] = (
        data["purchases"]
        / data["views"].replace(
            0,
            np.nan,
        )
        * 100.0
    )

    pivot = data.pivot(
        index="weekday_num",
        columns="hour",
        values="view_to_purchase_pct",
    )

    weekday_order = [
        1,
        2,
        3,
        4,
        5,
        6,
        0,
    ]

    pivot = pivot.reindex(weekday_order)

    weekday_labels = [
        "Mon",
        "Tue",
        "Wed",
        "Thu",
        "Fri",
        "Sat",
        "Sun",
    ]

    plt.figure(figsize=(14, 6))

    image = plt.imshow(
        pivot.values,
        aspect="auto",
    )

    plt.colorbar(
        image,
        label="View-to-Purchase Rate (%)",
    )

    plt.title("Hourly View-to-Purchase Conversion Heatmap")

    plt.xlabel("Hour of Day")

    plt.ylabel("Weekday")

    plt.xticks(
        range(24),
        [str(hour) for hour in range(24)],
    )

    plt.yticks(
        range(7),
        weekday_labels,
    )

    save_current_figure("02_hourly_conversion_heatmap.png")

    save_csv(
        data,
        "02_hourly_conversion_heatmap.csv",
    )


def chart_weekday_weekend_trend(
    connection: sqlite3.Connection,
) -> None:
    """Compare weekday and weekend daily behavior."""
    print("Generating 03 weekday-weekend trend...")

    query = """
    SELECT
        DATE(timestamp) AS event_date,

        CASE
            WHEN strftime(
                '%w',
                timestamp
            ) IN ('0', '6')
            THEN 'Weekend'
            ELSE 'Weekday'
        END AS day_type,

        COUNT(*) AS total_behaviors,

        COUNT(
            DISTINCT user_id
        ) AS dau,

        SUM(
            CASE
                WHEN behavior_type = 4
                THEN 1
                ELSE 0
            END
        ) AS purchases

    FROM user_behavior

    GROUP BY
        DATE(timestamp)

    ORDER BY
        event_date
    """

    data = read_sql(
        connection,
        query,
    )

    data["event_date"] = pd.to_datetime(data["event_date"])

    plt.figure(figsize=(14, 6))

    weekday = data[data["day_type"] == "Weekday"]

    weekend = data[data["day_type"] == "Weekend"]

    plt.plot(
        weekday["event_date"],
        weekday["total_behaviors"],
        marker="o",
        label="Weekday",
    )

    plt.scatter(
        weekend["event_date"],
        weekend["total_behaviors"],
        marker="s",
        label="Weekend",
    )

    plt.title("Daily Behavior Trend: Weekday vs Weekend")

    plt.xlabel("Date")

    plt.ylabel("Total Behaviors")

    plt.legend()

    plt.xticks(rotation=45)

    save_current_figure("03_weekday_weekend_behavior_trend.png")

    save_csv(
        data,
        "03_weekday_weekend_behavior_trend.csv",
    )


def chart_user_activity_distribution(
    connection: sqlite3.Connection,
) -> None:
    """Create user total activity distribution."""
    print("Generating 04 user activity distribution...")

    query = """
    SELECT
        user_id,

        COUNT(*) AS total_behaviors,

        SUM(
            CASE
                WHEN behavior_type = 4
                THEN 1
                ELSE 0
            END
        ) AS purchase_count

    FROM user_behavior

    GROUP BY user_id
    """

    data = read_sql(
        connection,
        query,
    )

    plt.figure(figsize=(10, 6))

    plt.hist(
        np.log1p(data["total_behaviors"]),
        bins=40,
    )

    plt.title("User Activity Distribution")

    plt.xlabel("log(1 + Total Behaviors)")

    plt.ylabel("Number of Users")

    save_current_figure("04_user_activity_distribution.png")

    save_csv(
        data,
        "04_user_activity_distribution.csv",
    )


def chart_purchase_count_distribution(
    connection: sqlite3.Connection,
) -> None:
    """Create user purchase count distribution."""
    print("Generating 05 purchase count distribution...")

    query = """
    SELECT
        user_id,

        SUM(
            CASE
                WHEN behavior_type = 4
                THEN 1
                ELSE 0
            END
        ) AS purchase_count

    FROM user_behavior

    GROUP BY user_id
    """

    data = read_sql(
        connection,
        query,
    )

    plt.figure(figsize=(10, 6))

    plt.hist(
        np.log1p(data["purchase_count"]),
        bins=35,
    )

    plt.title("User Purchase Count Distribution")

    plt.xlabel("log(1 + Purchase Count)")

    plt.ylabel("Number of Users")

    save_current_figure("05_purchase_count_distribution.png")

    save_csv(
        data,
        "05_purchase_count_distribution.csv",
    )


def build_category_summary(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """Build category-level summary."""
    query = """
    SELECT
        category_id,

        COUNT(*) AS total_behaviors,

        SUM(
            CASE
                WHEN behavior_type = 1
                THEN 1
                ELSE 0
            END
        ) AS views,

        SUM(
            CASE
                WHEN behavior_type = 4
                THEN 1
                ELSE 0
            END
        ) AS purchases,

        COUNT(
            DISTINCT user_id
        ) AS users,

        COUNT(
            DISTINCT item_id
        ) AS items

    FROM user_behavior

    GROUP BY category_id
    """

    data = read_sql(
        connection,
        query,
    )

    data["view_to_purchase_pct"] = (
        data["purchases"]
        / data["views"].replace(
            0,
            np.nan,
        )
        * 100.0
    )

    return data


def chart_top_categories(
    connection: sqlite3.Connection,
) -> None:
    """Create category ranking charts."""
    print("Generating 06 top category rankings...")

    data = build_category_summary(connection)

    top_by_volume = data.nlargest(
        20,
        "total_behaviors",
    ).sort_values("total_behaviors")

    plt.figure(figsize=(11, 8))

    plt.barh(
        top_by_volume["category_id"].astype(str),
        top_by_volume["total_behaviors"],
    )

    plt.title("Top 20 Categories by Interaction Volume")

    plt.xlabel("Total Behaviors")

    plt.ylabel("Category ID")

    save_current_figure("06a_top_categories_by_volume.png")

    eligible = data[data["views"] >= 100]

    top_by_conversion = eligible.nlargest(
        20,
        "view_to_purchase_pct",
    ).sort_values("view_to_purchase_pct")

    plt.figure(figsize=(11, 8))

    plt.barh(
        top_by_conversion["category_id"].astype(str),
        top_by_conversion["view_to_purchase_pct"],
    )

    plt.title("Top 20 Categories by View-to-Purchase Rate")

    plt.xlabel("Conversion Rate (%)")

    plt.ylabel("Category ID")

    save_current_figure("06b_top_categories_by_conversion.png")

    save_csv(
        data,
        "06_category_summary.csv",
    )


def build_item_summary(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """Build item-level summary."""
    query = """
    SELECT
        item_id,

        MIN(
            category_id
        ) AS category_id,

        COUNT(*) AS total_behaviors,

        SUM(
            CASE
                WHEN behavior_type = 1
                THEN 1
                ELSE 0
            END
        ) AS views,

        SUM(
            CASE
                WHEN behavior_type = 4
                THEN 1
                ELSE 0
            END
        ) AS purchases,

        COUNT(
            DISTINCT user_id
        ) AS users

    FROM user_behavior

    GROUP BY item_id
    """

    data = read_sql(
        connection,
        query,
    )

    data["view_to_purchase_pct"] = (
        data["purchases"]
        / data["views"].replace(
            0,
            np.nan,
        )
        * 100.0
    )

    return data


def chart_top_items(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """Create item ranking charts."""
    print("Generating 07 top item rankings...")

    data = build_item_summary(connection)

    top_by_volume = data.nlargest(
        20,
        "total_behaviors",
    ).sort_values("total_behaviors")

    plt.figure(figsize=(11, 8))

    plt.barh(
        top_by_volume["item_id"].astype(str),
        top_by_volume["total_behaviors"],
    )

    plt.title("Top 20 Items by Interaction Volume")

    plt.xlabel("Total Behaviors")

    plt.ylabel("Item ID")

    save_current_figure("07a_top_items_by_volume.png")

    eligible = data[data["views"] >= 50]

    top_by_conversion = eligible.nlargest(
        20,
        "view_to_purchase_pct",
    ).sort_values("view_to_purchase_pct")

    plt.figure(figsize=(11, 8))

    plt.barh(
        top_by_conversion["item_id"].astype(str),
        top_by_conversion["view_to_purchase_pct"],
    )

    plt.title("Top 20 Items by View-to-Purchase Rate")

    plt.xlabel("Conversion Rate (%)")

    plt.ylabel("Item ID")

    save_current_figure("07b_top_items_by_conversion.png")

    save_csv(
        data,
        "07_item_summary.csv",
    )

    return data


def chart_category_long_tail(
    connection: sqlite3.Connection,
) -> None:
    """Create category long-tail cumulative chart."""
    print("Generating 08 category long-tail chart...")

    data = build_category_summary(connection)

    data = data.sort_values(
        "total_behaviors",
        ascending=False,
    ).reset_index(drop=True)

    data["cumulative_behavior_share"] = (
        data["total_behaviors"].cumsum() / data["total_behaviors"].sum()
    )

    data["cumulative_purchase_share"] = (
        data["purchases"].cumsum() / data["purchases"].sum()
    )

    plt.figure(figsize=(11, 6))

    plt.plot(
        data.index + 1,
        data["cumulative_behavior_share"],
        label="Cumulative Behavior Share",
    )

    plt.plot(
        data.index + 1,
        data["cumulative_purchase_share"],
        label="Cumulative Purchase Share",
    )

    plt.axhline(
        0.8,
        linestyle="--",
    )

    plt.title("Category Long-Tail Distribution")

    plt.xlabel("Categories Ranked by Interaction Volume")

    plt.ylabel("Cumulative Share")

    plt.legend()

    save_current_figure("08_category_long_tail.png")

    save_csv(
        data,
        "08_category_long_tail.csv",
    )


def chart_item_quadrant(
    item_data: pd.DataFrame,
) -> None:
    """Create exposure-conversion item quadrant."""
    print("Generating 09 item exposure-conversion quadrant...")

    eligible = item_data[item_data["views"] >= 20].copy()

    if eligible.empty:
        print("No eligible items for quadrant chart.")
        return

    x_threshold = float(eligible["views"].median())

    y_threshold = float(eligible["view_to_purchase_pct"].median())

    eligible["quadrant"] = "Long-tail"

    eligible.loc[
        (eligible["views"] >= x_threshold)
        & (eligible["view_to_purchase_pct"] >= y_threshold),
        "quadrant",
    ] = "Star"

    eligible.loc[
        (eligible["views"] >= x_threshold)
        & (eligible["view_to_purchase_pct"] < y_threshold),
        "quadrant",
    ] = "Traffic"

    eligible.loc[
        (eligible["views"] < x_threshold)
        & (eligible["view_to_purchase_pct"] >= y_threshold),
        "quadrant",
    ] = "Potential"

    chart_data = eligible.sample(
        n=min(
            5000,
            len(eligible),
        ),
        random_state=42,
    )

    plt.figure(figsize=(11, 7))

    plt.scatter(
        np.log1p(chart_data["views"]),
        chart_data["view_to_purchase_pct"],
        alpha=0.35,
        s=12,
    )

    plt.axvline(
        np.log1p(x_threshold),
        linestyle="--",
    )

    plt.axhline(
        y_threshold,
        linestyle="--",
    )

    plt.title("Item Exposure-Conversion Quadrant")

    plt.xlabel("log(1 + Views)")

    plt.ylabel("View-to-Purchase Rate (%)")

    save_current_figure("09_item_exposure_conversion_quadrant.png")

    save_csv(
        eligible,
        "09_item_exposure_conversion_quadrant.csv",
    )


def chart_behavior_transition_matrix(
    connection: sqlite3.Connection,
) -> None:
    """Create next-behavior transition probability matrix."""
    print("Generating 10 behavior transition matrix...")

    query = """
    WITH ordered_events AS (
        SELECT
            user_id,
            timestamp,
            item_id,
            behavior_type,

            LEAD(
                behavior_type
            ) OVER (
                PARTITION BY user_id
                ORDER BY
                    timestamp,
                    item_id,
                    behavior_type
            ) AS next_behavior

        FROM user_behavior
    ),

    transition_counts AS (
        SELECT
            behavior_type,
            next_behavior,
            COUNT(*) AS transition_count

        FROM ordered_events

        WHERE next_behavior
            IS NOT NULL

        GROUP BY
            behavior_type,
            next_behavior
    )

    SELECT
        behavior_type,
        next_behavior,
        transition_count

    FROM transition_counts

    ORDER BY
        behavior_type,
        next_behavior
    """

    data = read_sql(
        connection,
        query,
    )

    totals = data.groupby("behavior_type")["transition_count"].transform("sum")

    data["transition_probability"] = data["transition_count"] / totals

    pivot = data.pivot(
        index="behavior_type",
        columns="next_behavior",
        values="transition_probability",
    ).fillna(0)

    pivot = pivot.reindex(
        index=[
            1,
            2,
            3,
            4,
        ],
        columns=[
            1,
            2,
            3,
            4,
        ],
        fill_value=0,
    )

    labels = [
        "View",
        "Favorite",
        "Cart",
        "Purchase",
    ]

    plt.figure(figsize=(8, 7))

    image = plt.imshow(
        pivot.values,
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    plt.colorbar(
        image,
        label="Transition Probability",
    )

    plt.xticks(
        range(4),
        labels,
        rotation=30,
    )

    plt.yticks(
        range(4),
        labels,
    )

    plt.xlabel("Next Behavior")

    plt.ylabel("Current Behavior")

    plt.title("Behavior Transition Probability Matrix")

    for row in range(4):
        for column in range(4):
            plt.text(
                column,
                row,
                f"{pivot.iloc[row, column]:.3f}",
                ha="center",
                va="center",
            )

    save_current_figure("10_behavior_transition_matrix.png")

    save_csv(
        data,
        "10_behavior_transition_matrix.csv",
    )


def chart_behavior_depth_conversion(
    connection: sqlite3.Connection,
) -> None:
    """Create behavior-depth purchase-rate chart."""
    print("Generating 11 behavior depth conversion...")

    query = """
    WITH pair_summary AS (
        SELECT
            user_id,
            item_id,

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

            CASE
                WHEN SUM(
                    CASE
                        WHEN behavior_type = 4
                        THEN 1
                        ELSE 0
                    END
                ) > 0
                THEN 1
                ELSE 0
            END AS purchased

        FROM user_behavior

        GROUP BY
            user_id,
            item_id
    )

    SELECT
        view_count,
        favorite_count,
        cart_count,
        purchased

    FROM pair_summary
    """

    data = read_sql(
        connection,
        query,
    )

    view_bins = [
        -1,
        0,
        1,
        2,
        5,
        10,
        20,
        50,
        np.inf,
    ]

    view_labels = [
        "0",
        "1",
        "2",
        "3-5",
        "6-10",
        "11-20",
        "21-50",
        "51+",
    ]

    data["view_bin"] = pd.cut(
        data["view_count"],
        bins=view_bins,
        labels=view_labels,
    )

    summary = (
        data.groupby(
            "view_bin",
            observed=False,
        )
        .agg(
            pairs=(
                "purchased",
                "count",
            ),
            purchase_rate=(
                "purchased",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["purchase_rate_pct"] = summary["purchase_rate"] * 100.0

    plt.figure(figsize=(10, 6))

    plt.plot(
        summary["view_bin"].astype(str),
        summary["purchase_rate_pct"],
        marker="o",
    )

    plt.title("View Depth vs Purchase Rate")

    plt.xlabel("Historical View Count")

    plt.ylabel("Purchase Rate (%)")

    save_current_figure("11_behavior_depth_conversion.png")

    save_csv(
        summary,
        "11_behavior_depth_conversion.csv",
    )


def chart_activity_category_preference(
    connection: sqlite3.Connection,
) -> None:
    """Create activity-level and category preference cross table."""
    print("Generating 12 activity-category preference...")

    query = """
    WITH user_activity AS (
        SELECT
            user_id,
            COUNT(*) AS total_behaviors

        FROM user_behavior

        GROUP BY user_id
    ),

    activity_level AS (
        SELECT
            user_id,
            total_behaviors,

            NTILE(3) OVER (
                ORDER BY total_behaviors
            ) AS activity_group

        FROM user_activity
    ),

    user_category AS (
        SELECT
            user_id,
            category_id,
            COUNT(*) AS category_behaviors

        FROM user_behavior

        GROUP BY
            user_id,
            category_id
    )

    SELECT
        CASE activity_group
            WHEN 1 THEN 'Low'
            WHEN 2 THEN 'Medium'
            ELSE 'High'
        END AS activity_level,

        user_category.category_id,

        SUM(
            user_category.category_behaviors
        ) AS behavior_count

    FROM activity_level

    INNER JOIN user_category
        ON activity_level.user_id
            = user_category.user_id

    GROUP BY
        activity_group,
        user_category.category_id

    ORDER BY
        activity_group,
        behavior_count DESC
    """

    data = read_sql(
        connection,
        query,
    )

    result_frames = []

    for level in [
        "Low",
        "Medium",
        "High",
    ]:
        subset = (
            data[data["activity_level"] == level]
            .nlargest(
                5,
                "behavior_count",
            )
            .copy()
        )

        result_frames.append(subset)

    top_categories = pd.concat(
        result_frames,
        ignore_index=True,
    )

    top_categories["label"] = (
        top_categories["activity_level"]
        + "-"
        + top_categories["category_id"].astype(str)
    )

    plt.figure(figsize=(12, 7))

    plt.barh(
        top_categories["label"],
        top_categories["behavior_count"],
    )

    plt.title("Top-5 Category Preferences by User Activity Level")

    plt.xlabel("Behavior Count")

    plt.ylabel("Activity Level - Category")

    plt.gca().invert_yaxis()

    save_current_figure("12_activity_category_preference.png")

    save_csv(
        top_categories,
        "12_activity_category_preference.csv",
    )


def main() -> None:
    """Generate all Dashboard 1 analysis outputs."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Opening SQLite database...")

    with sqlite3.connect(DATABASE_PATH) as connection:

        chart_hourly_activity_heatmap(connection)

        chart_hourly_conversion_heatmap(connection)

        chart_weekday_weekend_trend(connection)

        chart_user_activity_distribution(connection)

        chart_purchase_count_distribution(connection)

        chart_top_categories(connection)

        item_data = chart_top_items(connection)

        chart_category_long_tail(connection)

        chart_item_quadrant(item_data)

        chart_behavior_transition_matrix(connection)

        chart_behavior_depth_conversion(connection)

        chart_activity_category_preference(connection)

    print()
    print("=" * 70)
    print("DASHBOARD 1 ANALYSIS COMPLETED")
    print("=" * 70)

    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
