"""Build basic intermediate aggregation tables."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "user_behavior_clean.parquet"

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"


def load_clean_data(file_path: Path) -> pd.DataFrame:
    """Load the cleaned Parquet dataset.

    Args:
        file_path: Path to the cleaned Parquet dataset.

    Returns:
        Cleaned user behavior dataframe.

    Raises:
        FileNotFoundError: If the cleaned dataset does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {file_path}")

    dataframe: pd.DataFrame = pd.read_parquet(file_path)
    return dataframe


def build_user_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Build user-level aggregation features.

    Args:
        dataframe: Cleaned user behavior dataframe.

    Returns:
        User-level summary dataframe.
    """
    behavior_counts = (
        dataframe.pivot_table(
            index="user_id",
            columns="behavior_type",
            values="item_id",
            aggfunc="count",
            fill_value=0,
        )
        .rename(
            columns={
                1: "view_count",
                2: "favorite_count",
                3: "cart_count",
                4: "purchase_count",
            }
        )
        .reset_index()
    )

    user_summary = dataframe.groupby(
        "user_id",
        as_index=False,
    ).agg(
        total_behaviors=("behavior_type", "count"),
        unique_items=("item_id", "nunique"),
        unique_categories=("category_id", "nunique"),
        active_hours=("timestamp", "nunique"),
        first_behavior_time=("timestamp", "min"),
        last_behavior_time=("timestamp", "max"),
    )

    user_summary = user_summary.merge(
        behavior_counts,
        on="user_id",
        how="left",
    )

    count_columns = [
        "view_count",
        "favorite_count",
        "cart_count",
        "purchase_count",
    ]

    for column in count_columns:
        if column not in user_summary.columns:
            user_summary[column] = 0

    user_summary["view_ratio"] = (
        user_summary["view_count"] / user_summary["total_behaviors"]
    )

    user_summary["favorite_ratio"] = (
        user_summary["favorite_count"] / user_summary["total_behaviors"]
    )

    user_summary["cart_ratio"] = (
        user_summary["cart_count"] / user_summary["total_behaviors"]
    )

    user_summary["purchase_ratio"] = (
        user_summary["purchase_count"] / user_summary["total_behaviors"]
    )

    result: pd.DataFrame = user_summary
    return result


def build_item_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Build item-level aggregation features.

    Args:
        dataframe: Cleaned user behavior dataframe.

    Returns:
        Item-level summary dataframe.
    """
    behavior_counts = (
        dataframe.pivot_table(
            index="item_id",
            columns="behavior_type",
            values="user_id",
            aggfunc="count",
            fill_value=0,
        )
        .rename(
            columns={
                1: "view_count",
                2: "favorite_count",
                3: "cart_count",
                4: "purchase_count",
            }
        )
        .reset_index()
    )

    item_summary = dataframe.groupby(
        "item_id",
        as_index=False,
    ).agg(
        category_id=("category_id", "first"),
        total_behaviors=("behavior_type", "count"),
        unique_users=("user_id", "nunique"),
        first_behavior_time=("timestamp", "min"),
        last_behavior_time=("timestamp", "max"),
    )

    item_summary = item_summary.merge(
        behavior_counts,
        on="item_id",
        how="left",
    )

    count_columns = [
        "view_count",
        "favorite_count",
        "cart_count",
        "purchase_count",
    ]

    for column in count_columns:
        if column not in item_summary.columns:
            item_summary[column] = 0

    item_summary["purchase_rate"] = (
        item_summary["purchase_count"] / item_summary["total_behaviors"]
    )

    result: pd.DataFrame = item_summary
    return result


def build_time_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Build hourly time-level aggregation table.

    Args:
        dataframe: Cleaned user behavior dataframe.

    Returns:
        Hour-level summary dataframe.
    """
    time_dataframe = dataframe.copy()

    time_dataframe["date"] = time_dataframe["timestamp"].dt.date

    time_dataframe["hour"] = time_dataframe["timestamp"].dt.hour

    behavior_counts = (
        time_dataframe.pivot_table(
            index=["date", "hour"],
            columns="behavior_type",
            values="user_id",
            aggfunc="count",
            fill_value=0,
        )
        .rename(
            columns={
                1: "view_count",
                2: "favorite_count",
                3: "cart_count",
                4: "purchase_count",
            }
        )
        .reset_index()
    )

    time_summary = time_dataframe.groupby(
        ["date", "hour"],
        as_index=False,
    ).agg(
        total_behaviors=("behavior_type", "count"),
        unique_users=("user_id", "nunique"),
        unique_items=("item_id", "nunique"),
    )

    time_summary = time_summary.merge(
        behavior_counts,
        on=["date", "hour"],
        how="left",
    )

    result: pd.DataFrame = time_summary
    return result


def save_table(
    dataframe: pd.DataFrame,
    file_name: str,
) -> None:
    """Save an intermediate table as Parquet.

    Args:
        dataframe: Dataframe to save.
        file_name: Output file name.
    """
    INTERIM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = INTERIM_DIR / file_name

    dataframe.to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
    )

    print(f"Saved {file_name}: " f"{len(dataframe):,} rows")


def main() -> None:
    """Build and save all basic intermediate tables."""
    print("Loading cleaned dataset...")

    dataframe = load_clean_data(CLEAN_DATA_PATH)

    print(f"Loaded {len(dataframe):,} records.")

    print("\nBuilding user summary...")

    user_summary = build_user_summary(dataframe)

    save_table(
        user_summary,
        "user_summary.parquet",
    )

    print("\nBuilding item summary...")

    item_summary = build_item_summary(dataframe)

    save_table(
        item_summary,
        "item_summary.parquet",
    )

    print("\nBuilding time summary...")

    time_summary = build_time_summary(dataframe)

    save_table(
        time_summary,
        "time_summary.parquet",
    )

    print("\nIntermediate tables completed.")


if __name__ == "__main__":
    main()
