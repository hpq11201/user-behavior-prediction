"""Clean the raw user behavior dataset and save it as Parquet."""

from pathlib import Path
from typing import Any, cast

import pandas as pd

RAW_DATA_PATH = Path("data/raw/user_behavior.csv")
OUTPUT_PATH = Path("data/processed/user_behavior_clean.parquet")

COLUMN_MAPPING = {
    "time": "timestamp",
    "item_category": "category_id",
}

REQUIRED_COLUMNS = [
    "timestamp",
    "user_id",
    "item_id",
    "category_id",
    "behavior_type",
]

BUSINESS_KEY = [
    "user_id",
    "item_id",
    "behavior_type",
    "timestamp",
]

VALID_BEHAVIOR_TYPES = {1, 2, 3, 4}

MAX_INTERACTIONS_PER_HOUR = 100


def load_raw_data(file_path: Path) -> pd.DataFrame:
    """Load the raw user behavior dataset.

    Args:
        file_path: Path to the raw CSV dataset.

    Returns:
        Raw user behavior DataFrame.

    Raises:
        FileNotFoundError: If the raw dataset does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {file_path}")

    dataframe = cast(
        pd.DataFrame,
        pd.read_csv(file_path),
    )

    return dataframe


def standardize_schema(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Standardize source column names.

    Args:
        dataframe: Raw user behavior DataFrame.

    Returns:
        DataFrame with standardized column names.
    """
    standardized = cast(
        pd.DataFrame,
        dataframe.rename(columns=COLUMN_MAPPING).copy(),
    )

    return standardized


def validate_required_columns(dataframe: pd.DataFrame) -> None:
    """Validate that all required columns exist.

    Args:
        dataframe: User behavior DataFrame.

    Raises:
        ValueError: If required columns are missing.
    """
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def convert_timestamp(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert timestamp column to datetime.

    Args:
        dataframe: User behavior DataFrame.

    Returns:
        DataFrame with datetime timestamp.
    """
    converted = dataframe.copy()

    converted["timestamp"] = pd.to_datetime(
        converted["timestamp"],
        errors="coerce",
    )

    converted = cast(
        pd.DataFrame,
        converted,
    )

    return converted


def remove_missing_rows(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Remove rows with missing required values.

    Args:
        dataframe: User behavior DataFrame.

    Returns:
        Cleaned DataFrame and removed row count.
    """
    initial_rows = len(dataframe)

    cleaned = cast(
        pd.DataFrame,
        dataframe.dropna(
            subset=REQUIRED_COLUMNS,
        ).copy(),
    )

    removed_rows = initial_rows - len(cleaned)

    return cleaned, removed_rows


def remove_invalid_behavior_rows(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Remove rows with invalid behavior types.

    Args:
        dataframe: User behavior DataFrame.

    Returns:
        Cleaned DataFrame and removed row count.
    """
    valid_mask = dataframe["behavior_type"].isin(VALID_BEHAVIOR_TYPES)

    cleaned = cast(
        pd.DataFrame,
        dataframe.loc[valid_mask].copy(),
    )

    removed_rows = len(dataframe) - len(cleaned)

    return cleaned, removed_rows


def remove_invalid_timestamp_rows(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Remove rows with invalid timestamps.

    Args:
        dataframe: User behavior DataFrame.

    Returns:
        Cleaned DataFrame and removed row count.
    """
    valid_mask = dataframe["timestamp"].notna()

    cleaned = cast(
        pd.DataFrame,
        dataframe.loc[valid_mask].copy(),
    )

    removed_rows = len(dataframe) - len(cleaned)

    return cleaned, removed_rows


def remove_high_frequency_interactions(
    dataframe: pd.DataFrame,
    max_interactions_per_hour: int,
) -> tuple[pd.DataFrame, int, int]:
    """Remove abnormally frequent interaction groups.

    Groups are defined by user_id, item_id, behavior_type,
    and hourly timestamp. If a group frequency is above the
    configured threshold, the entire group is removed.

    Args:
        dataframe: User behavior DataFrame.
        max_interactions_per_hour: Maximum allowed frequency
            for one user-item-behavior-hour group.

    Returns:
        Cleaned DataFrame, removed row count,
        and removed group count.
    """
    interaction_frequency = dataframe.groupby(
        BUSINESS_KEY,
        dropna=False,
    )[
        BUSINESS_KEY[0]
    ].transform("size")

    abnormal_mask = interaction_frequency > max_interactions_per_hour

    removed_rows = int(abnormal_mask.sum())

    abnormal_dataframe = cast(
        pd.DataFrame,
        dataframe.loc[abnormal_mask].copy(),
    )

    if abnormal_dataframe.empty:
        removed_group_count = 0
    else:
        removed_group_count = int(
            abnormal_dataframe.groupby(
                BUSINESS_KEY,
                dropna=False,
            ).ngroups
        )

    cleaned = cast(
        pd.DataFrame,
        dataframe.loc[~abnormal_mask].copy(),
    )

    return (
        cleaned,
        removed_rows,
        removed_group_count,
    )


def optimize_data_types(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Optimize DataFrame data types.

    Args:
        dataframe: Cleaned user behavior DataFrame.

    Returns:
        DataFrame with optimized dtypes.
    """
    optimized = dataframe.copy()

    optimized["user_id"] = optimized["user_id"].astype("int32")
    optimized["item_id"] = optimized["item_id"].astype("int32")
    optimized["category_id"] = optimized["category_id"].astype("int32")
    optimized["behavior_type"] = optimized["behavior_type"].astype("int8")

    optimized = cast(
        pd.DataFrame,
        optimized,
    )

    return optimized


def clean_data(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the standardized user behavior dataset.

    Args:
        dataframe: Standardized user behavior DataFrame.

    Returns:
        Cleaned DataFrame and cleaning summary.
    """
    summary: dict[str, Any] = {}

    summary["initial_rows"] = len(dataframe)

    dataframe = convert_timestamp(dataframe)

    (
        dataframe,
        missing_rows_removed,
    ) = remove_missing_rows(dataframe)

    summary["missing_rows_removed"] = missing_rows_removed

    (
        dataframe,
        invalid_behavior_rows_removed,
    ) = remove_invalid_behavior_rows(dataframe)

    summary["invalid_behavior_rows_removed"] = invalid_behavior_rows_removed

    (
        dataframe,
        invalid_timestamp_rows_removed,
    ) = remove_invalid_timestamp_rows(dataframe)

    summary["invalid_timestamp_rows_removed"] = invalid_timestamp_rows_removed

    (
        dataframe,
        high_frequency_rows_removed,
        high_frequency_groups_removed,
    ) = remove_high_frequency_interactions(
        dataframe=dataframe,
        max_interactions_per_hour=(MAX_INTERACTIONS_PER_HOUR),
    )

    summary["high_frequency_rows_removed"] = high_frequency_rows_removed

    summary["high_frequency_groups_removed"] = high_frequency_groups_removed

    dataframe = optimize_data_types(dataframe)

    summary["final_rows"] = len(dataframe)

    return dataframe, summary


def save_clean_data(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save the cleaned dataset as Parquet.

    Args:
        dataframe: Cleaned user behavior DataFrame.
        output_path: Destination Parquet path.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_parquet(
        output_path,
        index=False,
    )


def print_summary(
    summary: dict[str, Any],
) -> None:
    """Print the cleaning summary.

    Args:
        summary: Cleaning result summary.
    """
    print()
    print("Cleaning summary:")
    print(f"Initial rows: " f"{summary['initial_rows']:,}")
    print("Missing rows removed: " f"{summary['missing_rows_removed']:,}")
    print(
        "Invalid behavior rows removed: "
        f"{summary['invalid_behavior_rows_removed']:,}"
    )
    print(
        "Invalid timestamp rows removed: "
        f"{summary['invalid_timestamp_rows_removed']:,}"
    )
    print(
        "High-frequency groups removed: "
        f"{summary['high_frequency_groups_removed']:,}"
    )
    print("High-frequency rows removed: " f"{summary['high_frequency_rows_removed']:,}")
    print(f"Final rows: " f"{summary['final_rows']:,}")


def main() -> None:
    """Run the complete data cleaning workflow."""
    print("Loading raw dataset...")

    raw_dataframe = load_raw_data(RAW_DATA_PATH)

    print(f"Loaded {len(raw_dataframe):,} " "raw records.")

    print("Standardizing schema...")

    standardized_dataframe = standardize_schema(raw_dataframe)

    validate_required_columns(standardized_dataframe)

    print("Cleaning dataset...")

    (
        cleaned_dataframe,
        summary,
    ) = clean_data(standardized_dataframe)

    print_summary(summary)

    print()
    print("Saving cleaned dataset...")

    save_clean_data(
        dataframe=cleaned_dataframe,
        output_path=OUTPUT_PATH,
    )

    print("Cleaned dataset saved to: " f"{OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
