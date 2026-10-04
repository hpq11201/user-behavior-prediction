"""Clean and standardize the raw user behavior dataset."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "user_behavior.csv"

PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "user_behavior_clean.parquet"
)

COLUMN_MAPPING = {
    "time": "timestamp",
    "item_category": "category_id",
}

VALID_BEHAVIOR_TYPES = {1, 2, 3, 4}


def load_raw_data(file_path: Path) -> pd.DataFrame:
    """Load the raw user behavior CSV dataset.

    Args:
        file_path: Path to the raw CSV file.

    Returns:
        Raw user behavior dataframe.

    Raises:
        FileNotFoundError: If the raw dataset does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {file_path}")

    dataframe: pd.DataFrame = pd.read_csv(file_path)

    return dataframe


def standardize_schema(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Standardize source column names.

    Args:
        dataframe: Raw user behavior dataframe.

    Returns:
        Dataframe with standardized column names.
    """
    standardized_dataframe: pd.DataFrame = dataframe.rename(columns=COLUMN_MAPPING)

    return standardized_dataframe


def clean_data(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean the standardized user behavior dataset.

    Args:
        dataframe: Standardized dataframe.

    Returns:
        Tuple containing the cleaned dataframe and
        cleaning statistics.
    """
    initial_rows = len(dataframe)

    required_columns = [
        "timestamp",
        "user_id",
        "item_id",
        "category_id",
        "behavior_type",
    ]

    missing_rows_removed = int(dataframe[required_columns].isna().any(axis=1).sum())

    dataframe = dataframe.dropna(subset=required_columns).copy()

    invalid_behavior_mask = ~dataframe["behavior_type"].isin(VALID_BEHAVIOR_TYPES)

    invalid_behavior_rows_removed = int(invalid_behavior_mask.sum())

    dataframe = dataframe.loc[~invalid_behavior_mask].copy()

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        format="%Y-%m-%d %H",
        errors="coerce",
    )

    invalid_timestamp_rows_removed = int(dataframe["timestamp"].isna().sum())

    dataframe = dataframe.dropna(subset=["timestamp"]).copy()

    duplicate_rows_removed = int(
        dataframe.duplicated(
            subset=[
                "user_id",
                "item_id",
                "behavior_type",
                "timestamp",
            ],
            keep="first",
        ).sum()
    )

    dataframe = dataframe.drop_duplicates(
        subset=[
            "user_id",
            "item_id",
            "behavior_type",
            "timestamp",
        ],
        keep="first",
    ).copy()

    dataframe["behavior_type"] = dataframe["behavior_type"].astype("int8")

    dataframe["category_id"] = dataframe["category_id"].astype("int32")

    dataframe["user_id"] = dataframe["user_id"].astype("int32")

    dataframe["item_id"] = dataframe["item_id"].astype("int32")

    cleaned_dataframe: pd.DataFrame = dataframe[
        [
            "timestamp",
            "user_id",
            "item_id",
            "category_id",
            "behavior_type",
        ]
    ].copy()

    final_rows = len(cleaned_dataframe)

    statistics = {
        "initial_rows": initial_rows,
        "missing_rows_removed": missing_rows_removed,
        "invalid_behavior_rows_removed": (invalid_behavior_rows_removed),
        "invalid_timestamp_rows_removed": (invalid_timestamp_rows_removed),
        "duplicate_rows_removed": (duplicate_rows_removed),
        "final_rows": final_rows,
    }

    return cleaned_dataframe, statistics


def save_clean_data(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save cleaned data in Parquet format.

    Args:
        dataframe: Cleaned user behavior dataframe.
        output_path: Output Parquet path.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
    )


def main() -> None:
    """Run the complete data-cleaning workflow."""
    print("Loading raw dataset...")

    dataframe = load_raw_data(RAW_DATA_PATH)

    print(f"Loaded {len(dataframe):,} raw records.")

    dataframe = standardize_schema(dataframe)

    print("Cleaning dataset...")

    cleaned_dataframe, statistics = clean_data(dataframe)

    print("\nCleaning summary:")

    print("Initial rows: " f"{statistics['initial_rows']:,}")

    print("Missing rows removed: " f"{statistics['missing_rows_removed']:,}")

    print(
        "Invalid behavior rows removed: "
        f"{statistics['invalid_behavior_rows_removed']:,}"
    )

    print(
        "Invalid timestamp rows removed: "
        f"{statistics['invalid_timestamp_rows_removed']:,}"
    )

    print("Duplicate rows removed: " f"{statistics['duplicate_rows_removed']:,}")

    print("Final rows: " f"{statistics['final_rows']:,}")

    print("\nSaving cleaned dataset...")

    save_clean_data(
        cleaned_dataframe,
        PROCESSED_DATA_PATH,
    )

    print("Cleaned dataset saved to: " f"{PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
