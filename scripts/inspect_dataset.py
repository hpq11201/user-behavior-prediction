"""Inspect the raw user behavior dataset before data cleaning."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "user_behavior.csv"

RAW_COLUMNS = [
    "time",
    "user_id",
    "item_id",
    "item_category",
    "behavior_type",
]

COLUMN_MAPPING = {
    "time": "timestamp",
    "item_category": "category_id",
}

EXPECTED_COLUMNS = [
    "timestamp",
    "user_id",
    "item_id",
    "category_id",
    "behavior_type",
]

VALID_BEHAVIOR_TYPES = {1, 2, 3, 4}
CHUNK_SIZE = 1_000_000


def standardize_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Rename raw dataset columns to the project standard schema.

    Args:
        dataframe: Raw dataframe using source column names.

    Returns:
        Dataframe with standardized project column names.
    """
    renamed_dataframe: pd.DataFrame = dataframe.rename(columns=COLUMN_MAPPING)
    return renamed_dataframe


def inspect_dataset(file_path: Path) -> None:
    """Inspect the structure and quality of the raw dataset.

    Args:
        file_path: Path to the raw CSV dataset.

    Raises:
        FileNotFoundError: If the dataset does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    print("=" * 60)
    print("RAW DATASET INSPECTION")
    print("=" * 60)
    print(f"Dataset path: {file_path}")
    print(f"File size: " f"{file_path.stat().st_size / (1024 ** 2):.2f} MB")

    total_rows = 0
    duplicate_rows = 0
    invalid_timestamp_count = 0

    null_counts = {column: 0 for column in EXPECTED_COLUMNS}

    behavior_counts: dict[object, int] = {}

    user_ids: set[object] = set()
    item_ids: set[object] = set()
    category_ids: set[object] = set()

    min_timestamp = None
    max_timestamp = None
    columns_checked = False

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        if not columns_checked:
            print("\nRaw columns:")
            print(list(chunk.columns))

            missing_raw_columns = set(RAW_COLUMNS) - set(chunk.columns)

            extra_raw_columns = set(chunk.columns) - set(RAW_COLUMNS)

            print("Missing raw columns: " f"{sorted(missing_raw_columns)}")

            print("Unexpected raw columns: " f"{sorted(extra_raw_columns)}")

            columns_checked = True

        chunk = standardize_columns(chunk)

        total_rows += len(chunk)

        for column in EXPECTED_COLUMNS:
            null_counts[column] += int(chunk[column].isna().sum())

        duplicate_rows += int(chunk.duplicated().sum())

        user_ids.update(chunk["user_id"].dropna().unique())

        item_ids.update(chunk["item_id"].dropna().unique())

        category_ids.update(chunk["category_id"].dropna().unique())

        counts = chunk["behavior_type"].value_counts(dropna=False)

        for behavior_type, count in counts.items():
            behavior_counts[behavior_type] = behavior_counts.get(
                behavior_type,
                0,
            ) + int(count)

        timestamps = pd.to_datetime(
            chunk["timestamp"],
            format="%Y-%m-%d %H",
            errors="coerce",
        )

        invalid_timestamp_count += int(timestamps.isna().sum())

        chunk_min = timestamps.min()
        chunk_max = timestamps.max()

        if pd.notna(chunk_min):
            if min_timestamp is None or chunk_min < min_timestamp:
                min_timestamp = chunk_min

        if pd.notna(chunk_max):
            if max_timestamp is None or chunk_max > max_timestamp:
                max_timestamp = chunk_max

        print(f"Processed chunk {chunk_number}: " f"{total_rows:,} rows processed")

    invalid_behavior_count = sum(
        count
        for behavior_type, count in behavior_counts.items()
        if behavior_type not in VALID_BEHAVIOR_TYPES
    )

    print("\n" + "=" * 60)
    print("INSPECTION RESULTS")
    print("=" * 60)

    print(f"Total rows: {total_rows:,}")

    print("Duplicate rows within chunks: " f"{duplicate_rows:,}")

    print("\nMissing values:")

    for column, count in null_counts.items():
        print(f"  {column}: {count:,}")

    print("\nUnique values:")

    print(f"  Users: {len(user_ids):,}")

    print(f"  Items: {len(item_ids):,}")

    print(f"  Categories: {len(category_ids):,}")

    print("\nBehavior distribution:")

    for behavior_type, count in sorted(
        behavior_counts.items(),
        key=lambda item: str(item[0]),
    ):
        print(f"  {behavior_type}: {count:,}")

    print("\nInvalid behavior records: " f"{invalid_behavior_count:,}")

    print("Invalid timestamp records: " f"{invalid_timestamp_count:,}")

    print("\nTimestamp range:")

    print(f"  Earliest: {min_timestamp}")

    print(f"  Latest: {max_timestamp}")

    print("\nInspection completed.")


if __name__ == "__main__":
    inspect_dataset(DATA_PATH)
