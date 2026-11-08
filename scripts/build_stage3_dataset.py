"""Build Stage 3 train/validation/test datasets."""

import sqlite3
from pathlib import Path
from typing import cast

import pandas as pd
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = PROJECT_ROOT / "data" / "processed" / "user_behavior.db"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "stage3"

SOURCE_TABLE = "modeling_dataset"

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.10


def read_modeling_dataset(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """Load the Stage 2 modeling dataset."""
    print(f"Loading source table: {SOURCE_TABLE}")

    query = f"""
    SELECT *
    FROM {SOURCE_TABLE}
    """

    data = pd.read_sql_query(
        query,
        connection,
    )

    return cast(
        pd.DataFrame,
        data,
    )


def find_label_column(
    dataframe: pd.DataFrame,
) -> str:
    """Find target column."""
    candidates = [
        "label_purchase",
        "label",
        "target",
    ]

    for column in candidates:
        if column in dataframe.columns:
            return column

    raise ValueError(
        "No label column found. " f"Available columns: {list(dataframe.columns)}"
    )


def print_dataset_summary(
    name: str,
    dataframe: pd.DataFrame,
    label_column: str,
) -> None:
    """Print sample distribution summary."""
    total_rows = len(dataframe)

    positives = int(dataframe[label_column].sum())

    negatives = total_rows - positives

    positive_rate = positives / total_rows if total_rows else 0.0

    print()
    print("=" * 70)

    print(name)

    print("=" * 70)

    print(f"Rows: {total_rows:,}")

    print(f"Positive samples: {positives:,}")

    print(f"Negative samples: {negatives:,}")

    print("Positive rate: " f"{positive_rate:.6%}")


def split_dataset(
    dataframe: pd.DataFrame,
    label_column: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create stratified 70/20/10 split."""
    print()
    print("Creating stratified " "70/20/10 split...")

    train_data, temp_data = train_test_split(
        dataframe,
        test_size=(VALIDATION_RATIO + TEST_RATIO),
        stratify=dataframe[label_column],
        random_state=RANDOM_STATE,
    )

    validation_fraction_of_temp = VALIDATION_RATIO / (VALIDATION_RATIO + TEST_RATIO)

    validation_data, test_data = train_test_split(
        temp_data,
        test_size=(1.0 - validation_fraction_of_temp),
        stratify=temp_data[label_column],
        random_state=RANDOM_STATE,
    )

    return (
        cast(
            pd.DataFrame,
            train_data,
        ),
        cast(
            pd.DataFrame,
            validation_data,
        ),
        cast(
            pd.DataFrame,
            test_data,
        ),
    )


def save_dataset(
    dataframe: pd.DataFrame,
    filename: str,
) -> None:
    """Save dataset to parquet."""
    output_path = OUTPUT_DIR / filename

    dataframe.to_parquet(
        output_path,
        index=False,
    )

    print(f"Saved: {output_path}")


def validate_split(
    full_data: pd.DataFrame,
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
    label_column: str,
) -> None:
    """Validate sample counts and class distribution."""
    total_split_rows = len(train_data) + len(validation_data) + len(test_data)

    if total_split_rows != len(full_data):
        raise ValueError("Split row count does not match " "source dataset.")

    full_positive_rate = float(full_data[label_column].mean())

    split_rates = {
        "train": float(train_data[label_column].mean()),
        "validation": float(validation_data[label_column].mean()),
        "test": float(test_data[label_column].mean()),
    }

    print()
    print("=" * 70)

    print("SPLIT VALIDATION")

    print("=" * 70)

    print("Source positive rate: " f"{full_positive_rate:.6%}")

    for name, rate in split_rates.items():
        difference = abs(rate - full_positive_rate)

        print(f"{name} positive rate: " f"{rate:.6%} " f"(difference={difference:.8f})")

    print()
    print("Split validation passed.")


def main() -> None:
    """Build Stage 3 datasets."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Opening SQLite database...")

    with sqlite3.connect(DATABASE_PATH) as connection:
        data = read_modeling_dataset(connection)

    label_column = find_label_column(data)

    print(f"Target column: {label_column}")

    print_dataset_summary(
        "FULL MODELING DATASET",
        data,
        label_column,
    )

    (
        train_data,
        validation_data,
        test_data,
    ) = split_dataset(
        data,
        label_column,
    )

    print_dataset_summary(
        "TRAIN DATASET",
        train_data,
        label_column,
    )

    print_dataset_summary(
        "VALIDATION DATASET",
        validation_data,
        label_column,
    )

    print_dataset_summary(
        "TEST DATASET",
        test_data,
        label_column,
    )

    validate_split(
        data,
        train_data,
        validation_data,
        test_data,
        label_column,
    )

    save_dataset(
        train_data,
        "train.parquet",
    )

    save_dataset(
        validation_data,
        "validation.parquet",
    )

    save_dataset(
        test_data,
        "test.parquet",
    )

    summary = pd.DataFrame(
        [
            {
                "dataset": "full",
                "rows": len(data),
                "positives": int(data[label_column].sum()),
                "positive_rate": float(data[label_column].mean()),
            },
            {
                "dataset": "train",
                "rows": len(train_data),
                "positives": int(train_data[label_column].sum()),
                "positive_rate": float(train_data[label_column].mean()),
            },
            {
                "dataset": "validation",
                "rows": len(validation_data),
                "positives": int(validation_data[label_column].sum()),
                "positive_rate": float(validation_data[label_column].mean()),
            },
            {
                "dataset": "test",
                "rows": len(test_data),
                "positives": int(test_data[label_column].sum()),
                "positive_rate": float(test_data[label_column].mean()),
            },
        ]
    )

    summary_path = OUTPUT_DIR / "split_summary.csv"

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(f"Saved: {summary_path}")

    print()
    print("=" * 70)

    print("STAGE 3 DATASET BUILD COMPLETED")

    print("=" * 70)


if __name__ == "__main__":
    main()
