"""Validate cleaned data and intermediate aggregation tables."""

from pathlib import Path
from typing import cast

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "user_behavior_clean.parquet"

USER_SUMMARY_PATH = PROJECT_ROOT / "data" / "interim" / "user_summary.parquet"

ITEM_SUMMARY_PATH = PROJECT_ROOT / "data" / "interim" / "item_summary.parquet"

TIME_SUMMARY_PATH = PROJECT_ROOT / "data" / "interim" / "time_summary.parquet"

EXPECTED_COLUMNS = [
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


def print_result(
    check_name: str,
    passed: bool,
    detail: str,
) -> bool:
    """Print a validation result.

    Args:
        check_name: Name of the validation check.
        passed: Whether the check passed.
        detail: Additional validation information.

    Returns:
        The original passed status.
    """
    status = "PASS" if passed else "FAIL"

    print(f"[{status}] {check_name}: {detail}")

    return passed


def validate_interaction_frequency(
    dataframe: pd.DataFrame,
) -> bool:
    """Validate hourly interaction frequency.

    A user-item-behavior-hour group is considered invalid when
    its interaction count exceeds MAX_INTERACTIONS_PER_HOUR.

    Args:
        dataframe: Cleaned user behavior dataframe.

    Returns:
        Whether no abnormal high-frequency groups remain.
    """
    interaction_frequency = cast(
        pd.Series,
        dataframe.groupby(
            BUSINESS_KEY,
            dropna=False,
        ).size(),
    )

    abnormal_frequency = interaction_frequency[
        interaction_frequency > MAX_INTERACTIONS_PER_HOUR
    ]

    abnormal_groups = len(abnormal_frequency)

    abnormal_rows = int(abnormal_frequency.sum())

    maximum_frequency = int(interaction_frequency.max())

    return print_result(
        "High-frequency interactions",
        abnormal_groups == 0,
        (
            f"{abnormal_groups:,} groups above "
            f"{MAX_INTERACTIONS_PER_HOUR} interactions/hour, "
            f"{abnormal_rows:,} rows affected, "
            f"maximum frequency={maximum_frequency:,}"
        ),
    )


def validate_clean_data(
    dataframe: pd.DataFrame,
) -> list[bool]:
    """Validate the cleaned interaction dataset.

    Args:
        dataframe: Cleaned user behavior dataframe.

    Returns:
        Validation results for each check.
    """
    results: list[bool] = []

    results.append(
        print_result(
            "Required columns",
            set(EXPECTED_COLUMNS) == set(dataframe.columns),
            str(list(dataframe.columns)),
        )
    )

    missing_values = int(dataframe.isna().sum().sum())

    results.append(
        print_result(
            "Missing values",
            missing_values == 0,
            f"{missing_values:,} missing values",
        )
    )

    invalid_behaviors = int(
        (~dataframe["behavior_type"].isin(VALID_BEHAVIOR_TYPES)).sum()
    )

    results.append(
        print_result(
            "Behavior values",
            invalid_behaviors == 0,
            (f"{invalid_behaviors:,} " "invalid behavior records"),
        )
    )

    invalid_timestamps = int(dataframe["timestamp"].isna().sum())

    results.append(
        print_result(
            "Timestamp validity",
            invalid_timestamps == 0,
            (f"{invalid_timestamps:,} " "invalid timestamp records"),
        )
    )

    results.append(validate_interaction_frequency(dataframe))

    for column in [
        "user_id",
        "item_id",
        "category_id",
    ]:
        invalid_ids = int((dataframe[column] <= 0).sum())

        results.append(
            print_result(
                f"{column} validity",
                invalid_ids == 0,
                (f"{invalid_ids:,} " "non-positive values"),
            )
        )

    return results


def validate_intermediate_tables(
    clean_dataframe: pd.DataFrame,
    user_summary: pd.DataFrame,
    item_summary: pd.DataFrame,
    time_summary: pd.DataFrame,
) -> list[bool]:
    """Validate intermediate aggregation tables.

    Args:
        clean_dataframe: Cleaned interaction dataset.
        user_summary: User-level aggregation table.
        item_summary: Item-level aggregation table.
        time_summary: Time-level aggregation table.

    Returns:
        Validation results for each check.
    """
    results: list[bool] = []

    expected_total = len(clean_dataframe)

    user_total = int(user_summary["total_behaviors"].sum())

    item_total = int(item_summary["total_behaviors"].sum())

    time_total = int(time_summary["total_behaviors"].sum())

    results.append(
        print_result(
            "User table behavior total",
            user_total == expected_total,
            f"{user_total:,} / {expected_total:,}",
        )
    )

    results.append(
        print_result(
            "Item table behavior total",
            item_total == expected_total,
            f"{item_total:,} / {expected_total:,}",
        )
    )

    results.append(
        print_result(
            "Time table behavior total",
            time_total == expected_total,
            f"{time_total:,} / {expected_total:,}",
        )
    )

    clean_purchase_total = int((clean_dataframe["behavior_type"] == 4).sum())

    user_purchase_total = int(user_summary["purchase_count"].sum())

    item_purchase_total = int(item_summary["purchase_count"].sum())

    time_purchase_total = int(time_summary["purchase_count"].sum())

    purchase_consistent = (
        clean_purchase_total
        == user_purchase_total
        == item_purchase_total
        == time_purchase_total
    )

    results.append(
        print_result(
            "Purchase count consistency",
            purchase_consistent,
            (
                f"clean={clean_purchase_total:,}, "
                f"user={user_purchase_total:,}, "
                f"item={item_purchase_total:,}, "
                f"time={time_purchase_total:,}"
            ),
        )
    )

    intermediate_missing = int(
        user_summary.isna().sum().sum()
        + item_summary.isna().sum().sum()
        + time_summary.isna().sum().sum()
    )

    results.append(
        print_result(
            "Intermediate missing values",
            intermediate_missing == 0,
            (f"{intermediate_missing:,} " "missing values"),
        )
    )

    return results


def load_parquet(
    file_path: Path,
) -> pd.DataFrame:
    """Load a Parquet dataset.

    Args:
        file_path: Path to the Parquet file.

    Returns:
        Loaded dataframe.
    """
    dataframe = cast(
        pd.DataFrame,
        pd.read_parquet(file_path),
    )

    return dataframe


def main() -> None:
    """Run all automated data quality checks."""
    print("=" * 60)
    print("DATA QUALITY VALIDATION")
    print("=" * 60)

    required_files = [
        CLEAN_DATA_PATH,
        USER_SUMMARY_PATH,
        ITEM_SUMMARY_PATH,
        TIME_SUMMARY_PATH,
    ]

    missing_files = [path for path in required_files if not path.exists()]

    if missing_files:
        print("[FAIL] Required files")

        for path in missing_files:
            print(f"  Missing: {path}")

        raise FileNotFoundError("Required validation files are missing.")

    print("[PASS] Required files: " "all files found")

    print("\nLoading datasets...")

    clean_dataframe = load_parquet(CLEAN_DATA_PATH)

    user_summary = load_parquet(USER_SUMMARY_PATH)

    item_summary = load_parquet(ITEM_SUMMARY_PATH)

    time_summary = load_parquet(TIME_SUMMARY_PATH)

    print("\nClean dataset checks:")

    clean_results = validate_clean_data(clean_dataframe)

    print("\nIntermediate table checks:")

    intermediate_results = validate_intermediate_tables(
        clean_dataframe,
        user_summary,
        item_summary,
        time_summary,
    )

    all_results = clean_results + intermediate_results

    print("\n" + "=" * 60)

    if all(all_results):
        print("FINAL RESULT: PASS")
        print("All data quality checks passed.")
    else:
        print("FINAL RESULT: FAIL")
        print("One or more data quality " "checks failed.")

    print("=" * 60)


if __name__ == "__main__":
    main()
