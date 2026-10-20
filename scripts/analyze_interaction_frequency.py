"""Analyze hourly interaction frequency in the raw user behavior dataset."""

from pathlib import Path

import pandas as pd

RAW_DATA_PATH = Path("data/raw/user_behavior.csv")

BUSINESS_KEY = [
    "user_id",
    "item_id",
    "behavior_type",
    "time",
]

THRESHOLDS = [10, 20, 50, 100, 200]


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

    dataframe: pd.DataFrame = pd.read_csv(file_path)
    return dataframe


def calculate_interaction_frequency(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """Count repeated interactions for each user-item-behavior-hour group.

    Args:
        dataframe: Raw user behavior DataFrame.

    Returns:
        Frequency count for each business-key group.
    """
    frequency = dataframe.groupby(BUSINESS_KEY, dropna=False).size()
    return frequency


def print_frequency_summary(
    frequency: pd.Series,
    total_rows: int,
) -> None:
    """Print summary statistics for interaction frequency.

    Args:
        frequency: Frequency count for each business-key group.
        total_rows: Total number of rows in the raw dataset.
    """
    print("=" * 70)
    print("INTERACTION FREQUENCY ANALYSIS")
    print("=" * 70)

    print(f"Total rows: {total_rows:,}")
    print(f"Total interaction groups: {len(frequency):,}")
    print(f"Maximum frequency: {int(frequency.max()):,}")
    print()

    for threshold in THRESHOLDS:
        abnormal_groups = frequency[frequency > threshold]

        group_count = len(abnormal_groups)
        row_count = int(abnormal_groups.sum())
        row_ratio = row_count / total_rows * 100

        print(
            f"Frequency > {threshold:3d}: "
            f"{group_count:,} groups, "
            f"{row_count:,} rows "
            f"({row_ratio:.4f}% of total rows)"
        )

    print()
    print("Top 10 highest-frequency groups:")
    print(frequency.sort_values(ascending=False).head(10).to_string())


def main() -> None:
    """Run the interaction frequency analysis."""
    print("Loading raw dataset...")

    dataframe = load_raw_data(RAW_DATA_PATH)

    print(f"Loaded {len(dataframe):,} rows.")
    print("Calculating interaction frequencies...")

    frequency = calculate_interaction_frequency(dataframe)

    print_frequency_summary(
        frequency=frequency,
        total_rows=len(dataframe),
    )


if __name__ == "__main__":
    main()
