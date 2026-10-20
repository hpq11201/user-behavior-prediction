"""Unit tests for the data cleaning workflow."""

from typing import cast

import pandas as pd

from scripts.clean_data import (
    MAX_INTERACTIONS_PER_HOUR,
    clean_data,
    remove_high_frequency_interactions,
    standardize_schema,
)


def build_test_dataframe(
    repeated_rows: int,
    behavior_type: int = 1,
) -> pd.DataFrame:
    """Build a small user behavior dataframe for testing.

    Args:
        repeated_rows: Number of repeated business-key rows.
        behavior_type: Behavior type used in the repeated rows.

    Returns:
        Test dataframe containing repeated interactions.
    """
    dataframe = cast(
        pd.DataFrame,
        pd.DataFrame(
            {
                "timestamp": ["2025-12-01 10" for _ in range(repeated_rows)],
                "user_id": [1 for _ in range(repeated_rows)],
                "item_id": [100 for _ in range(repeated_rows)],
                "category_id": [10 for _ in range(repeated_rows)],
                "behavior_type": [behavior_type for _ in range(repeated_rows)],
            }
        ),
    )

    return dataframe


def test_standardize_schema() -> None:
    """Test raw column names are standardized."""
    dataframe = cast(
        pd.DataFrame,
        pd.DataFrame(
            {
                "time": ["2025-12-01 10"],
                "user_id": [1],
                "item_id": [100],
                "item_category": [10],
                "behavior_type": [1],
            }
        ),
    )

    result = standardize_schema(dataframe)

    assert "timestamp" in result.columns
    assert "category_id" in result.columns
    assert "time" not in result.columns
    assert "item_category" not in result.columns


def test_normal_repeated_interactions_are_kept() -> None:
    """Test normal repeated interactions are preserved."""
    dataframe = build_test_dataframe(repeated_rows=5)

    cleaned, removed_rows, removed_groups = remove_high_frequency_interactions(
        dataframe=dataframe,
        max_interactions_per_hour=(MAX_INTERACTIONS_PER_HOUR),
    )

    assert len(cleaned) == 5
    assert removed_rows == 0
    assert removed_groups == 0


def test_threshold_frequency_is_kept() -> None:
    """Test interactions exactly at the threshold are kept."""
    dataframe = build_test_dataframe(repeated_rows=MAX_INTERACTIONS_PER_HOUR)

    cleaned, removed_rows, removed_groups = remove_high_frequency_interactions(
        dataframe=dataframe,
        max_interactions_per_hour=(MAX_INTERACTIONS_PER_HOUR),
    )

    assert len(cleaned) == MAX_INTERACTIONS_PER_HOUR
    assert removed_rows == 0
    assert removed_groups == 0


def test_high_frequency_interactions_are_removed() -> None:
    """Test groups above the threshold are removed completely."""
    repeated_rows = MAX_INTERACTIONS_PER_HOUR + 1

    dataframe = build_test_dataframe(repeated_rows=repeated_rows)

    cleaned, removed_rows, removed_groups = remove_high_frequency_interactions(
        dataframe=dataframe,
        max_interactions_per_hour=(MAX_INTERACTIONS_PER_HOUR),
    )

    assert len(cleaned) == 0
    assert removed_rows == repeated_rows
    assert removed_groups == 1


def test_clean_data_removes_invalid_behavior() -> None:
    """Test invalid behavior values are removed."""
    dataframe = cast(
        pd.DataFrame,
        pd.DataFrame(
            {
                "timestamp": [
                    "2025-12-01 10",
                    "2025-12-01 11",
                ],
                "user_id": [
                    1,
                    2,
                ],
                "item_id": [
                    100,
                    200,
                ],
                "category_id": [
                    10,
                    20,
                ],
                "behavior_type": [
                    1,
                    9,
                ],
            }
        ),
    )

    cleaned, summary = clean_data(dataframe)

    assert len(cleaned) == 1
    assert summary["invalid_behavior_rows_removed"] == 1


def test_clean_data_keeps_valid_repeated_rows() -> None:
    """Test full cleaning keeps valid repeated rows."""
    dataframe = build_test_dataframe(repeated_rows=10)

    cleaned, summary = clean_data(dataframe)

    assert len(cleaned) == 10
    assert summary["high_frequency_rows_removed"] == 0
    assert summary["high_frequency_groups_removed"] == 0
