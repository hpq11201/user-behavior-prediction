"""Tests for the data cleaning workflow."""

import pandas as pd

from scripts.clean_data import clean_data, standardize_schema


def test_standardize_schema() -> None:
    """Test raw column name standardization."""
    dataframe = pd.DataFrame(
        {
            "time": ["2025-12-01 10"],
            "user_id": [1],
            "item_id": [100],
            "item_category": [20],
            "behavior_type": [1],
        }
    )

    result = standardize_schema(dataframe)

    assert "timestamp" in result.columns
    assert "category_id" in result.columns
    assert "time" not in result.columns
    assert "item_category" not in result.columns


def test_clean_data_removes_duplicates() -> None:
    """Test duplicate business-key removal."""
    dataframe = pd.DataFrame(
        {
            "timestamp": [
                "2025-12-01 10",
                "2025-12-01 10",
            ],
            "user_id": [1, 1],
            "item_id": [100, 100],
            "category_id": [20, 20],
            "behavior_type": [1, 1],
        }
    )

    cleaned, statistics = clean_data(dataframe)

    assert len(cleaned) == 1
    assert statistics["duplicate_rows_removed"] == 1
    assert statistics["final_rows"] == 1


def test_clean_data_removes_invalid_behavior() -> None:
    """Test invalid behavior filtering."""
    dataframe = pd.DataFrame(
        {
            "timestamp": [
                "2025-12-01 10",
                "2025-12-01 11",
            ],
            "user_id": [1, 2],
            "item_id": [100, 200],
            "category_id": [20, 30],
            "behavior_type": [1, 9],
        }
    )

    cleaned, statistics = clean_data(dataframe)

    assert len(cleaned) == 1
    assert statistics["invalid_behavior_rows_removed"] == 1
