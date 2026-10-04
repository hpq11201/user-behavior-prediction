"""Tests for intermediate aggregation tables."""

import pandas as pd

from scripts.build_intermediate_tables import (
    build_item_summary,
    build_time_summary,
    build_user_summary,
)


def build_sample_dataframe() -> pd.DataFrame:
    """Create a small cleaned dataset for testing."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-12-01 10:00:00",
                    "2025-12-01 11:00:00",
                    "2025-12-01 12:00:00",
                ]
            ),
            "user_id": [1, 1, 2],
            "item_id": [100, 100, 200],
            "category_id": [10, 10, 20],
            "behavior_type": [1, 4, 1],
        }
    )


def test_user_summary_total_behavior_count() -> None:
    """Test user-level behavior aggregation."""
    dataframe = build_sample_dataframe()

    result = build_user_summary(dataframe)

    assert result["total_behaviors"].sum() == 3
    assert result["purchase_count"].sum() == 1


def test_item_summary_total_behavior_count() -> None:
    """Test item-level behavior aggregation."""
    dataframe = build_sample_dataframe()

    result = build_item_summary(dataframe)

    assert result["total_behaviors"].sum() == 3
    assert result["purchase_count"].sum() == 1


def test_time_summary_total_behavior_count() -> None:
    """Test time-level behavior aggregation."""
    dataframe = build_sample_dataframe()

    result = build_time_summary(dataframe)

    assert result["total_behaviors"].sum() == 3
    assert result["purchase_count"].sum() == 1
