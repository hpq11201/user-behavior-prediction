"""Analyze feature effectiveness for the purchase prediction task."""

import pickle
import sqlite3
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = PROJECT_ROOT / "data" / "processed" / "user_behavior.db"

MODEL_PATH = PROJECT_ROOT / "models" / "hist_gradient_boosting.pkl"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "feature_effectiveness"

TABLE_NAME = "modeling_dataset"

FEATURE_COLUMNS = [
    "total_behaviors",
    "view_count",
    "favorite_count",
    "cart_count",
    "active_days",
    "days_since_last_behavior",
    "seen_historically",
    "from_category_recall",
]

TARGET_COLUMN = "label_purchase"

CORRELATION_SAMPLE_SIZE = 200_000
IMPORTANCE_NEGATIVE_SAMPLE_SIZE = 50_000
CHUNK_SIZE = 200_000
RANDOM_STATE = 42


def load_random_feature_sample(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """Load a manageable random-like sample for correlation analysis."""
    query = f"""
    SELECT
        {", ".join(FEATURE_COLUMNS)},
        {TARGET_COLUMN}
    FROM {TABLE_NAME}
    """

    sampled_chunks: list[pd.DataFrame] = []
    total_rows = 10_137_889

    sampling_fraction = CORRELATION_SAMPLE_SIZE / total_rows

    print("Correlation sampling fraction: " f"{sampling_fraction:.6f}")

    for chunk_number, chunk in enumerate(
        pd.read_sql_query(
            query,
            connection,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        sample_size = int(round(len(chunk) * sampling_fraction))

        if sample_size > 0:
            sampled = chunk.sample(
                n=min(
                    sample_size,
                    len(chunk),
                ),
                random_state=(RANDOM_STATE + chunk_number),
            )

            sampled_chunks.append(
                cast(
                    pd.DataFrame,
                    sampled,
                )
            )

        if chunk_number % 10 == 0:
            print("Processed correlation chunk " f"{chunk_number}")

    result = pd.concat(
        sampled_chunks,
        ignore_index=True,
    )

    return cast(
        pd.DataFrame,
        result,
    )


def build_importance_sample(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """Build a sample for permutation importance."""
    columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    column_sql = ", ".join(columns)

    print("Loading all positive validation samples...")

    positive_query = f"""
    SELECT
        {column_sql}
    FROM {TABLE_NAME}
    WHERE {TARGET_COLUMN} = 1
    """

    positive_samples = cast(
        pd.DataFrame,
        pd.read_sql_query(
            positive_query,
            connection,
        ),
    )

    print(f"Positive samples loaded: " f"{len(positive_samples):,}")

    print("Sampling negative validation samples...")

    negative_query = f"""
    SELECT
        {column_sql}
    FROM {TABLE_NAME}
    WHERE {TARGET_COLUMN} = 0
    """

    total_negative_rows = 10_129_242

    sampling_fraction = IMPORTANCE_NEGATIVE_SAMPLE_SIZE / total_negative_rows

    negative_chunks: list[pd.DataFrame] = []

    for chunk_number, chunk in enumerate(
        pd.read_sql_query(
            negative_query,
            connection,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        sample_size = int(round(len(chunk) * sampling_fraction))

        if sample_size > 0:
            sampled = chunk.sample(
                n=min(
                    sample_size,
                    len(chunk),
                ),
                random_state=(RANDOM_STATE + chunk_number),
            )

            negative_chunks.append(
                cast(
                    pd.DataFrame,
                    sampled,
                )
            )

        if chunk_number % 10 == 0:
            print("Processed importance negative chunk " f"{chunk_number}")

    negative_samples = pd.concat(
        negative_chunks,
        ignore_index=True,
    )

    if len(negative_samples) > IMPORTANCE_NEGATIVE_SAMPLE_SIZE:
        negative_samples = negative_samples.sample(
            n=IMPORTANCE_NEGATIVE_SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        )

    analysis_sample = pd.concat(
        [
            positive_samples,
            negative_samples,
        ],
        ignore_index=True,
    )

    analysis_sample = analysis_sample.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    print(f"Importance sample rows: " f"{len(analysis_sample):,}")

    return cast(
        pd.DataFrame,
        analysis_sample,
    )


def create_correlation_heatmap(
    sample: pd.DataFrame,
) -> pd.DataFrame:
    """Create and save feature correlation heatmap."""
    correlation_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    correlation_matrix = sample[correlation_columns].astype(float).corr()

    correlation_path = OUTPUT_DIR / "feature_correlation_matrix.csv"

    correlation_matrix.to_csv(correlation_path)

    figure_path = OUTPUT_DIR / "feature_correlation_heatmap.png"

    plt.figure(figsize=(11, 9))

    image = plt.imshow(
        correlation_matrix,
        aspect="auto",
        vmin=-1,
        vmax=1,
    )

    plt.colorbar(
        image,
        label="Correlation",
    )

    plt.xticks(
        range(len(correlation_columns)),
        correlation_columns,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(correlation_columns)),
        correlation_columns,
    )

    for row_index in range(len(correlation_columns)):
        for column_index in range(len(correlation_columns)):
            value = correlation_matrix.iloc[
                row_index,
                column_index,
            ]

            plt.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )

    plt.title("Feature Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=200,
    )

    plt.close()

    print(f"Saved correlation heatmap: " f"{figure_path}")

    return cast(
        pd.DataFrame,
        correlation_matrix,
    )


def create_class_statistics(
    sample: pd.DataFrame,
) -> pd.DataFrame:
    """Compare feature means between positive and negative samples."""
    grouped = sample.groupby(TARGET_COLUMN)[FEATURE_COLUMNS].mean().transpose()

    grouped.columns = [
        "negative_mean",
        "positive_mean",
    ]

    grouped["positive_to_negative_ratio"] = grouped["positive_mean"] / grouped[
        "negative_mean"
    ].replace(
        0,
        np.nan,
    )

    output_path = OUTPUT_DIR / "feature_class_statistics.csv"

    grouped.to_csv(output_path)

    print(f"Saved class statistics: " f"{output_path}")

    return cast(
        pd.DataFrame,
        grouped,
    )


def create_permutation_importance(
    sample: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate and save model permutation importance."""
    print("Loading HistGradientBoosting model...")

    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)

    x_sample = cast(
        pd.DataFrame,
        sample[FEATURE_COLUMNS].astype(float),
    )

    y_sample = cast(
        pd.Series,
        sample[TARGET_COLUMN].astype(int),
    )

    print("Calculating permutation importance...")

    result = permutation_importance(
        model,
        x_sample,
        y_sample,
        scoring="average_precision",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_table = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance_mean": (result.importances_mean),
            "importance_std": (result.importances_std),
        }
    )

    importance_table = importance_table.sort_values(
        by="importance_mean",
        ascending=False,
    ).reset_index(drop=True)

    csv_path = OUTPUT_DIR / "permutation_feature_importance.csv"

    importance_table.to_csv(
        csv_path,
        index=False,
    )

    figure_path = OUTPUT_DIR / "permutation_feature_importance.png"

    plot_table = importance_table.sort_values(
        by="importance_mean",
        ascending=True,
    )

    plt.figure(figsize=(9, 6))

    plt.barh(
        plot_table["feature"],
        plot_table["importance_mean"],
        xerr=plot_table["importance_std"],
    )

    plt.xlabel("Decrease in PR-AUC after permutation")

    plt.ylabel("Feature")

    plt.title("Permutation Feature Importance")

    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=200,
    )

    plt.close()

    print(f"Saved feature importance chart: " f"{figure_path}")

    return cast(
        pd.DataFrame,
        importance_table,
    )


def main() -> None:
    """Run feature-effectiveness analysis."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Opening SQLite database...")

    with sqlite3.connect(DATABASE_PATH) as connection:
        print()
        print("Building correlation sample...")

        correlation_sample = load_random_feature_sample(connection)

        print(f"Correlation sample rows: " f"{len(correlation_sample):,}")

        print()
        print("Creating correlation heatmap...")

        correlation_matrix = create_correlation_heatmap(correlation_sample)

        print()
        print("Creating feature class statistics...")

        class_statistics = create_class_statistics(correlation_sample)

        print()
        print("Building importance sample...")

        importance_sample = build_importance_sample(connection)

    print()
    print("Calculating feature importance...")

    importance_table = create_permutation_importance(importance_sample)

    print()
    print("=" * 70)
    print("FEATURE EFFECTIVENESS SUMMARY")
    print("=" * 70)

    print()
    print("Correlation with label_purchase:")

    label_correlations = (
        correlation_matrix[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
        .sort_values(
            key=lambda values: (values.abs()),
            ascending=False,
        )
    )

    print(label_correlations.to_string())

    print()
    print("Positive vs negative feature means:")

    print(class_statistics.to_string())

    print()
    print("Permutation importance ranking:")

    print(importance_table.to_string(index=False))

    print()
    print("Feature effectiveness analysis completed.")


if __name__ == "__main__":
    main()
