# mypy: warn_return_any=False
"""Compare imbalance handling strategies for Stage 3."""

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "stage3"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "stage3" / "imbalance"

TRAIN_PATH = DATA_DIR / "train.parquet"
VALIDATION_PATH = DATA_DIR / "validation.parquet"

RANDOM_STATE = 42

NEGATIVE_TO_POSITIVE_RATIO = 20

MAX_SMOTE_ROWS = 300_000


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


def select_feature_columns(
    dataframe: pd.DataFrame,
    label_column: str,
) -> list[str]:
    """Select numeric model features."""
    excluded = {
        label_column,
        "user_id",
        "item_id",
        "category_id",
        "timestamp",
        "event_date",
    }

    numeric_columns = dataframe.select_dtypes(
        include=[
            np.number,
        ]
    ).columns.tolist()

    features = [column for column in numeric_columns if column not in excluded]

    if not features:
        raise ValueError("No usable numeric features found.")

    return features


def prepare_xy(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare X and y."""
    x = dataframe[feature_columns].copy()

    x = x.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    ).fillna(0)

    y = dataframe[label_column].astype(int)

    return (
        cast(
            pd.DataFrame,
            x,
        ),
        cast(
            pd.Series,
            y,
        ),
    )


def undersample_training_data(
    x: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create deterministic negative undersampling."""
    print("Building undersampled training data...")

    positives = x[y == 1].copy()

    negative_pool = x[y == 0].copy()

    positive_y = y[y == 1].copy()

    target_negatives = min(
        len(negative_pool),
        len(positives) * NEGATIVE_TO_POSITIVE_RATIO,
    )

    sampled_negatives = negative_pool.sample(
        n=target_negatives,
        random_state=RANDOM_STATE,
    )

    sampled_negative_y = y.loc[sampled_negatives.index]

    combined_x = pd.concat(
        [
            positives,
            sampled_negatives,
        ],
        axis=0,
    )

    combined_y = pd.concat(
        [
            positive_y,
            sampled_negative_y,
        ],
        axis=0,
    )

    shuffled_index = combined_x.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).index

    combined_x = combined_x.loc[shuffled_index]

    combined_y = combined_y.loc[shuffled_index]

    return (
        cast(
            pd.DataFrame,
            combined_x,
        ),
        cast(
            pd.Series,
            combined_y,
        ),
    )


def build_smote_training_sample(
    x: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build manageable sample for SMOTE experiment."""
    print("Building SMOTE experiment sample...")

    positives = x[y == 1].copy()

    positive_y = y[y == 1].copy()

    negatives = x[y == 0].copy()

    max_negatives = max(
        0,
        MAX_SMOTE_ROWS - len(positives),
    )

    sample_negatives = min(
        len(negatives),
        max_negatives,
    )

    sampled_negatives = negatives.sample(
        n=sample_negatives,
        random_state=RANDOM_STATE,
    )

    sampled_negative_y = y.loc[sampled_negatives.index]

    sample_x = pd.concat(
        [
            positives,
            sampled_negatives,
        ],
        axis=0,
    )

    sample_y = pd.concat(
        [
            positive_y,
            sampled_negative_y,
        ],
        axis=0,
    )

    shuffled_index = sample_x.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).index

    sample_x = sample_x.loc[shuffled_index]

    sample_y = sample_y.loc[shuffled_index]

    smote = SMOTE(
        random_state=RANDOM_STATE,
        sampling_strategy=0.20,
    )

    resampled_x, resampled_y = smote.fit_resample(
        sample_x,
        sample_y,
    )

    resampled_x = pd.DataFrame(
        resampled_x,
        columns=sample_x.columns,
    )

    resampled_y = pd.Series(
        resampled_y,
        name=sample_y.name,
    )

    return (
        resampled_x,
        resampled_y,
    )


def build_pipeline(
    class_weight: str | None,
) -> Pipeline:
    """Build logistic regression pipeline."""
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="liblinear",
    )

    pipeline = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


def evaluate_model(
    name: str,
    model: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> dict[str, float | int | str]:
    """Train and evaluate one imbalance strategy."""
    print()
    print("=" * 80)

    print(f"Training strategy: {name}")

    print("=" * 80)

    print(f"Training rows: {len(x_train):,}")

    train_positives = int(y_train.sum())

    train_positive_rate = float(y_train.mean())

    print(f"Training positives: {train_positives:,}")

    print("Training positive rate: " f"{train_positive_rate:.6%}")

    model.fit(
        x_train,
        y_train,
    )

    probabilities = model.predict_proba(x_validation)[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    roc_auc = roc_auc_score(
        y_validation,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_validation,
        probabilities,
    )

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    predicted_positive_rate = float(predictions.mean())

    print(f"ROC-AUC:   {roc_auc:.6f}")

    print(f"PR-AUC:    {pr_auc:.6f}")

    print(f"Precision: {precision:.6f}")

    print(f"Recall:    {recall:.6f}")

    print(f"F1:        {f1:.6f}")

    return {
        "strategy": name,
        "train_rows": len(x_train),
        "train_positives": train_positives,
        "train_positive_rate": train_positive_rate,
        "validation_rows": len(x_validation),
        "validation_positive_rate": float(y_validation.mean()),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "precision_at_0_5": float(precision),
        "recall_at_0_5": float(recall),
        "f1_at_0_5": float(f1),
        "predicted_positive_rate": predicted_positive_rate,
    }


def create_comparison_chart(
    results: pd.DataFrame,
) -> None:
    """Create comparison chart."""
    metrics = [
        "roc_auc",
        "pr_auc",
        "precision_at_0_5",
        "recall_at_0_5",
        "f1_at_0_5",
    ]

    positions = np.arange(len(results))

    width = 0.15

    plt.figure(figsize=(13, 7))

    for index, metric in enumerate(metrics):
        plt.bar(
            positions + index * width,
            results[metric],
            width=width,
            label=metric,
        )

    center_offset = width * (len(metrics) - 1) / 2

    plt.xticks(
        positions + center_offset,
        results["strategy"].astype(str).tolist(),
    )

    plt.ylabel("Metric Value")

    plt.xlabel("Imbalance Strategy")

    plt.title("Imbalance Strategy Comparison " "with Logistic Regression")

    plt.legend()

    plt.tight_layout()

    output_path = OUTPUT_DIR / "imbalance_strategy_comparison.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved chart: {output_path}")


def main() -> None:
    """Run imbalance strategy comparison."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading Stage 3 datasets...")

    train_data = pd.read_parquet(TRAIN_PATH)

    validation_data = pd.read_parquet(VALIDATION_PATH)

    label_column = find_label_column(train_data)

    feature_columns = select_feature_columns(
        train_data,
        label_column,
    )

    print(f"Target column: {label_column}")

    print(f"Number of features: {len(feature_columns)}")

    print("Features:")

    for feature in feature_columns:
        print(f"  - {feature}")

    x_train, y_train = prepare_xy(
        train_data,
        feature_columns,
        label_column,
    )

    (
        x_validation,
        y_validation,
    ) = prepare_xy(
        validation_data,
        feature_columns,
        label_column,
    )

    results = []

    undersampled_x, undersampled_y = undersample_training_data(
        x_train,
        y_train,
    )

    undersample_model = build_pipeline(
        class_weight=None,
    )

    results.append(
        evaluate_model(
            "Undersampling 20:1",
            undersample_model,
            undersampled_x,
            undersampled_y,
            x_validation,
            y_validation,
        )
    )

    class_weight_model = build_pipeline(
        class_weight="balanced",
    )

    results.append(
        evaluate_model(
            "Class Weight",
            class_weight_model,
            x_train,
            y_train,
            x_validation,
            y_validation,
        )
    )

    smote_x, smote_y = build_smote_training_sample(
        x_train,
        y_train,
    )

    smote_model = build_pipeline(
        class_weight=None,
    )

    results.append(
        evaluate_model(
            "SMOTE",
            smote_model,
            smote_x,
            smote_y,
            x_validation,
            y_validation,
        )
    )

    result_table = pd.DataFrame(results)

    result_table = result_table.sort_values(
        "pr_auc",
        ascending=False,
    ).reset_index(drop=True)

    output_csv = OUTPUT_DIR / "imbalance_strategy_comparison.csv"

    result_table.to_csv(
        output_csv,
        index=False,
    )

    print()
    print("=" * 100)

    print("IMBALANCE STRATEGY COMPARISON")

    print("=" * 100)

    print(result_table.to_string(index=False))

    print()
    print(f"Saved comparison table: {output_csv}")

    create_comparison_chart(result_table)

    best_strategy = str(result_table.iloc[0]["strategy"])

    best_pr_auc = float(result_table.iloc[0]["pr_auc"])

    summary_path = OUTPUT_DIR / "imbalance_strategy_summary.txt"

    summary_text = (
        "Stage 3 imbalance strategy comparison\n"
        "=====================================\n\n"
        f"Primary selection metric: PR-AUC\n"
        f"Best strategy: {best_strategy}\n"
        f"Best validation PR-AUC: "
        f"{best_pr_auc:.6f}\n\n"
        "Because the purchase target is extremely "
        "imbalanced, PR-AUC is treated as the primary "
        "selection metric. ROC-AUC and threshold-based "
        "Precision/Recall/F1 are reported as supporting "
        "metrics.\n"
    )

    summary_path.write_text(
        summary_text,
        encoding="utf-8",
    )

    print(f"Saved summary: {summary_path}")

    print()
    print("=" * 80)

    print("STAGE 3 IMBALANCE COMPARISON COMPLETED")

    print("=" * 80)


if __name__ == "__main__":
    main()
