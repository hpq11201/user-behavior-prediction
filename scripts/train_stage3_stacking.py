# mypy: warn_return_any=False
"""Train leakage-safe Stage 3 stacking model."""

from pathlib import Path
from typing import Any, cast

import joblib  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "stage3"

TRAIN_PATH = DATA_DIR / "train.parquet"

VALIDATION_PATH = DATA_DIR / "validation.parquet"

LIGHTGBM_PREDICTION_PATH = (
    PROJECT_ROOT
    / "reports"
    / "stage3"
    / "optuna"
    / "tuned_validation_predictions.parquet"
)

DIN_PREDICTION_PATH = (
    PROJECT_ROOT / "reports" / "stage3" / "din" / "din_validation_predictions.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "stage3" / "stacking"

MODEL_DIR = PROJECT_ROOT / "models" / "stage3"

RANDOM_STATE = 42

NEGATIVE_TO_POSITIVE_RATIO = 20

META_TEST_SIZE = 0.50

THRESHOLD = 0.5


def find_label_column(
    dataframe: pd.DataFrame,
) -> str:
    """Find target label column."""
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
    """Select numeric features for base LR model."""
    excluded = {
        label_column,
        "user_id",
        "item_id",
        "category_id",
        "timestamp",
        "event_date",
    }

    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()

    features = [column for column in numeric_columns if column not in excluded]

    if not features:
        raise ValueError("No usable numeric features found.")

    return features


def prepare_xy(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix and labels."""
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
        cast(pd.DataFrame, x),
        cast(pd.Series, y),
    )


def undersample_training_data(
    x: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply the established 20:1 undersampling strategy."""
    print()
    print("Building 20:1 undersampled LR training data...")

    positive_indices = y[y == 1].index

    negative_indices = y[y == 0].index

    positive_count = len(positive_indices)

    target_negative_count = min(
        len(negative_indices),
        positive_count * NEGATIVE_TO_POSITIVE_RATIO,
    )

    sampled_negative_indices = (
        y.loc[negative_indices]
        .sample(
            n=target_negative_count,
            random_state=RANDOM_STATE,
        )
        .index
    )

    selected_indices = positive_indices.tolist() + sampled_negative_indices.tolist()

    sampled_x = x.loc[selected_indices].copy()

    sampled_y = y.loc[selected_indices].copy()

    shuffled_indices = sampled_x.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).index

    sampled_x = sampled_x.loc[shuffled_indices].reset_index(drop=True)

    sampled_y = sampled_y.loc[shuffled_indices].reset_index(drop=True)

    print(f"LR training rows: " f"{len(sampled_x):,}")

    print(f"LR positives: " f"{int(sampled_y.sum()):,}")

    print("LR positive rate: " f"{float(sampled_y.mean()):.6%}")

    return (
        sampled_x,
        sampled_y,
    )


def train_base_logistic_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
) -> tuple[
    Pipeline,
    np.ndarray,
]:
    """Train LR base model and generate validation probabilities."""
    print()
    print("=" * 80)

    print("TRAINING BASE LOGISTIC REGRESSION")

    print("=" * 80)

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model.fit(
        x_train,
        y_train,
    )

    probability_matrix = np.asarray(model.predict_proba(x_validation))

    probabilities = probability_matrix[:, 1]

    return (
        model,
        probabilities,
    )


def validate_prediction_files(
    validation_data: pd.DataFrame,
    label_column: str,
    lightgbm_predictions: pd.DataFrame,
    din_predictions: pd.DataFrame,
) -> None:
    """Check that saved prediction files match validation data."""
    expected_rows = len(validation_data)

    if len(lightgbm_predictions) != expected_rows:
        raise ValueError(
            "LightGBM prediction row count does not " "match validation dataset."
        )

    if len(din_predictions) != expected_rows:
        raise ValueError(
            "DIN prediction row count does not " "match validation dataset."
        )

    validation_labels = validation_data[label_column].astype(int).to_numpy()

    lightgbm_labels = lightgbm_predictions["label_purchase"].astype(int).to_numpy()

    din_labels = din_predictions["label_purchase"].astype(int).to_numpy()

    if not np.array_equal(
        validation_labels,
        lightgbm_labels,
    ):
        raise ValueError("LightGBM labels are not aligned " "with validation data.")

    if not np.array_equal(
        validation_labels,
        din_labels,
    ):
        raise ValueError("DIN labels are not aligned " "with validation data.")

    print("Prediction alignment validation passed.")


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Calculate evaluation metrics."""
    predictions = (probabilities >= THRESHOLD).astype(int)

    return {
        "roc_auc": float(
            roc_auc_score(
                labels,
                probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                labels,
                probabilities,
            )
        ),
        "precision_at_0_5": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall_at_0_5": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "f1_at_0_5": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "predicted_positive_rate": float(predictions.mean()),
    }


def build_meta_dataset(
    lr_probabilities: np.ndarray,
    lightgbm_probabilities: np.ndarray,
    din_probabilities: np.ndarray,
) -> pd.DataFrame:
    """Build stacking meta-feature dataset."""
    return pd.DataFrame(
        {
            "lr_probability": (lr_probabilities),
            "lightgbm_probability": (lightgbm_probabilities),
            "din_probability": (din_probabilities),
        }
    )


def train_meta_model(
    meta_features: pd.DataFrame,
    labels: np.ndarray,
) -> tuple[
    LogisticRegression,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Train leakage-safe logistic meta learner."""
    indices = np.arange(len(labels))

    (
        meta_train_indices,
        meta_test_indices,
    ) = train_test_split(
        indices,
        test_size=META_TEST_SIZE,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    meta_train_x = meta_features.iloc[meta_train_indices]

    meta_train_y = labels[meta_train_indices]

    meta_test_x = meta_features.iloc[meta_test_indices]

    print()
    print("=" * 80)

    print("STACKING META SPLIT")

    print("=" * 80)

    print(f"Meta-train rows: " f"{len(meta_train_indices):,}")

    print(f"Meta-test rows: " f"{len(meta_test_indices):,}")

    print("Meta-train positive rate: " f"{float(meta_train_y.mean()):.6%}")

    print("Meta-test positive rate: " f"{float(labels[meta_test_indices].mean()):.6%}")

    model = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    model.fit(
        meta_train_x,
        meta_train_y,
    )

    probability_matrix = np.asarray(model.predict_proba(meta_test_x))

    probabilities = probability_matrix[:, 1]

    return (
        model,
        meta_train_indices,
        meta_test_indices,
        probabilities,
    )


def build_comparison(
    labels: np.ndarray,
    meta_test_indices: np.ndarray,
    lr_probabilities: np.ndarray,
    lightgbm_probabilities: np.ndarray,
    din_probabilities: np.ndarray,
    stacking_probabilities: np.ndarray,
) -> pd.DataFrame:
    """Compare stacking against base models on same meta-test."""
    test_labels = labels[meta_test_indices]

    model_probabilities = {
        "Stacking": (stacking_probabilities),
        "LightGBM Optuna": (lightgbm_probabilities[meta_test_indices]),
        "Simplified DIN": (din_probabilities[meta_test_indices]),
        "Logistic Regression": (lr_probabilities[meta_test_indices]),
    }

    rows: list[dict[str, Any]] = []

    for model_name, probabilities in model_probabilities.items():
        metrics = calculate_metrics(
            test_labels,
            probabilities,
        )

        rows.append(
            {
                "model": model_name,
                **metrics,
            }
        )

    comparison = pd.DataFrame(rows)

    comparison = comparison.sort_values(
        "pr_auc",
        ascending=False,
    ).reset_index(drop=True)

    return comparison


def create_comparison_chart(
    comparison: pd.DataFrame,
) -> None:
    """Create PR-AUC model comparison chart."""
    positions = np.arange(len(comparison))

    plt.figure(figsize=(10, 6))

    plt.bar(
        positions,
        comparison["pr_auc"],
    )

    plt.xticks(
        positions,
        comparison["model"].astype(str).tolist(),
        rotation=15,
    )

    plt.ylabel("Meta-test PR-AUC")

    plt.xlabel("Model")

    plt.title("Stage 3 Stacking Model Comparison")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "stacking_model_comparison.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved chart: {output_path}")


def save_predictions(
    validation_data: pd.DataFrame,
    label_column: str,
    meta_test_indices: np.ndarray,
    lr_probabilities: np.ndarray,
    lightgbm_probabilities: np.ndarray,
    din_probabilities: np.ndarray,
    stacking_probabilities: np.ndarray,
) -> None:
    """Save meta-test predictions."""
    identifier_columns = [
        column
        for column in [
            "user_id",
            "item_id",
            "category_id",
        ]
        if column in validation_data.columns
    ]

    output = (
        validation_data.iloc[meta_test_indices][identifier_columns]
        .copy()
        .reset_index(drop=True)
    )

    output["label_purchase"] = (
        validation_data.iloc[meta_test_indices][label_column].astype(int).to_numpy()
    )

    output["lr_probability"] = lr_probabilities[meta_test_indices]

    output["lightgbm_probability"] = lightgbm_probabilities[meta_test_indices]

    output["din_probability"] = din_probabilities[meta_test_indices]

    output["stacking_probability"] = stacking_probabilities

    output_path = OUTPUT_DIR / "stacking_meta_test_predictions.parquet"

    output.to_parquet(
        output_path,
        index=False,
    )

    print(f"Saved predictions: {output_path}")


def save_meta_coefficients(
    model: LogisticRegression,
) -> None:
    """Save meta learner coefficients."""
    coefficients = np.asarray(model.coef_).reshape(-1)

    result = pd.DataFrame(
        {
            "base_model": [
                "Logistic Regression",
                "LightGBM Optuna",
                "Simplified DIN",
            ],
            "coefficient": coefficients,
        }
    )

    output_path = OUTPUT_DIR / "stacking_meta_coefficients.csv"

    result.to_csv(
        output_path,
        index=False,
    )

    print()
    print("Meta learner coefficients:")

    print(result.to_string(index=False))

    print(f"Saved coefficients: {output_path}")


def main() -> None:
    """Run leakage-safe Stage 3 stacking."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
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

    print(f"Target column: " f"{label_column}")

    print(f"Number of LR features: " f"{len(feature_columns)}")

    x_train, y_train = prepare_xy(
        train_data,
        feature_columns,
        label_column,
    )

    (
        x_validation,
        _,
    ) = prepare_xy(
        validation_data,
        feature_columns,
        label_column,
    )

    (
        x_train,
        y_train,
    ) = undersample_training_data(
        x_train,
        y_train,
    )

    (
        lr_model,
        lr_probabilities,
    ) = train_base_logistic_regression(
        x_train,
        y_train,
        x_validation,
    )

    print()
    print("Loading saved LightGBM " "and DIN validation predictions...")

    lightgbm_data = pd.read_parquet(LIGHTGBM_PREDICTION_PATH)

    din_data = pd.read_parquet(DIN_PREDICTION_PATH)

    validate_prediction_files(
        validation_data,
        label_column,
        lightgbm_data,
        din_data,
    )

    lightgbm_probabilities = (
        lightgbm_data["lightgbm_tuned_probability"].astype(float).to_numpy()
    )

    din_probabilities = din_data["din_probability"].astype(float).to_numpy()

    labels = validation_data[label_column].astype(int).to_numpy()

    meta_features = build_meta_dataset(
        lr_probabilities,
        lightgbm_probabilities,
        din_probabilities,
    )

    (
        meta_model,
        _,
        meta_test_indices,
        stacking_probabilities,
    ) = train_meta_model(
        meta_features,
        labels,
    )

    comparison = build_comparison(
        labels,
        meta_test_indices,
        lr_probabilities,
        lightgbm_probabilities,
        din_probabilities,
        stacking_probabilities,
    )

    comparison_path = OUTPUT_DIR / "stacking_model_comparison.csv"

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    create_comparison_chart(comparison)

    save_predictions(
        validation_data,
        label_column,
        meta_test_indices,
        lr_probabilities,
        lightgbm_probabilities,
        din_probabilities,
        stacking_probabilities,
    )

    save_meta_coefficients(meta_model)

    joblib.dump(
        meta_model,
        MODEL_DIR / "stacking_meta_logistic.pkl",
    )

    joblib.dump(
        lr_model,
        MODEL_DIR / "stacking_base_logistic.pkl",
    )

    print()
    print("=" * 105)

    print("STAGE 3 STACKING MODEL COMPARISON")

    print("=" * 105)

    print(comparison.to_string(index=False))

    best_model = str(comparison.iloc[0]["model"])

    best_pr_auc = float(comparison.iloc[0]["pr_auc"])

    stacking_row = comparison[comparison["model"] == "Stacking"]

    stacking_pr_auc = float(stacking_row.iloc[0]["pr_auc"])

    summary_path = OUTPUT_DIR / "stacking_summary.txt"

    summary = (
        "Stage 3 Stacking Summary\n"
        "========================\n\n"
        "Base models:\n"
        "- Logistic Regression\n"
        "- Optuna-tuned LightGBM\n"
        "- Simplified DIN\n\n"
        "Meta learner:\n"
        "- Logistic Regression\n\n"
        "Leakage control:\n"
        "The original validation dataset was split "
        "into separate meta-train and meta-test "
        "subsets using stratified sampling. The meta "
        "learner was fitted only on meta-train and "
        "all reported stacking metrics were calculated "
        "only on meta-test.\n\n"
        "Primary metric: PR-AUC\n\n"
        f"Stacking meta-test PR-AUC: "
        f"{stacking_pr_auc:.6f}\n"
        f"Best model on meta-test: "
        f"{best_model}\n"
        f"Best meta-test PR-AUC: "
        f"{best_pr_auc:.6f}\n"
    )

    summary_path.write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print(f"Saved comparison: " f"{comparison_path}")

    print(f"Saved summary: " f"{summary_path}")

    print()
    print("=" * 80)

    print("STAGE 3 STACKING TRAINING COMPLETED")

    print("=" * 80)


if __name__ == "__main__":
    main()
