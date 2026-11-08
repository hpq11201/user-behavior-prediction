# mypy: warn_return_any=False
"""Train Stage 3 tree-based models with 20:1 undersampling."""

import time
from pathlib import Path
from typing import Any, cast

import joblib  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import (  # type: ignore[import-untyped]
    LGBMClassifier,
    early_stopping,
    log_evaluation,
)
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "stage3"

TRAIN_PATH = DATA_DIR / "train.parquet"
VALIDATION_PATH = DATA_DIR / "validation.parquet"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "stage3" / "tree_models"

MODEL_DIR = PROJECT_ROOT / "models" / "stage3"

IMBALANCE_RESULT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "stage3"
    / "imbalance"
    / "imbalance_strategy_comparison.csv"
)

RANDOM_STATE = 42
NEGATIVE_TO_POSITIVE_RATIO = 20
THRESHOLD = 0.5


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
    """Prepare model features and labels."""
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
    """Apply 20:1 negative undersampling."""
    print()
    print("Building 20:1 undersampled " "training dataset...")

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

    sampled = x.loc[selected_indices].copy()

    sampled_y = y.loc[selected_indices].copy()

    shuffled_indices = sampled.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).index

    sampled = sampled.loc[shuffled_indices].reset_index(drop=True)

    sampled_y = sampled_y.loc[shuffled_indices].reset_index(drop=True)

    print(f"Training rows: {len(sampled):,}")

    print("Training positives: " f"{int(sampled_y.sum()):,}")

    print("Training negatives: " f"{int((sampled_y == 0).sum()):,}")

    print("Training positive rate: " f"{float(sampled_y.mean()):.6%}")

    return (
        sampled,
        sampled_y,
    )


def calculate_metrics(
    labels: pd.Series,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Calculate validation metrics."""
    predictions = (probabilities >= THRESHOLD).astype(int)

    roc_auc = roc_auc_score(
        labels,
        probabilities,
    )

    pr_auc = average_precision_score(
        labels,
        probabilities,
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0,
    )

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "precision_at_0_5": float(precision),
        "recall_at_0_5": float(recall),
        "f1_at_0_5": float(f1),
        "predicted_positive_rate": float(predictions.mean()),
    }


def print_metrics(
    model_name: str,
    metrics: dict[str, float],
    training_seconds: float,
) -> None:
    """Print model metrics."""
    print()
    print("=" * 80)
    print(f"{model_name} VALIDATION")
    print("=" * 80)

    print(f"ROC-AUC:   " f"{metrics['roc_auc']:.6f}")

    print(f"PR-AUC:    " f"{metrics['pr_auc']:.6f}")

    print(f"Precision: " f"{metrics['precision_at_0_5']:.6f}")

    print(f"Recall:    " f"{metrics['recall_at_0_5']:.6f}")

    print(f"F1:        " f"{metrics['f1_at_0_5']:.6f}")

    print("Predicted positive rate: " f"{metrics['predicted_positive_rate']:.6%}")

    print(f"Training time: " f"{training_seconds:.2f} seconds")


def train_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[
    XGBClassifier,
    np.ndarray,
    float,
]:
    """Train baseline XGBoost model."""
    print()
    print("=" * 80)
    print("TRAINING XGBOOST")
    print("=" * 80)

    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    start_time = time.perf_counter()

    model.fit(
        x_train,
        y_train,
        eval_set=[
            (
                x_validation,
                y_validation,
            )
        ],
        verbose=False,
    )

    training_seconds = time.perf_counter() - start_time

    prediction_matrix = np.asarray(model.predict_proba(x_validation))

    probabilities = prediction_matrix[:, 1]

    return (
        model,
        probabilities,
        training_seconds,
    )


def train_lightgbm(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[
    LGBMClassifier,
    np.ndarray,
    float,
]:
    """Train baseline LightGBM model."""
    print()
    print("=" * 80)
    print("TRAINING LIGHTGBM")
    print("=" * 80)

    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    start_time = time.perf_counter()

    model.fit(
        x_train,
        y_train,
        eval_set=[
            (
                x_validation,
                y_validation,
            )
        ],
        eval_metric="average_precision",
        callbacks=[
            early_stopping(
                stopping_rounds=50,
                verbose=False,
            ),
            log_evaluation(period=0),
        ],
    )

    training_seconds = time.perf_counter() - start_time

    prediction_matrix = np.asarray(model.predict_proba(x_validation))

    probabilities = prediction_matrix[:, 1]

    return (
        model,
        probabilities,
        training_seconds,
    )


def save_validation_predictions(
    validation_data: pd.DataFrame,
    label_column: str,
    xgb_probabilities: np.ndarray,
    lgbm_probabilities: np.ndarray,
) -> None:
    """Save validation predictions."""
    output_columns = [
        column
        for column in [
            "user_id",
            "item_id",
            "category_id",
        ]
        if column in validation_data.columns
    ]

    predictions = validation_data[output_columns].copy()

    predictions["label_purchase"] = validation_data[label_column].astype(int).to_numpy()

    predictions["xgboost_probability"] = xgb_probabilities

    predictions["lightgbm_probability"] = lgbm_probabilities

    output_path = OUTPUT_DIR / "tree_model_validation_predictions.parquet"

    predictions.to_parquet(
        output_path,
        index=False,
    )

    print(f"Saved predictions: {output_path}")


def load_logistic_baseline() -> dict[str, Any] | None:
    """Load previous logistic undersampling result."""
    if not IMBALANCE_RESULT_PATH.exists():
        return None

    results = pd.read_csv(IMBALANCE_RESULT_PATH)

    matching = results[results["strategy"].astype(str) == "Undersampling 20:1"]

    if matching.empty:
        return None

    row = matching.iloc[0]

    return {
        "model": "Logistic Regression",
        "roc_auc": float(row["roc_auc"]),
        "pr_auc": float(row["pr_auc"]),
        "precision_at_0_5": float(row["precision_at_0_5"]),
        "recall_at_0_5": float(row["recall_at_0_5"]),
        "f1_at_0_5": float(row["f1_at_0_5"]),
        "training_seconds": np.nan,
    }


def create_model_comparison_chart(
    results: pd.DataFrame,
) -> None:
    """Create tree-model comparison chart."""
    metrics = [
        "roc_auc",
        "pr_auc",
        "precision_at_0_5",
        "recall_at_0_5",
        "f1_at_0_5",
    ]

    positions = np.arange(len(results))

    width = 0.15

    plt.figure(figsize=(14, 7))

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
        results["model"].astype(str).tolist(),
    )

    plt.ylabel("Metric Value")

    plt.xlabel("Model")

    plt.title("Stage 3 Traditional Model Comparison")

    plt.legend()

    plt.tight_layout()

    output_path = OUTPUT_DIR / "traditional_model_comparison.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved chart: {output_path}")


def save_feature_importance(
    model_name: str,
    model: Any,
    feature_columns: list[str],
) -> None:
    """Save model feature importance."""
    if not hasattr(
        model,
        "feature_importances_",
    ):
        return

    importances = np.asarray(model.feature_importances_)

    result = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importances,
        }
    )

    result = result.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    csv_path = OUTPUT_DIR / f"{model_name}_feature_importance.csv"

    result.to_csv(
        csv_path,
        index=False,
    )

    top_result = result.head(20).sort_values("importance")

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_result["feature"],
        top_result["importance"],
    )

    plt.xlabel("Feature Importance")

    plt.ylabel("Feature")

    plt.title(f"{model_name} Feature Importance")

    plt.tight_layout()

    chart_path = OUTPUT_DIR / f"{model_name}_feature_importance.png"

    plt.savefig(
        chart_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved importance: {csv_path}")

    print(f"Saved chart: {chart_path}")


def main() -> None:
    """Train Stage 3 traditional tree models."""
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

    print(f"Target column: {label_column}")

    print(f"Number of features: " f"{len(feature_columns)}")

    print()
    print("Features:")

    for feature in feature_columns:
        print(f"  - {feature}")

    x_train, y_train = prepare_xy(
        train_data,
        feature_columns,
        label_column,
    )

    x_validation, y_validation = prepare_xy(
        validation_data,
        feature_columns,
        label_column,
    )

    x_train, y_train = undersample_training_data(
        x_train,
        y_train,
    )

    (
        xgb_model,
        xgb_probabilities,
        xgb_time,
    ) = train_xgboost(
        x_train,
        y_train,
        x_validation,
        y_validation,
    )

    xgb_metrics = calculate_metrics(
        y_validation,
        xgb_probabilities,
    )

    print_metrics(
        "XGBoost",
        xgb_metrics,
        xgb_time,
    )

    (
        lgbm_model,
        lgbm_probabilities,
        lgbm_time,
    ) = train_lightgbm(
        x_train,
        y_train,
        x_validation,
        y_validation,
    )

    lgbm_metrics = calculate_metrics(
        y_validation,
        lgbm_probabilities,
    )

    print_metrics(
        "LightGBM",
        lgbm_metrics,
        lgbm_time,
    )

    xgb_model_path = MODEL_DIR / "xgboost_baseline.pkl"

    lgbm_model_path = MODEL_DIR / "lightgbm_baseline.pkl"

    joblib.dump(
        xgb_model,
        xgb_model_path,
    )

    joblib.dump(
        lgbm_model,
        lgbm_model_path,
    )

    print()
    print(f"Saved XGBoost model: " f"{xgb_model_path}")

    print(f"Saved LightGBM model: " f"{lgbm_model_path}")

    save_validation_predictions(
        validation_data,
        label_column,
        xgb_probabilities,
        lgbm_probabilities,
    )

    save_feature_importance(
        "xgboost",
        xgb_model,
        feature_columns,
    )

    save_feature_importance(
        "lightgbm",
        lgbm_model,
        feature_columns,
    )

    result_rows: list[dict[str, Any]] = []

    logistic_result = load_logistic_baseline()

    if logistic_result is not None:
        result_rows.append(logistic_result)

    result_rows.append(
        {
            "model": "XGBoost",
            **xgb_metrics,
            "training_seconds": xgb_time,
        }
    )

    result_rows.append(
        {
            "model": "LightGBM",
            **lgbm_metrics,
            "training_seconds": lgbm_time,
        }
    )

    results = pd.DataFrame(result_rows)

    results = results.sort_values(
        "pr_auc",
        ascending=False,
    ).reset_index(drop=True)

    output_csv = OUTPUT_DIR / "traditional_model_comparison.csv"

    results.to_csv(
        output_csv,
        index=False,
    )

    create_model_comparison_chart(results)

    print()
    print("=" * 100)

    print("STAGE 3 TRADITIONAL MODEL COMPARISON")

    print("=" * 100)

    print(results.to_string(index=False))

    print()
    print(f"Saved comparison table: " f"{output_csv}")

    best_model = str(results.iloc[0]["model"])

    best_pr_auc = float(results.iloc[0]["pr_auc"])

    summary_path = OUTPUT_DIR / "traditional_model_summary.txt"

    summary = (
        "Stage 3 Traditional Model Comparison\n"
        "====================================\n\n"
        "Imbalance strategy: "
        "20:1 negative undersampling\n"
        "Primary model selection metric: PR-AUC\n\n"
        f"Best baseline model: {best_model}\n"
        f"Best validation PR-AUC: "
        f"{best_pr_auc:.6f}\n\n"
        "XGBoost and LightGBM are baseline tree "
        "models at this stage. Hyperparameters "
        "have not yet been optimized with Optuna.\n"
    )

    summary_path.write_text(
        summary,
        encoding="utf-8",
    )

    print(f"Saved summary: {summary_path}")

    print()
    print("=" * 80)
    print("STAGE 3 TREE MODEL TRAINING COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
