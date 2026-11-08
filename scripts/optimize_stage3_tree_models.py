# mypy: warn_return_any=False
"""Optimize Stage 3 LightGBM and XGBoost models using Optuna."""

import json
import time
from pathlib import Path
from typing import Any, cast

import joblib  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import optuna  # type: ignore[import-untyped]
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

BASELINE_RESULT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "stage3"
    / "tree_models"
    / "traditional_model_comparison.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "stage3" / "optuna"

MODEL_DIR = PROJECT_ROOT / "models" / "stage3"

RANDOM_STATE = 42

NEGATIVE_TO_POSITIVE_RATIO = 20

LIGHTGBM_TRIALS = 30

XGBOOST_TRIALS = 15

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
    """Select numeric modeling features."""
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
    """Prepare feature matrix and target."""
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
    print("Building 20:1 undersampled training data...")

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

    shuffled_index = sampled_x.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).index

    sampled_x = sampled_x.loc[shuffled_index].reset_index(drop=True)

    sampled_y = sampled_y.loc[shuffled_index].reset_index(drop=True)

    print(f"Training rows: {len(sampled_x):,}")

    print("Training positives: " f"{int(sampled_y.sum()):,}")

    print("Training negatives: " f"{int((sampled_y == 0).sum()):,}")

    print("Training positive rate: " f"{float(sampled_y.mean()):.6%}")

    return (
        sampled_x,
        sampled_y,
    )


def calculate_metrics(
    labels: pd.Series,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Calculate validation metrics."""
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


def build_lightgbm_objective(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> Any:
    """Create LightGBM Optuna objective."""

    def objective(
        trial: Any,
    ) -> float:
        params = {
            "n_estimators": 1500,
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.15,
                log=True,
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves",
                15,
                127,
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                12,
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples",
                10,
                100,
            ),
            "subsample": trial.suggest_float(
                "subsample",
                0.6,
                1.0,
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.6,
                1.0,
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                1e-8,
                10.0,
                log=True,
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                1e-8,
                10.0,
                log=True,
            ),
        }

        model = LGBMClassifier(
            **params,
            objective="binary",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )

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

        probability_matrix = np.asarray(model.predict_proba(x_validation))

        probabilities = probability_matrix[:, 1]

        score = average_precision_score(
            y_validation,
            probabilities,
        )

        return float(score)

    return objective


def build_xgboost_objective(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> Any:
    """Create XGBoost Optuna objective."""

    def objective(
        trial: Any,
    ) -> float:
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                300,
                900,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.15,
                log=True,
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                10,
            ),
            "min_child_weight": trial.suggest_float(
                "min_child_weight",
                1.0,
                20.0,
            ),
            "subsample": trial.suggest_float(
                "subsample",
                0.6,
                1.0,
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.6,
                1.0,
            ),
            "gamma": trial.suggest_float(
                "gamma",
                0.0,
                5.0,
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                1e-8,
                10.0,
                log=True,
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                1e-8,
                10.0,
                log=True,
            ),
        }

        model = XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

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

        probability_matrix = np.asarray(model.predict_proba(x_validation))

        probabilities = probability_matrix[:, 1]

        score = average_precision_score(
            y_validation,
            probabilities,
        )

        return float(score)

    return objective


def optimize_lightgbm(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> Any:
    """Run LightGBM Optuna optimization."""
    print()
    print("=" * 80)
    print("OPTUNA - LIGHTGBM")
    print("=" * 80)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="stage3_lightgbm",
    )

    study.optimize(
        build_lightgbm_objective(
            x_train,
            y_train,
            x_validation,
            y_validation,
        ),
        n_trials=LIGHTGBM_TRIALS,
        show_progress_bar=True,
    )

    print()
    print("Best LightGBM PR-AUC: " f"{study.best_value:.6f}")

    print("Best LightGBM parameters:")

    print(study.best_params)

    return study


def optimize_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> Any:
    """Run XGBoost Optuna optimization."""
    print()
    print("=" * 80)
    print("OPTUNA - XGBOOST")
    print("=" * 80)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="stage3_xgboost",
    )

    study.optimize(
        build_xgboost_objective(
            x_train,
            y_train,
            x_validation,
            y_validation,
        ),
        n_trials=XGBOOST_TRIALS,
        show_progress_bar=True,
    )

    print()
    print("Best XGBoost PR-AUC: " f"{study.best_value:.6f}")

    print("Best XGBoost parameters:")

    print(study.best_params)

    return study


def train_best_lightgbm(
    best_params: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[
    Any,
    np.ndarray,
    float,
]:
    """Train optimized LightGBM."""
    params = {
        **best_params,
        "n_estimators": 1500,
    }

    model = LGBMClassifier(
        **params,
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    start = time.perf_counter()

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

    training_seconds = time.perf_counter() - start

    probability_matrix = np.asarray(model.predict_proba(x_validation))

    probabilities = probability_matrix[:, 1]

    return (
        model,
        probabilities,
        training_seconds,
    )


def train_best_xgboost(
    best_params: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[
    Any,
    np.ndarray,
    float,
]:
    """Train optimized XGBoost."""
    model = XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    start = time.perf_counter()

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

    training_seconds = time.perf_counter() - start

    probability_matrix = np.asarray(model.predict_proba(x_validation))

    probabilities = probability_matrix[:, 1]

    return (
        model,
        probabilities,
        training_seconds,
    )


def save_study_outputs(
    study: Any,
    model_name: str,
) -> None:
    """Save Optuna parameters and trial history."""
    best_params_path = OUTPUT_DIR / f"{model_name}_best_params.json"

    with best_params_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            study.best_params,
            file,
            indent=2,
        )

    history = study.trials_dataframe()

    history_path = OUTPUT_DIR / f"{model_name}_optuna_history.csv"

    history.to_csv(
        history_path,
        index=False,
    )

    print(f"Saved: {best_params_path}")

    print(f"Saved: {history_path}")


def save_validation_predictions(
    validation_data: pd.DataFrame,
    label_column: str,
    lightgbm_probabilities: np.ndarray,
    xgboost_probabilities: np.ndarray,
) -> None:
    """Save optimized model predictions."""
    identifier_columns = [
        column
        for column in [
            "user_id",
            "item_id",
            "category_id",
        ]
        if column in validation_data.columns
    ]

    output = validation_data[identifier_columns].copy()

    output["label_purchase"] = validation_data[label_column].astype(int).to_numpy()

    output["lightgbm_tuned_probability"] = lightgbm_probabilities

    output["xgboost_tuned_probability"] = xgboost_probabilities

    output_path = OUTPUT_DIR / "tuned_validation_predictions.parquet"

    output.to_parquet(
        output_path,
        index=False,
    )

    print(f"Saved predictions: {output_path}")


def load_baseline_results() -> pd.DataFrame:
    """Load baseline comparison results."""
    if not BASELINE_RESULT_PATH.exists():
        return pd.DataFrame()

    data = pd.read_csv(BASELINE_RESULT_PATH)

    return cast(
        pd.DataFrame,
        data,
    )


def create_comparison_chart(
    dataframe: pd.DataFrame,
) -> None:
    """Create tuned vs baseline PR-AUC chart."""
    plt.figure(figsize=(10, 6))

    positions = np.arange(len(dataframe))

    plt.bar(
        positions,
        dataframe["pr_auc"],
    )

    plt.xticks(
        positions,
        dataframe["model"].astype(str).tolist(),
        rotation=20,
    )

    plt.ylabel("Validation PR-AUC")

    plt.xlabel("Model")

    plt.title("Baseline vs Optuna-Tuned Model Comparison")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "baseline_vs_tuned_comparison.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved chart: {output_path}")


def main() -> None:
    """Run Stage 3 Optuna optimization."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("Loading Stage 3 datasets...")

    train_data = pd.read_parquet(TRAIN_PATH)

    validation_data = pd.read_parquet(VALIDATION_PATH)

    label_column = find_label_column(train_data)

    feature_columns = select_feature_columns(
        train_data,
        label_column,
    )

    print(f"Target column: {label_column}")

    print("Number of features: " f"{len(feature_columns)}")

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

    lightgbm_study = optimize_lightgbm(
        x_train,
        y_train,
        x_validation,
        y_validation,
    )

    save_study_outputs(
        lightgbm_study,
        "lightgbm",
    )

    xgboost_study = optimize_xgboost(
        x_train,
        y_train,
        x_validation,
        y_validation,
    )

    save_study_outputs(
        xgboost_study,
        "xgboost",
    )

    (
        lightgbm_model,
        lightgbm_probabilities,
        lightgbm_seconds,
    ) = train_best_lightgbm(
        lightgbm_study.best_params,
        x_train,
        y_train,
        x_validation,
        y_validation,
    )

    (
        xgboost_model,
        xgboost_probabilities,
        xgboost_seconds,
    ) = train_best_xgboost(
        xgboost_study.best_params,
        x_train,
        y_train,
        x_validation,
        y_validation,
    )

    lightgbm_metrics = calculate_metrics(
        y_validation,
        lightgbm_probabilities,
    )

    xgboost_metrics = calculate_metrics(
        y_validation,
        xgboost_probabilities,
    )

    joblib.dump(
        lightgbm_model,
        MODEL_DIR / "lightgbm_optuna.pkl",
    )

    joblib.dump(
        xgboost_model,
        MODEL_DIR / "xgboost_optuna.pkl",
    )

    print()
    print("Saved optimized models.")

    save_validation_predictions(
        validation_data,
        label_column,
        lightgbm_probabilities,
        xgboost_probabilities,
    )

    baseline_results = load_baseline_results()

    rows: list[dict[str, Any]] = []

    if not baseline_results.empty:
        for _, row in baseline_results.iterrows():
            if str(row["model"]) in [
                "LightGBM",
                "XGBoost",
            ]:
                rows.append(
                    {
                        "model": (f"{row['model']} Baseline"),
                        "roc_auc": float(row["roc_auc"]),
                        "pr_auc": float(row["pr_auc"]),
                        "precision_at_0_5": float(row["precision_at_0_5"]),
                        "recall_at_0_5": float(row["recall_at_0_5"]),
                        "f1_at_0_5": float(row["f1_at_0_5"]),
                        "training_seconds": float(row["training_seconds"]),
                    }
                )

    rows.append(
        {
            "model": "LightGBM Optuna",
            **lightgbm_metrics,
            "training_seconds": (lightgbm_seconds),
        }
    )

    rows.append(
        {
            "model": "XGBoost Optuna",
            **xgboost_metrics,
            "training_seconds": (xgboost_seconds),
        }
    )

    comparison = pd.DataFrame(rows)

    comparison = comparison.sort_values(
        "pr_auc",
        ascending=False,
    ).reset_index(drop=True)

    comparison_path = OUTPUT_DIR / "baseline_vs_tuned_comparison.csv"

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    create_comparison_chart(comparison)

    print()
    print("=" * 110)
    print("STAGE 3 OPTUNA MODEL COMPARISON")
    print("=" * 110)

    print(comparison.to_string(index=False))

    best_model = str(comparison.iloc[0]["model"])

    best_pr_auc = float(comparison.iloc[0]["pr_auc"])

    summary_path = OUTPUT_DIR / "optuna_summary.txt"

    summary = (
        "Stage 3 Optuna Optimization Summary\n"
        "===================================\n\n"
        "Primary optimization metric: "
        "Validation PR-AUC\n"
        "Imbalance strategy: "
        "20:1 negative undersampling\n\n"
        f"LightGBM trials: {LIGHTGBM_TRIALS}\n"
        f"XGBoost trials: {XGBOOST_TRIALS}\n\n"
        f"Best overall model: {best_model}\n"
        f"Best validation PR-AUC: "
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
    print("STAGE 3 OPTUNA OPTIMIZATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
