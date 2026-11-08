# mypy: warn_return_any=False
"""Generate model evaluation outputs for Dashboard 2."""

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "dashboard2"

LOGISTIC_SCORE_PATH = DATA_DIR / "logistic_validation_scores.parquet"

HGB_SCORE_PATH = DATA_DIR / "hist_gradient_boosting_validation_scores.parquet"

RANDOM_STATE = 42

TOP_K_VALUES = [
    100,
    500,
    1000,
    5000,
    10000,
]

PER_USER_K_VALUES = [
    1,
    5,
    10,
]


def find_probability_column(
    dataframe: pd.DataFrame,
) -> str:
    """Find prediction probability column."""
    candidates = [
        "prediction_probability",
        "probability",
        "score",
        "predicted_probability",
        "y_score",
    ]

    for column in candidates:
        if column in dataframe.columns:
            return column

    raise ValueError(
        "Could not find prediction probability column. "
        f"Available columns: {list(dataframe.columns)}"
    )


def find_label_column(
    dataframe: pd.DataFrame,
) -> str:
    """Find binary label column."""
    candidates = [
        "label_purchase",
        "label",
        "target",
        "y_true",
    ]

    for column in candidates:
        if column in dataframe.columns:
            return column

    raise ValueError(
        "Could not find label column. " f"Available columns: {list(dataframe.columns)}"
    )


def load_scores(
    path: Path,
    model_name: str,
) -> pd.DataFrame:
    """Load validation prediction scores."""
    print(f"Loading {model_name} scores...")

    if not path.exists():
        raise FileNotFoundError(f"Missing score file: {path}")

    data = pd.read_parquet(path)

    print(f"{model_name} rows: " f"{len(data):,}")

    return cast(
        pd.DataFrame,
        data,
    )


def save_csv(
    dataframe: pd.DataFrame,
    filename: str,
) -> None:
    """Save table output."""
    output_path = OUTPUT_DIR / filename

    dataframe.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved: {output_path}")


def save_figure(
    filename: str,
) -> None:
    """Save current matplotlib figure."""
    output_path = OUTPUT_DIR / filename

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


def align_models(
    logistic: pd.DataFrame,
    hgb: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align model score files using shared identifiers."""
    join_columns = [
        column
        for column in [
            "user_id",
            "item_id",
        ]
        if (column in logistic.columns and column in hgb.columns)
    ]

    if len(join_columns) < 2:
        print("Warning: user_id/item_id are not both " "available in both score files.")

        minimum_rows = min(
            len(logistic),
            len(hgb),
        )

        return (
            logistic.iloc[:minimum_rows].reset_index(drop=True),
            hgb.iloc[:minimum_rows].reset_index(drop=True),
        )

    logistic_keyed = logistic.set_index(join_columns)

    hgb_keyed = hgb.set_index(join_columns)

    shared_index = logistic_keyed.index.intersection(hgb_keyed.index)

    logistic_aligned = logistic_keyed.loc[shared_index].reset_index()

    hgb_aligned = hgb_keyed.loc[shared_index].reset_index()

    print("Aligned validation rows: " f"{len(shared_index):,}")

    return (
        cast(
            pd.DataFrame,
            logistic_aligned,
        ),
        cast(
            pd.DataFrame,
            hgb_aligned,
        ),
    )


def calculate_curve_metrics(
    dataframe: pd.DataFrame,
) -> dict[str, np.ndarray | float]:
    """Calculate ROC and PR curve metrics."""
    label_column = find_label_column(dataframe)

    probability_column = find_probability_column(dataframe)

    labels = dataframe[label_column].astype(int).to_numpy()

    scores = dataframe[probability_column].astype(float).to_numpy()

    fpr, tpr, _ = roc_curve(
        labels,
        scores,
    )

    precision, recall, _ = precision_recall_curve(
        labels,
        scores,
    )

    roc_auc = auc(
        fpr,
        tpr,
    )

    pr_auc = auc(
        recall,
        precision,
    )

    return {
        "labels": labels,
        "scores": scores,
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }


def chart_roc_comparison(
    logistic: pd.DataFrame,
    hgb: pd.DataFrame,
) -> None:
    """Compare ROC curves."""
    print("Generating 01 ROC comparison...")

    logistic_metrics = calculate_curve_metrics(logistic)

    hgb_metrics = calculate_curve_metrics(hgb)

    plt.figure(figsize=(8, 7))

    plt.plot(
        logistic_metrics["fpr"],
        logistic_metrics["tpr"],
        label=("Logistic Regression " f"(AUC={logistic_metrics['roc_auc']:.4f})"),
    )

    plt.plot(
        hgb_metrics["fpr"],
        hgb_metrics["tpr"],
        label=("HistGradientBoosting " f"(AUC={hgb_metrics['roc_auc']:.4f})"),
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()

    save_figure("01_roc_curve_comparison.png")

    result = pd.DataFrame(
        {
            "model": [
                "Logistic Regression",
                "HistGradientBoosting",
            ],
            "roc_auc": [
                logistic_metrics["roc_auc"],
                hgb_metrics["roc_auc"],
            ],
        }
    )

    save_csv(
        result,
        "01_roc_curve_summary.csv",
    )


def chart_pr_comparison(
    logistic: pd.DataFrame,
    hgb: pd.DataFrame,
) -> None:
    """Compare Precision-Recall curves."""
    print("Generating 02 PR comparison...")

    logistic_metrics = calculate_curve_metrics(logistic)

    hgb_metrics = calculate_curve_metrics(hgb)

    logistic_labels = cast(
        np.ndarray,
        logistic_metrics["labels"],
    )

    baseline = float(logistic_labels.mean())

    plt.figure(figsize=(8, 7))

    plt.plot(
        logistic_metrics["recall"],
        logistic_metrics["precision"],
        label=("Logistic Regression " f"(AUC={logistic_metrics['pr_auc']:.4f})"),
    )

    plt.plot(
        hgb_metrics["recall"],
        hgb_metrics["precision"],
        label=("HistGradientBoosting " f"(AUC={hgb_metrics['pr_auc']:.4f})"),
    )

    plt.axhline(
        baseline,
        linestyle="--",
        label=("Random baseline " f"({baseline:.6f})"),
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison")
    plt.legend()

    save_figure("02_pr_curve_comparison.png")

    result = pd.DataFrame(
        {
            "model": [
                "Logistic Regression",
                "HistGradientBoosting",
            ],
            "pr_auc": [
                logistic_metrics["pr_auc"],
                hgb_metrics["pr_auc"],
            ],
            "random_baseline": [
                baseline,
                baseline,
            ],
        }
    )

    save_csv(
        result,
        "02_pr_curve_summary.csv",
    )


def threshold_analysis(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate multiple classification thresholds."""
    label_column = find_label_column(dataframe)

    probability_column = find_probability_column(dataframe)

    labels = dataframe[label_column].astype(int).to_numpy()

    scores = dataframe[probability_column].astype(float).to_numpy()

    thresholds = np.arange(
        0.05,
        0.96,
        0.05,
    )

    rows = []

    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)

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

        rows.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicted_positive_rate": float(predictions.mean()),
            }
        )

    return pd.DataFrame(rows)


def chart_threshold_sensitivity(
    hgb: pd.DataFrame,
) -> None:
    """Create threshold sensitivity chart."""
    print("Generating 03 threshold sensitivity...")

    result = threshold_analysis(hgb)

    plt.figure(figsize=(10, 6))

    plt.plot(
        result["threshold"],
        result["precision"],
        marker="o",
        label="Precision",
    )

    plt.plot(
        result["threshold"],
        result["recall"],
        marker="o",
        label="Recall",
    )

    plt.plot(
        result["threshold"],
        result["f1"],
        marker="o",
        label="F1",
    )

    plt.xlabel("Classification Threshold")

    plt.ylabel("Metric Value")

    plt.title("HistGradientBoosting Threshold Sensitivity")

    plt.legend()

    save_figure("03_threshold_sensitivity.png")

    save_csv(
        result,
        "03_threshold_sensitivity.csv",
    )


def chart_normalized_confusion_matrix(
    hgb: pd.DataFrame,
) -> None:
    """Create normalized confusion matrix."""
    print("Generating 04 normalized confusion matrix...")

    label_column = find_label_column(hgb)

    probability_column = find_probability_column(hgb)

    labels = hgb[label_column].astype(int).to_numpy()

    scores = hgb[probability_column].astype(float).to_numpy()

    predictions = (scores >= 0.5).astype(int)

    matrix = confusion_matrix(
        labels,
        predictions,
        normalize="true",
    )

    plt.figure(figsize=(7, 6))

    image = plt.imshow(
        matrix,
        vmin=0,
        vmax=1,
    )

    plt.colorbar(
        image,
        label="Normalized proportion",
    )

    labels_text = [
        "Negative",
        "Positive",
    ]

    plt.xticks(
        [0, 1],
        labels_text,
    )

    plt.yticks(
        [0, 1],
        labels_text,
    )

    plt.xlabel("Predicted Label")

    plt.ylabel("True Label")

    plt.title("Normalized Confusion Matrix " "(Threshold = 0.5)")

    for row in range(2):
        for column in range(2):
            plt.text(
                column,
                row,
                f"{matrix[row, column]:.3f}",
                ha="center",
                va="center",
            )

    save_figure("04_normalized_confusion_matrix.png")

    result = pd.DataFrame(
        matrix,
        index=[
            "true_negative",
            "true_positive",
        ],
        columns=[
            "pred_negative",
            "pred_positive",
        ],
    ).reset_index()

    save_csv(
        result,
        "04_normalized_confusion_matrix.csv",
    )


def chart_calibration_curve(
    logistic: pd.DataFrame,
    hgb: pd.DataFrame,
) -> None:
    """Create probability calibration comparison."""
    print("Generating 05 calibration curve...")

    plt.figure(figsize=(8, 7))

    for model_name, dataframe in [
        (
            "Logistic Regression",
            logistic,
        ),
        (
            "HistGradientBoosting",
            hgb,
        ),
    ]:
        label_column = find_label_column(dataframe)

        probability_column = find_probability_column(dataframe)

        sample_size = min(
            1_000_000,
            len(dataframe),
        )

        if len(dataframe) > sample_size:
            sample = dataframe.sample(
                n=sample_size,
                random_state=RANDOM_STATE,
            )
        else:
            sample = dataframe

        fraction_positive, mean_predicted = calibration_curve(
            sample[label_column].astype(int),
            sample[probability_column].astype(float),
            n_bins=10,
            strategy="quantile",
        )

        plt.plot(
            mean_predicted,
            fraction_positive,
            marker="o",
            label=model_name,
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    plt.xlabel("Mean Predicted Probability")

    plt.ylabel("Observed Positive Rate")

    plt.title("Probability Calibration Curve")

    plt.legend()

    save_figure("05_probability_calibration.png")


def calculate_global_topk(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate global Top-K ranking metrics."""
    label_column = find_label_column(dataframe)

    probability_column = find_probability_column(dataframe)

    ranked = dataframe.sort_values(
        probability_column,
        ascending=False,
    )

    total_positives = int(ranked[label_column].sum())

    baseline = float(ranked[label_column].mean())

    rows = []

    for k in TOP_K_VALUES:
        actual_k = min(
            k,
            len(ranked),
        )

        top = ranked.head(actual_k)

        hits = int(top[label_column].sum())

        precision = hits / actual_k if actual_k else 0.0

        recall = hits / total_positives if total_positives else 0.0

        lift = precision / baseline if baseline else 0.0

        rows.append(
            {
                "top_k": actual_k,
                "hits": hits,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "lift_at_k": lift,
            }
        )

    return pd.DataFrame(rows)


def chart_global_topk(
    logistic: pd.DataFrame,
    hgb: pd.DataFrame,
) -> None:
    """Create Global Top-K comparison."""
    print("Generating 06 global Top-K...")

    logistic_result = calculate_global_topk(logistic)

    logistic_result["model"] = "Logistic Regression"

    hgb_result = calculate_global_topk(hgb)

    hgb_result["model"] = "HistGradientBoosting"

    combined = pd.concat(
        [
            logistic_result,
            hgb_result,
        ],
        ignore_index=True,
    )

    plt.figure(figsize=(10, 6))

    for model_name, subset in combined.groupby("model"):
        plt.plot(
            subset["top_k"],
            subset["precision_at_k"],
            marker="o",
            label=str(model_name),
        )

    plt.xlabel("Top-K")
    plt.ylabel("Precision@K")

    plt.title("Global Top-K Precision Comparison")

    plt.xscale("log")

    plt.legend()

    save_figure("06a_global_topk_precision.png")

    plt.figure(figsize=(10, 6))

    for model_name, subset in combined.groupby("model"):
        plt.plot(
            subset["top_k"],
            subset["recall_at_k"],
            marker="o",
            label=str(model_name),
        )

    plt.xlabel("Top-K")
    plt.ylabel("Recall@K")

    plt.title("Global Top-K Recall Comparison")

    plt.xscale("log")

    plt.legend()

    save_figure("06b_global_topk_recall.png")

    save_csv(
        combined,
        "06_global_topk_summary.csv",
    )


def calculate_per_user_topk(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate per-user Top-K ranking metrics."""
    if "user_id" not in dataframe.columns:
        return pd.DataFrame()

    label_column = find_label_column(dataframe)

    probability_column = find_probability_column(dataframe)

    ranked = dataframe.sort_values(
        [
            "user_id",
            probability_column,
        ],
        ascending=[
            True,
            False,
        ],
    ).copy()

    ranked["rank_within_user"] = ranked.groupby("user_id").cumcount() + 1

    user_positive_counts = ranked.groupby("user_id")[label_column].sum()

    positive_users = user_positive_counts[user_positive_counts > 0]

    rows = []

    for k in PER_USER_K_VALUES:
        top = ranked[ranked["rank_within_user"] <= k]

        grouped = top.groupby("user_id").agg(
            recommended=(
                label_column,
                "count",
            ),
            hits=(
                label_column,
                "sum",
            ),
        )

        grouped = grouped.join(
            user_positive_counts.rename("total_positives"),
            how="left",
        )

        precision = float((grouped["hits"] / grouped["recommended"]).mean())

        eligible = grouped[grouped["total_positives"] > 0]

        recall = (
            float((eligible["hits"] / eligible["total_positives"]).mean())
            if not eligible.empty
            else 0.0
        )

        hit_users = int((eligible["hits"] > 0).sum())

        hit_rate = hit_users / len(positive_users) if len(positive_users) else 0.0

        rows.append(
            {
                "top_k": k,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "hit_users": hit_users,
                "hit_rate_at_k": hit_rate,
            }
        )

    return pd.DataFrame(rows)


def chart_per_user_topk(
    logistic: pd.DataFrame,
    hgb: pd.DataFrame,
) -> None:
    """Create per-user Top-K comparison."""
    print("Generating 07 per-user Top-K...")

    logistic_result = calculate_per_user_topk(logistic)

    hgb_result = calculate_per_user_topk(hgb)

    if logistic_result.empty or hgb_result.empty:
        print("Skipped per-user Top-K: " "user_id is missing.")
        return

    logistic_result["model"] = "Logistic Regression"

    hgb_result["model"] = "HistGradientBoosting"

    combined = pd.concat(
        [
            logistic_result,
            hgb_result,
        ],
        ignore_index=True,
    )

    plt.figure(figsize=(10, 6))

    for model_name, subset in combined.groupby("model"):
        plt.plot(
            subset["top_k"],
            subset["hit_rate_at_k"],
            marker="o",
            label=str(model_name),
        )

    plt.xlabel("Per-user Top-K")

    plt.ylabel("Hit Rate@K")

    plt.title("Per-user Top-K Hit Rate")

    plt.xticks(
        PER_USER_K_VALUES,
        [str(value) for value in PER_USER_K_VALUES],
    )

    plt.legend()

    save_figure("07_per_user_topk_hit_rate.png")

    save_csv(
        combined,
        "07_per_user_topk_summary.csv",
    )


def calculate_user_activity_strata(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate model by user candidate activity."""
    if "user_id" not in dataframe.columns:
        return pd.DataFrame()

    label_column = find_label_column(dataframe)

    probability_column = find_probability_column(dataframe)

    user_counts = dataframe.groupby("user_id").size().rename("candidate_count")

    quantiles = user_counts.quantile(
        [
            0.33,
            0.67,
        ]
    )

    low_threshold = float(quantiles.iloc[0])

    high_threshold = float(quantiles.iloc[1])

    mapping = user_counts.to_dict()

    working = dataframe[
        [
            "user_id",
            label_column,
            probability_column,
        ]
    ].copy()

    working["candidate_count"] = working["user_id"].map(mapping)

    working["activity_level"] = "Medium"

    working.loc[
        working["candidate_count"] <= low_threshold,
        "activity_level",
    ] = "Low"

    working.loc[
        working["candidate_count"] > high_threshold,
        "activity_level",
    ] = "High"

    rows = []

    for level in [
        "Low",
        "Medium",
        "High",
    ]:
        subset = working[working["activity_level"] == level]

        if subset.empty or subset[label_column].nunique() < 2:
            continue

        metrics = calculate_curve_metrics(subset)

        rows.append(
            {
                "segment": level,
                "rows": len(subset),
                "positive_rate": float(subset[label_column].mean()),
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
            }
        )

    return pd.DataFrame(rows)


def chart_user_stratified_performance(
    hgb: pd.DataFrame,
) -> None:
    """Evaluate performance by user activity group."""
    print("Generating 08 user-stratified performance...")

    result = calculate_user_activity_strata(hgb)

    if result.empty:
        print("Skipped user stratification: " "required identifiers unavailable.")
        return

    plt.figure(figsize=(9, 6))

    positions = np.arange(len(result))

    plt.bar(
        positions,
        result["pr_auc"],
    )

    plt.xticks(
        positions,
        result["segment"].astype(str).tolist(),
    )

    plt.ylabel("PR-AUC")
    plt.xlabel("User Activity Segment")

    plt.title("Model Performance by User Activity Level")

    save_figure("08_user_stratified_performance.png")

    save_csv(
        result,
        "08_user_stratified_performance.csv",
    )


def calculate_item_popularity_strata(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate model by item candidate frequency."""
    if "item_id" not in dataframe.columns:
        return pd.DataFrame()

    label_column = find_label_column(dataframe)

    probability_column = find_probability_column(dataframe)

    item_counts = dataframe.groupby("item_id").size().rename("candidate_count")

    quantiles = item_counts.quantile(
        [
            0.33,
            0.67,
        ]
    )

    low_threshold = float(quantiles.iloc[0])

    high_threshold = float(quantiles.iloc[1])

    mapping = item_counts.to_dict()

    working = dataframe[
        [
            "item_id",
            label_column,
            probability_column,
        ]
    ].copy()

    working["candidate_count"] = working["item_id"].map(mapping)

    working["popularity_level"] = "Medium"

    working.loc[
        working["candidate_count"] <= low_threshold,
        "popularity_level",
    ] = "Long-tail"

    working.loc[
        working["candidate_count"] > high_threshold,
        "popularity_level",
    ] = "Popular"

    rows = []

    for level in [
        "Long-tail",
        "Medium",
        "Popular",
    ]:
        subset = working[working["popularity_level"] == level]

        if subset.empty or subset[label_column].nunique() < 2:
            continue

        metrics = calculate_curve_metrics(subset)

        rows.append(
            {
                "segment": level,
                "rows": len(subset),
                "positive_rate": float(subset[label_column].mean()),
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
            }
        )

    return pd.DataFrame(rows)


def chart_item_stratified_performance(
    hgb: pd.DataFrame,
) -> None:
    """Evaluate performance by item popularity."""
    print("Generating 09 item-stratified performance...")

    result = calculate_item_popularity_strata(hgb)

    if result.empty:
        print("Skipped item stratification: " "item_id unavailable.")
        return

    plt.figure(figsize=(9, 6))

    positions = np.arange(len(result))

    plt.bar(
        positions,
        result["pr_auc"],
    )

    plt.xticks(
        positions,
        result["segment"].astype(str).tolist(),
    )

    plt.ylabel("PR-AUC")
    plt.xlabel("Item Popularity Segment")

    plt.title("Model Performance by Item Popularity")

    save_figure("09_item_stratified_performance.png")

    save_csv(
        result,
        "09_item_stratified_performance.csv",
    )


def cold_start_analysis(
    hgb: pd.DataFrame,
) -> None:
    """Evaluate historical and recalled candidate groups."""
    print("Generating 10 cold-start analysis...")

    required_columns = [
        "seen_historically",
        "from_category_recall",
    ]

    if not all(column in hgb.columns for column in required_columns):
        print(
            "Cold-start feature columns are not " "present in validation score parquet."
        )

        note_path = OUTPUT_DIR / "10_cold_start_limitation.txt"

        note_path.write_text(
            (
                "Cold-start evaluation could not be "
                "computed directly because the saved "
                "validation score file does not contain "
                "seen_historically or "
                "from_category_recall columns. "
                "Candidate-source coverage is evaluated "
                "separately in the candidate-set analysis."
            ),
            encoding="utf-8",
        )

        print(f"Saved limitation note: {note_path}")

        return

    label_column = find_label_column(hgb)

    rows = []

    group_definitions = [
        (
            "Historical",
            hgb["seen_historically"] == 1,
        ),
        (
            "Category recall",
            hgb["from_category_recall"] == 1,
        ),
    ]

    for name, mask in group_definitions:
        subset = hgb[mask]

        if subset.empty or subset[label_column].nunique() < 2:
            continue

        metrics = calculate_curve_metrics(subset)

        rows.append(
            {
                "candidate_type": name,
                "rows": len(subset),
                "positive_rate": float(subset[label_column].mean()),
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return

    save_csv(
        result,
        "10_cold_start_analysis.csv",
    )


def error_analysis(
    hgb: pd.DataFrame,
) -> None:
    """Create high-confidence prediction error tables."""
    print("Generating 11 error analysis...")

    label_column = find_label_column(hgb)

    probability_column = find_probability_column(hgb)

    false_positives = hgb[
        (hgb[label_column] == 0) & (hgb[probability_column] >= 0.9)
    ].copy()

    false_negatives = hgb[
        (hgb[label_column] == 1) & (hgb[probability_column] <= 0.1)
    ].copy()

    false_positives = false_positives.sort_values(
        probability_column,
        ascending=False,
    ).head(500)

    false_negatives = false_negatives.sort_values(
        probability_column,
        ascending=True,
    ).head(500)

    save_csv(
        false_positives,
        "11a_high_confidence_false_positives.csv",
    )

    save_csv(
        false_negatives,
        "11b_high_confidence_false_negatives.csv",
    )

    summary = pd.DataFrame(
        [
            {
                "error_type": ("High-confidence false positive"),
                "count": len(false_positives),
            },
            {
                "error_type": ("High-confidence false negative"),
                "count": len(false_negatives),
            },
        ]
    )

    save_csv(
        summary,
        "11_error_analysis_summary.csv",
    )


def write_stability_limitation() -> None:
    """Document time-stability limitation."""
    print("Generating 12 time-stability note...")

    note_path = OUTPUT_DIR / "12_time_stability_limitation.txt"

    note = """
Time-stability limitation

The current validation predictions correspond to one fixed future label window.
Therefore, a true comparison of model performance across multiple independent
time windows cannot be calculated from the current saved predictions.

The project already uses time-based feature/label splitting rather than a random
train-test split to reduce temporal leakage.

A stronger temporal stability evaluation would require rebuilding features and
labels for multiple rolling cut-off dates and retraining/evaluating the model
for each period.

This limitation is documented rather than estimating unsupported time-stability
results.
""".strip()

    note_path.write_text(
        note,
        encoding="utf-8",
    )

    print(f"Saved: {note_path}")


def model_summary(
    logistic: pd.DataFrame,
    hgb: pd.DataFrame,
) -> None:
    """Create overall model comparison summary."""
    print("Generating 13 model comparison summary...")

    logistic_metrics = calculate_curve_metrics(logistic)

    hgb_metrics = calculate_curve_metrics(hgb)

    logistic_label = find_label_column(logistic)

    hgb_label = find_label_column(hgb)

    result = pd.DataFrame(
        [
            {
                "model": ("Logistic Regression"),
                "rows": len(logistic),
                "positive_rate": float(logistic[logistic_label].mean()),
                "roc_auc": logistic_metrics["roc_auc"],
                "pr_auc": logistic_metrics["pr_auc"],
            },
            {
                "model": ("HistGradientBoosting"),
                "rows": len(hgb),
                "positive_rate": float(hgb[hgb_label].mean()),
                "roc_auc": hgb_metrics["roc_auc"],
                "pr_auc": hgb_metrics["pr_auc"],
            },
        ]
    )

    save_csv(
        result,
        "13_model_comparison_summary.csv",
    )

    print()
    print("MODEL COMPARISON")

    print(result.to_string(index=False))


def main() -> None:
    """Generate Dashboard 2 model evaluation outputs."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logistic = load_scores(
        LOGISTIC_SCORE_PATH,
        "Logistic Regression",
    )

    hgb = load_scores(
        HGB_SCORE_PATH,
        "HistGradientBoosting",
    )

    logistic, hgb = align_models(
        logistic,
        hgb,
    )

    print()
    print("=" * 80)
    print("DASHBOARD 2 MODEL EVALUATION")
    print("=" * 80)

    chart_roc_comparison(
        logistic,
        hgb,
    )

    chart_pr_comparison(
        logistic,
        hgb,
    )

    chart_threshold_sensitivity(hgb)

    chart_normalized_confusion_matrix(hgb)

    chart_calibration_curve(
        logistic,
        hgb,
    )

    chart_global_topk(
        logistic,
        hgb,
    )

    chart_per_user_topk(
        logistic,
        hgb,
    )

    chart_user_stratified_performance(hgb)

    chart_item_stratified_performance(hgb)

    cold_start_analysis(hgb)

    error_analysis(hgb)

    write_stability_limitation()

    model_summary(
        logistic,
        hgb,
    )

    print()
    print("=" * 80)
    print("DASHBOARD 2 EVALUATION COMPLETED")
    print("=" * 80)

    print(f"Outputs saved to: " f"{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
