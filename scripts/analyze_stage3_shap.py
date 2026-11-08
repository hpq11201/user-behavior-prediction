# mypy: ignore_missing_imports=True, warn_return_any=False
"""Run SHAP analysis for the Stage 3 final LightGBM model."""

from pathlib import Path
from typing import Any, cast

import joblib  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "stage3"

VALIDATION_PATH = DATA_DIR / "validation.parquet"

MODEL_PATH = PROJECT_ROOT / "models" / "stage3" / "lightgbm_optuna.pkl"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "stage3" / "shap"

RANDOM_STATE = 42

MAX_SHAP_ROWS = 5000

TOP_FEATURES = 15


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
    """Select model feature columns."""
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


def prepare_features(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Prepare model features."""
    features = dataframe[feature_columns].copy()

    features = features.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    ).fillna(0)

    return cast(
        pd.DataFrame,
        features,
    )


def sample_validation_rows(
    dataframe: pd.DataFrame,
    max_rows: int,
) -> pd.DataFrame:
    """Take manageable random sample for SHAP."""
    if len(dataframe) <= max_rows:
        return dataframe.reset_index(drop=True)

    sampled = dataframe.sample(
        n=max_rows,
        random_state=RANDOM_STATE,
    )

    return sampled.reset_index(drop=True)


def normalize_shap_values(
    raw_values: Any,
) -> np.ndarray:
    """Normalize SHAP output to a two-dimensional array."""
    values = np.asarray(raw_values)

    if values.ndim == 3:
        values = values[
            :,
            :,
            -1,
        ]

    if values.ndim != 2:
        raise ValueError("Unexpected SHAP value dimensions: " f"{values.shape}")

    return values


def build_global_importance(
    shap_values: np.ndarray,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Create mean absolute SHAP importance table."""
    mean_abs_shap = np.mean(
        np.abs(shap_values),
        axis=0,
    )

    result = pd.DataFrame(
        {
            "feature": feature_columns,
            "mean_abs_shap": mean_abs_shap,
        }
    )

    result = result.sort_values(
        "mean_abs_shap",
        ascending=False,
    ).reset_index(drop=True)

    return result


def create_global_importance_chart(
    importance: pd.DataFrame,
) -> None:
    """Create global SHAP feature importance plot."""
    plot_data = importance.head(TOP_FEATURES).sort_values("mean_abs_shap")

    plt.figure(figsize=(10, 7))

    plt.barh(
        plot_data["feature"],
        plot_data["mean_abs_shap"],
    )

    plt.xlabel("Mean absolute SHAP value")

    plt.ylabel("Feature")

    plt.title("LightGBM Global SHAP Feature Importance")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "01_global_shap_importance.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


def create_summary_plot(
    shap_values: np.ndarray,
    feature_data: pd.DataFrame,
) -> None:
    """Create SHAP summary beeswarm plot."""
    plt.figure()

    shap.summary_plot(
        shap_values,
        feature_data,
        show=False,
        max_display=TOP_FEATURES,
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / "02_shap_summary_plot.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


def create_direction_summary(
    shap_values: np.ndarray,
    feature_data: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate direction of each feature's SHAP effect."""
    rows: list[dict[str, Any]] = []

    for index, feature in enumerate(feature_data.columns):
        feature_values = feature_data[feature].astype(float).to_numpy()

        feature_shap = shap_values[
            :,
            index,
        ]

        if np.std(feature_values) == 0:
            correlation = 0.0

        else:
            correlation_matrix = np.corrcoef(
                feature_values,
                feature_shap,
            )

            correlation = float(
                correlation_matrix[
                    0,
                    1,
                ]
            )

            if np.isnan(correlation):
                correlation = 0.0

        if correlation > 0.05:
            direction = (
                "Higher values generally " "increase predicted purchase probability"
            )

        elif correlation < -0.05:
            direction = (
                "Higher values generally " "decrease predicted purchase probability"
            )

        else:
            direction = "Effect is weak or non-linear"

        rows.append(
            {
                "feature": feature,
                "shap_value_correlation": correlation,
                "direction": direction,
            }
        )

    result = pd.DataFrame(rows)

    return result


def create_local_explanations(
    model: Any,
    feature_data: pd.DataFrame,
    shap_values: np.ndarray,
    importance: pd.DataFrame,
) -> pd.DataFrame:
    """Create explanations for selected individual samples."""
    probability_matrix = np.asarray(model.predict_proba(feature_data))

    probabilities = probability_matrix[
        :,
        1,
    ]

    highest_index = int(np.argmax(probabilities))

    lowest_index = int(np.argmin(probabilities))

    median_order = np.argsort(probabilities)

    median_index = int(median_order[len(median_order) // 2])

    selected = {
        "highest_probability": highest_index,
        "median_probability": median_index,
        "lowest_probability": lowest_index,
    }

    top_features = importance["feature"].head(10).tolist()

    rows: list[dict[str, Any]] = []

    feature_lookup = {
        feature: index for index, feature in enumerate(feature_data.columns)
    }

    for sample_name, row_index in selected.items():
        for feature in top_features:
            feature_index = feature_lookup[feature]

            rows.append(
                {
                    "sample_type": sample_name,
                    "row_index": row_index,
                    "predicted_probability": float(probabilities[row_index]),
                    "feature": feature,
                    "feature_value": float(feature_data.iloc[row_index][feature]),
                    "shap_value": float(
                        shap_values[
                            row_index,
                            feature_index,
                        ]
                    ),
                }
            )

    result = pd.DataFrame(rows)

    return result


def create_dependence_plots(
    shap_values: np.ndarray,
    feature_data: pd.DataFrame,
    importance: pd.DataFrame,
) -> None:
    """Create dependence plots for top features."""
    top_features = importance["feature"].head(4).tolist()

    for number, feature in enumerate(
        top_features,
        start=1,
    ):
        shap.dependence_plot(
            feature,
            shap_values,
            feature_data,
            interaction_index=None,
            show=False,
        )

        plt.tight_layout()

        output_path = OUTPUT_DIR / (f"03_{number:02d}_" f"dependence_{feature}.png")

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()

        print(f"Saved: {output_path}")


def create_business_summary(
    importance: pd.DataFrame,
    directions: pd.DataFrame,
) -> str:
    """Create simple business interpretation summary."""
    merged = importance.merge(
        directions,
        on="feature",
        how="left",
    )

    top_rows = merged.head(10)

    lines = [
        "Stage 3 SHAP Interpretation Summary",
        "===================================",
        "",
        "Final model: Optuna-tuned LightGBM",
        "",
        "Top global predictive features:",
        "",
    ]

    for index, row in top_rows.iterrows():
        lines.append(
            f"{index + 1}. "
            f"{row['feature']} "
            f"(mean |SHAP|="
            f"{row['mean_abs_shap']:.6f})"
        )

        lines.append(f"   Direction: " f"{row['direction']}")

    lines.extend(
        [
            "",
            "Interpretation note:",
            (
                "SHAP values describe how model inputs "
                "affect the model prediction. They show "
                "predictive relationships rather than "
                "causal effects."
            ),
            "",
            "Business interpretation:",
            (
                "Features with larger mean absolute SHAP "
                "values have a stronger influence on the "
                "model's ranking of user-item purchase "
                "probability."
            ),
            (
                "Behavior intensity, recency, cart activity "
                "and related interaction features should be "
                "prioritized when designing targeting or "
                "recommendation rules, subject to the actual "
                "top-feature ranking generated by this run."
            ),
        ]
    )

    return "\n".join(lines)


def main() -> None:
    """Run Stage 3 SHAP analysis."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading validation dataset...")

    validation_data = pd.read_parquet(VALIDATION_PATH)

    label_column = find_label_column(validation_data)

    feature_columns = select_feature_columns(
        validation_data,
        label_column,
    )

    feature_data = prepare_features(
        validation_data,
        feature_columns,
    )

    feature_data = sample_validation_rows(
        feature_data,
        MAX_SHAP_ROWS,
    )

    print(f"SHAP sample rows: " f"{len(feature_data):,}")

    print(f"Number of features: " f"{len(feature_columns)}")

    print("Loading Optuna-tuned LightGBM model...")

    model = joblib.load(MODEL_PATH)

    print("Building SHAP TreeExplainer...")

    explainer = shap.TreeExplainer(model)

    print("Calculating SHAP values...")

    raw_shap_values = explainer.shap_values(feature_data)

    if isinstance(
        raw_shap_values,
        list,
    ):
        raw_shap_values = raw_shap_values[-1]

    shap_values = normalize_shap_values(raw_shap_values)

    print(f"SHAP matrix shape: " f"{shap_values.shape}")

    importance = build_global_importance(
        shap_values,
        feature_columns,
    )

    importance_path = OUTPUT_DIR / "global_shap_importance.csv"

    importance.to_csv(
        importance_path,
        index=False,
    )

    print(f"Saved: {importance_path}")

    create_global_importance_chart(importance)

    create_summary_plot(
        shap_values,
        feature_data,
    )

    directions = create_direction_summary(
        shap_values,
        feature_data,
    )

    direction_path = OUTPUT_DIR / "shap_feature_directions.csv"

    directions.to_csv(
        direction_path,
        index=False,
    )

    print(f"Saved: {direction_path}")

    create_dependence_plots(
        shap_values,
        feature_data,
        importance,
    )

    local_explanations = create_local_explanations(
        model,
        feature_data,
        shap_values,
        importance,
    )

    local_path = OUTPUT_DIR / "local_shap_explanations.csv"

    local_explanations.to_csv(
        local_path,
        index=False,
    )

    print(f"Saved: {local_path}")

    summary = create_business_summary(
        importance,
        directions,
    )

    summary_path = OUTPUT_DIR / "shap_interpretation_summary.txt"

    summary_path.write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print("=" * 90)

    print("TOP SHAP FEATURES")

    print("=" * 90)

    print(importance.head(TOP_FEATURES).to_string(index=False))

    print()
    print(f"Saved summary: {summary_path}")

    print()
    print("=" * 80)

    print("STAGE 3 SHAP ANALYSIS COMPLETED")

    print("=" * 80)


if __name__ == "__main__":
    main()
