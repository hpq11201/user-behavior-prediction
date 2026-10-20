"""Train, evaluate, and save a logistic regression purchase baseline."""

import pickle
import sqlite3
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = PROJECT_ROOT / "data" / "processed" / "user_behavior.db"

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "logistic_regression.pkl"

SCORE_PATH = PROJECT_ROOT / "data" / "processed" / "logistic_validation_scores.parquet"

TRAIN_TABLE = "training_modeling_dataset"
VALIDATION_TABLE = "modeling_dataset"

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

NEGATIVE_TO_POSITIVE_RATIO = 20
CHUNK_SIZE = 200_000
RANDOM_STATE = 42


def get_class_counts(
    connection: sqlite3.Connection,
    table_name: str,
) -> tuple[int, int]:
    """Return positive and negative sample counts."""
    cursor = connection.cursor()

    cursor.execute(f"""
        SELECT
            SUM(
                CASE
                    WHEN {TARGET_COLUMN} = 1
                    THEN 1
                    ELSE 0
                END
            ),
            SUM(
                CASE
                    WHEN {TARGET_COLUMN} = 0
                    THEN 1
                    ELSE 0
                END
            )
        FROM {table_name}
        """)

    positive_count, negative_count = cursor.fetchone()

    return int(positive_count), int(negative_count)


def load_positive_training_samples(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """Load all positive training samples."""
    columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    column_sql = ", ".join(columns)

    query = f"""
    SELECT
        {column_sql}
    FROM {TRAIN_TABLE}
    WHERE {TARGET_COLUMN} = 1
    """

    result = pd.read_sql_query(
        query,
        connection,
    )

    return cast(
        pd.DataFrame,
        result,
    )


def sample_negative_training_samples(
    connection: sqlite3.Connection,
    target_negative_count: int,
    total_negative_count: int,
) -> pd.DataFrame:
    """Sample negative training rows in chunks."""
    sampling_fraction = min(
        1.0,
        target_negative_count / total_negative_count,
    )

    print("Negative sampling fraction: " f"{sampling_fraction:.6f}")

    columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    column_sql = ", ".join(columns)

    query = f"""
    SELECT
        {column_sql}
    FROM {TRAIN_TABLE}
    WHERE {TARGET_COLUMN} = 0
    """

    sampled_chunks: list[pd.DataFrame] = []

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
                n=min(sample_size, len(chunk)),
                random_state=(RANDOM_STATE + chunk_number),
            )

            sampled_chunks.append(
                cast(
                    pd.DataFrame,
                    sampled,
                )
            )

        if chunk_number % 10 == 0:
            print("Processed negative chunk " f"{chunk_number}")

    negative_samples = pd.concat(
        sampled_chunks,
        ignore_index=True,
    )

    if len(negative_samples) > target_negative_count:
        negative_samples = negative_samples.sample(
            n=target_negative_count,
            random_state=RANDOM_STATE,
        )

    return cast(
        pd.DataFrame,
        negative_samples.reset_index(drop=True),
    )


def build_training_sample(
    connection: sqlite3.Connection,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build sampled training data."""
    (
        positive_count,
        negative_count,
    ) = get_class_counts(
        connection,
        TRAIN_TABLE,
    )

    target_negative_count = min(
        negative_count,
        positive_count * NEGATIVE_TO_POSITIVE_RATIO,
    )

    print()
    print("=" * 60)
    print("TRAINING SAMPLE")
    print("=" * 60)

    print(f"Available positive samples: " f"{positive_count:,}")

    print(f"Available negative samples: " f"{negative_count:,}")

    print(f"Target negative samples: " f"{target_negative_count:,}")

    print()
    print("Loading positive samples...")

    positive_samples = load_positive_training_samples(connection)

    print("Sampling negative samples...")

    negative_samples = sample_negative_training_samples(
        connection,
        target_negative_count,
        negative_count,
    )

    training_data = pd.concat(
        [
            positive_samples,
            negative_samples,
        ],
        ignore_index=True,
    )

    training_data = training_data.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    print()
    print(f"Final training rows: " f"{len(training_data):,}")

    print("Final training positives: " f"{int(training_data[TARGET_COLUMN].sum()):,}")

    print(
        "Final training negatives: "
        f"{int((training_data[TARGET_COLUMN] == 0).sum()):,}"
    )

    x_train = cast(
        pd.DataFrame,
        training_data[FEATURE_COLUMNS].astype(float),
    )

    y_train = cast(
        pd.Series,
        training_data[TARGET_COLUMN].astype(int),
    )

    return x_train, y_train


def build_model() -> Pipeline:
    """Create logistic regression pipeline."""
    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def save_model(
    model: Pipeline,
) -> None:
    """Save the trained model to disk."""
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with MODEL_PATH.open("wb") as file:
        pickle.dump(
            model,
            file,
        )

    print(f"Saved model to: " f"{MODEL_PATH}")


def evaluate_and_save_validation_scores(
    connection: sqlite3.Connection,
    model: Pipeline,
) -> None:
    """Evaluate model and save validation prediction scores."""
    columns = (
        [
            "user_id",
            "item_id",
        ]
        + FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )

    column_sql = ", ".join(columns)

    query = f"""
    SELECT
        {column_sql}
    FROM {VALIDATION_TABLE}
    """

    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    score_chunks: list[pd.DataFrame] = []

    print()
    print("Evaluating full validation dataset...")

    for chunk_number, chunk in enumerate(
        pd.read_sql_query(
            query,
            connection,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        x_chunk = cast(
            pd.DataFrame,
            chunk[FEATURE_COLUMNS].astype(float),
        )

        y_chunk = cast(
            pd.Series,
            chunk[TARGET_COLUMN].astype(int),
        )

        probability = model.predict_proba(x_chunk)[:, 1]

        prediction = (probability >= 0.5).astype(int)

        all_labels.append(y_chunk.to_numpy(dtype=np.int8))

        all_probabilities.append(probability.astype(np.float32))

        all_predictions.append(prediction.astype(np.int8))

        score_chunk = pd.DataFrame(
            {
                "user_id": chunk["user_id"].to_numpy(),
                "item_id": chunk["item_id"].to_numpy(),
                "label_purchase": y_chunk.to_numpy(dtype=np.int8),
                "prediction_probability": probability.astype(np.float32),
            }
        )

        score_chunks.append(score_chunk)

        if chunk_number % 10 == 0:
            print("Processed validation chunk " f"{chunk_number}")

    y_true = np.concatenate(all_labels)

    y_probability = np.concatenate(all_probabilities)

    y_prediction = np.concatenate(all_predictions)

    score_data = pd.concat(
        score_chunks,
        ignore_index=True,
    )

    print("Saving validation scores...")

    score_data.to_parquet(
        SCORE_PATH,
        index=False,
    )

    print(f"Saved validation scores to: " f"{SCORE_PATH}")

    roc_auc = roc_auc_score(
        y_true,
        y_probability,
    )

    pr_auc = average_precision_score(
        y_true,
        y_probability,
    )

    precision = precision_score(
        y_true,
        y_prediction,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_prediction,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_prediction,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_prediction,
    )

    true_negative = int(matrix[0, 0])
    false_positive = int(matrix[0, 1])
    false_negative = int(matrix[1, 0])
    true_positive = int(matrix[1, 1])

    positive_rate = 100.0 * y_true.sum() / len(y_true)

    print()
    print("=" * 60)
    print("LOGISTIC REGRESSION VALIDATION")
    print("=" * 60)

    print(f"Validation rows: " f"{len(y_true):,}")

    print(f"Validation positives: " f"{int(y_true.sum()):,}")

    print(f"Validation positive rate: " f"{positive_rate:.4f}%")

    print()

    print(f"ROC-AUC: " f"{roc_auc:.4f}")

    print(f"PR-AUC: " f"{pr_auc:.6f}")

    print(f"Precision @ 0.5: " f"{precision:.4f}")

    print(f"Recall @ 0.5: " f"{recall:.4f}")

    print(f"F1 @ 0.5: " f"{f1:.4f}")

    print()

    print("Confusion matrix:")

    print(f"True negatives: " f"{true_negative:,}")

    print(f"False positives: " f"{false_positive:,}")

    print(f"False negatives: " f"{false_negative:,}")

    print(f"True positives: " f"{true_positive:,}")

    print()

    baseline_pr_auc = float(y_true.mean())

    print("Random baseline PR-AUC: " f"{baseline_pr_auc:.6f}")

    if baseline_pr_auc > 0:
        lift = pr_auc / baseline_pr_auc

        print("PR-AUC lift over random: " f"{lift:.2f}x")


def print_model_coefficients(
    model: Pipeline,
) -> None:
    """Print standardized logistic regression coefficients."""
    logistic_model = model.named_steps["model"]

    coefficients = logistic_model.coef_[0]

    coefficient_table = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "coefficient": coefficients,
        }
    )

    coefficient_table["abs_coefficient"] = coefficient_table["coefficient"].abs()

    coefficient_table = coefficient_table.sort_values(
        by="abs_coefficient",
        ascending=False,
    )

    print()
    print("=" * 60)
    print("MODEL COEFFICIENTS")
    print("=" * 60)

    print(
        coefficient_table[
            [
                "feature",
                "coefficient",
            ]
        ].to_string(index=False)
    )


def main() -> None:
    """Train, evaluate, and save logistic regression baseline."""
    print("Opening SQLite database...")

    with sqlite3.connect(DATABASE_PATH) as connection:
        (
            x_train,
            y_train,
        ) = build_training_sample(connection)

        print()
        print("Training Logistic Regression...")

        model = build_model()

        model.fit(
            x_train,
            y_train,
        )

        print("Model training completed.")

        save_model(model)

        print_model_coefficients(model)

        evaluate_and_save_validation_scores(
            connection,
            model,
        )

    print()
    print("Logistic regression pipeline completed.")


if __name__ == "__main__":
    main()
