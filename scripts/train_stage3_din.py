# mypy: ignore_missing_imports=True, warn_return_any=False
"""Train a simplified Stage 3 DIN-style baseline."""

import gc
import time
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
from tensorflow import keras  # type: ignore[import-untyped]
from tensorflow.keras import layers  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "stage3"

TRAIN_PATH = DATA_DIR / "train.parquet"
VALIDATION_PATH = DATA_DIR / "validation.parquet"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "stage3" / "din"

MODEL_DIR = PROJECT_ROOT / "models" / "stage3"

SEQUENCE_RESULT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "stage3"
    / "sequence_models"
    / "sequence_model_comparison.csv"
)

OPTUNA_RESULT_PATH = (
    PROJECT_ROOT / "reports" / "stage3" / "optuna" / "baseline_vs_tuned_comparison.csv"
)

RANDOM_STATE = 42
NEGATIVE_TO_POSITIVE_RATIO = 20

EPOCHS = 30
BATCH_SIZE = 2048
PREDICTION_BATCH_SIZE = 8192

THRESHOLD = 0.5


def set_random_seed() -> None:
    """Set reproducible random seeds."""
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)


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


def select_numeric_features(
    dataframe: pd.DataFrame,
    label_column: str,
) -> list[str]:
    """Select numeric behavioral features."""
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


def undersample_training_data(
    dataframe: pd.DataFrame,
    label_column: str,
) -> pd.DataFrame:
    """Apply 20:1 negative undersampling."""
    print()
    print("Building 20:1 undersampled training data...")

    positives = dataframe[dataframe[label_column] == 1]

    negatives = dataframe[dataframe[label_column] == 0]

    target_negatives = min(
        len(negatives),
        len(positives) * NEGATIVE_TO_POSITIVE_RATIO,
    )

    sampled_negatives = negatives.sample(
        n=target_negatives,
        random_state=RANDOM_STATE,
    )

    sampled = pd.concat(
        [
            positives,
            sampled_negatives,
        ],
        ignore_index=True,
    )

    sampled = sampled.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    print(f"Training rows: {len(sampled):,}")

    print("Training positives: " f"{int(sampled[label_column].sum()):,}")

    print("Training negatives: " f"{int((sampled[label_column] == 0).sum()):,}")

    print("Training positive rate: " f"{float(sampled[label_column].mean()):.6%}")

    return cast(
        pd.DataFrame,
        sampled,
    )


def build_id_mapping(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    column: str,
) -> dict[Any, int]:
    """Build embedding vocabulary."""
    if column not in train_data.columns or column not in validation_data.columns:
        return {}

    combined = pd.concat(
        [
            train_data[column],
            validation_data[column],
        ],
        ignore_index=True,
    )

    unique_values = combined.dropna().unique()

    return {value: index + 1 for index, value in enumerate(unique_values)}


def encode_id(
    dataframe: pd.DataFrame,
    column: str,
    mapping: dict[Any, int],
) -> np.ndarray:
    """Encode ID column for embedding."""
    if column not in dataframe.columns or not mapping:
        return np.zeros(
            len(dataframe),
            dtype=np.int32,
        )

    return dataframe[column].map(mapping).fillna(0).astype(np.int32).to_numpy()


def prepare_numeric_features(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    StandardScaler,
]:
    """Scale behavioral features."""
    train_numeric = (
        train_data[feature_columns]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(0)
        .astype(np.float32)
    )

    validation_numeric = (
        validation_data[feature_columns]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(0)
        .astype(np.float32)
    )

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(train_numeric).astype(np.float32)

    validation_scaled = scaler.transform(validation_numeric).astype(np.float32)

    return (
        train_scaled,
        validation_scaled,
        scaler,
    )


def prepare_inputs(
    dataframe: pd.DataFrame,
    numeric_features: np.ndarray,
    user_mapping: dict[Any, int],
    item_mapping: dict[Any, int],
    category_mapping: dict[Any, int],
) -> dict[str, np.ndarray]:
    """Prepare model inputs."""
    return {
        "numeric_features": numeric_features,
        "user_id": encode_id(
            dataframe,
            "user_id",
            user_mapping,
        ),
        "item_id": encode_id(
            dataframe,
            "item_id",
            item_mapping,
        ),
        "category_id": encode_id(
            dataframe,
            "category_id",
            category_mapping,
        ),
    }


def choose_embedding_dim(
    vocabulary_size: int,
) -> int:
    """Choose embedding dimension."""
    if vocabulary_size <= 10:
        return 4

    if vocabulary_size <= 100:
        return 8

    if vocabulary_size <= 10_000:
        return 16

    return 24


def build_din_model(
    numeric_feature_count: int,
    user_vocab_size: int,
    item_vocab_size: int,
    category_vocab_size: int,
) -> Any:
    """Build simplified DIN-style network."""
    numeric_input = keras.Input(
        shape=(numeric_feature_count,),
        name="numeric_features",
    )

    user_input = keras.Input(
        shape=(),
        dtype="int32",
        name="user_id",
    )

    item_input = keras.Input(
        shape=(),
        dtype="int32",
        name="item_id",
    )

    category_input = keras.Input(
        shape=(),
        dtype="int32",
        name="category_id",
    )

    embedding_dim = 24

    user_embedding = layers.Embedding(
        input_dim=max(
            user_vocab_size,
            1,
        ),
        output_dim=embedding_dim,
        name="user_embedding",
    )(user_input)

    item_embedding = layers.Embedding(
        input_dim=max(
            item_vocab_size,
            1,
        ),
        output_dim=embedding_dim,
        name="item_embedding",
    )(item_input)

    category_embedding = layers.Embedding(
        input_dim=max(
            category_vocab_size,
            1,
        ),
        output_dim=embedding_dim,
        name="category_embedding",
    )(category_input)

    user_vector = layers.Flatten(name="user_vector")(user_embedding)

    item_vector = layers.Flatten(name="item_vector")(item_embedding)

    category_vector = layers.Flatten(name="category_vector")(category_embedding)

    interest_vector = layers.Concatenate(name="interest_concat")(
        [
            user_vector,
            category_vector,
        ]
    )

    interest_projection = layers.Dense(
        embedding_dim,
        activation="relu",
        name="interest_projection",
    )(interest_vector)

    attention_features = layers.Concatenate(name="attention_features")(
        [
            interest_projection,
            item_vector,
            layers.Multiply()(
                [
                    interest_projection,
                    item_vector,
                ]
            ),
            layers.Subtract()(
                [
                    interest_projection,
                    item_vector,
                ]
            ),
        ]
    )

    attention_hidden = layers.Dense(
        64,
        activation="relu",
        name="attention_dense_1",
    )(attention_features)

    attention_hidden = layers.Dense(
        32,
        activation="relu",
        name="attention_dense_2",
    )(attention_hidden)

    attention_score = layers.Dense(
        1,
        activation="sigmoid",
        name="attention_score",
    )(attention_hidden)

    attended_interest = layers.Multiply(name="attended_interest")(
        [
            interest_projection,
            attention_score,
        ]
    )

    combined = layers.Concatenate(name="final_features")(
        [
            numeric_input,
            attended_interest,
            item_vector,
            category_vector,
        ]
    )

    combined = layers.Dense(
        128,
        activation="relu",
    )(combined)

    combined = layers.Dropout(0.35)(combined)

    combined = layers.Dense(
        64,
        activation="relu",
    )(combined)

    combined = layers.Dropout(0.25)(combined)

    combined = layers.Dense(
        32,
        activation="relu",
    )(combined)

    output = layers.Dense(
        1,
        activation="sigmoid",
        name="purchase_probability",
    )(combined)

    model = keras.Model(
        inputs={
            "numeric_features": numeric_input,
            "user_id": user_input,
            "item_id": item_input,
            "category_id": category_input,
        },
        outputs=output,
        name="simplified_din",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(
                curve="ROC",
                name="roc_auc",
            ),
            keras.metrics.AUC(
                curve="PR",
                name="pr_auc",
            ),
        ],
    )

    return model


def calculate_metrics(
    labels: np.ndarray,
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


def save_training_history(
    history: pd.DataFrame,
) -> None:
    """Save DIN training history."""
    csv_path = OUTPUT_DIR / "din_training_history.csv"

    history.to_csv(
        csv_path,
        index=False,
    )

    print(f"Saved: {csv_path}")

    if "pr_auc" not in history.columns or "val_pr_auc" not in history.columns:
        return

    plt.figure(figsize=(9, 6))

    plt.plot(
        history["epoch"],
        history["pr_auc"],
        label="Train PR-AUC",
    )

    plt.plot(
        history["epoch"],
        history["val_pr_auc"],
        label="Validation PR-AUC",
    )

    plt.xlabel("Epoch")

    plt.ylabel("PR-AUC")

    plt.title("Simplified DIN Training History")

    plt.legend()

    plt.tight_layout()

    chart_path = OUTPUT_DIR / "din_training_history.png"

    plt.savefig(
        chart_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {chart_path}")


def save_validation_predictions(
    validation_data: pd.DataFrame,
    label_column: str,
    probabilities: np.ndarray,
) -> None:
    """Save DIN validation predictions."""
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

    output["din_probability"] = probabilities

    output_path = OUTPUT_DIR / "din_validation_predictions.parquet"

    output.to_parquet(
        output_path,
        index=False,
    )

    print(f"Saved predictions: {output_path}")


def load_previous_results() -> pd.DataFrame:
    """Load previous best model results."""
    rows: list[dict[str, Any]] = []

    if SEQUENCE_RESULT_PATH.exists():
        sequence_results = pd.read_csv(SEQUENCE_RESULT_PATH)

        for _, row in sequence_results.iterrows():
            rows.append(
                {
                    "model": str(row["model"]),
                    "roc_auc": float(row["roc_auc"]),
                    "pr_auc": float(row["pr_auc"]),
                    "precision_at_0_5": float(row["precision_at_0_5"]),
                    "recall_at_0_5": float(row["recall_at_0_5"]),
                    "f1_at_0_5": float(row["f1_at_0_5"]),
                }
            )

    if OPTUNA_RESULT_PATH.exists():
        optuna_results = pd.read_csv(OPTUNA_RESULT_PATH)

        matching = optuna_results[
            optuna_results["model"].astype(str) == "LightGBM Optuna"
        ]

        if not matching.empty:
            row = matching.iloc[0]

            rows.append(
                {
                    "model": "LightGBM Optuna",
                    "roc_auc": float(row["roc_auc"]),
                    "pr_auc": float(row["pr_auc"]),
                    "precision_at_0_5": float(row["precision_at_0_5"]),
                    "recall_at_0_5": float(row["recall_at_0_5"]),
                    "f1_at_0_5": float(row["f1_at_0_5"]),
                }
            )

    return pd.DataFrame(rows)


def create_model_comparison(
    din_metrics: dict[str, float],
) -> pd.DataFrame:
    """Create cross-model comparison."""
    previous = load_previous_results()

    din_row = pd.DataFrame(
        [
            {
                "model": "Simplified DIN",
                "roc_auc": din_metrics["roc_auc"],
                "pr_auc": din_metrics["pr_auc"],
                "precision_at_0_5": din_metrics["precision_at_0_5"],
                "recall_at_0_5": din_metrics["recall_at_0_5"],
                "f1_at_0_5": din_metrics["f1_at_0_5"],
            }
        ]
    )

    comparison = pd.concat(
        [
            previous,
            din_row,
        ],
        ignore_index=True,
    )

    comparison = comparison.sort_values(
        "pr_auc",
        ascending=False,
    ).reset_index(drop=True)

    output_path = OUTPUT_DIR / "din_model_comparison.csv"

    comparison.to_csv(
        output_path,
        index=False,
    )

    plt.figure(figsize=(10, 6))

    positions = np.arange(len(comparison))

    plt.bar(
        positions,
        comparison["pr_auc"],
    )

    plt.xticks(
        positions,
        comparison["model"].astype(str).tolist(),
        rotation=20,
    )

    plt.ylabel("Validation PR-AUC")

    plt.title("Stage 3 Deep Learning and Best Tree Model Comparison")

    plt.tight_layout()

    chart_path = OUTPUT_DIR / "din_model_comparison.png"

    plt.savefig(
        chart_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved comparison: {output_path}")

    print(f"Saved chart: {chart_path}")

    return comparison


def main() -> None:
    """Train simplified DIN baseline."""
    set_random_seed()

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

    feature_columns = select_numeric_features(
        train_data,
        label_column,
    )

    print(f"Target column: {label_column}")

    print("Numeric features: " f"{len(feature_columns)}")

    train_data = undersample_training_data(
        train_data,
        label_column,
    )

    print()
    print("Building embedding vocabularies...")

    user_mapping = build_id_mapping(
        train_data,
        validation_data,
        "user_id",
    )

    item_mapping = build_id_mapping(
        train_data,
        validation_data,
        "item_id",
    )

    category_mapping = build_id_mapping(
        train_data,
        validation_data,
        "category_id",
    )

    print(f"User vocabulary: " f"{len(user_mapping) + 1:,}")

    print(f"Item vocabulary: " f"{len(item_mapping) + 1:,}")

    print(f"Category vocabulary: " f"{len(category_mapping) + 1:,}")

    (
        train_numeric,
        validation_numeric,
        _,
    ) = prepare_numeric_features(
        train_data,
        validation_data,
        feature_columns,
    )

    train_inputs = prepare_inputs(
        train_data,
        train_numeric,
        user_mapping,
        item_mapping,
        category_mapping,
    )

    validation_inputs = prepare_inputs(
        validation_data,
        validation_numeric,
        user_mapping,
        item_mapping,
        category_mapping,
    )

    train_labels = train_data[label_column].astype(np.float32).to_numpy()

    validation_labels = validation_data[label_column].astype(np.float32).to_numpy()

    model = build_din_model(
        len(feature_columns),
        len(user_mapping) + 1,
        len(item_mapping) + 1,
        len(category_mapping) + 1,
    )

    print()
    print("=" * 80)

    print("TRAINING SIMPLIFIED DIN")

    print("=" * 80)

    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_pr_auc",
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    start_time = time.perf_counter()

    history = model.fit(
        train_inputs,
        train_labels,
        validation_data=(
            validation_inputs,
            validation_labels,
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    training_seconds = time.perf_counter() - start_time

    probability_matrix = np.asarray(
        model.predict(
            validation_inputs,
            batch_size=PREDICTION_BATCH_SIZE,
            verbose=1,
        )
    )

    probabilities = probability_matrix.reshape(-1)

    metrics = calculate_metrics(
        validation_labels,
        probabilities,
    )

    print()
    print("=" * 80)

    print("SIMPLIFIED DIN VALIDATION RESULTS")

    print("=" * 80)

    print(f"ROC-AUC:   " f"{metrics['roc_auc']:.6f}")

    print(f"PR-AUC:    " f"{metrics['pr_auc']:.6f}")

    print(f"Precision: " f"{metrics['precision_at_0_5']:.6f}")

    print(f"Recall:    " f"{metrics['recall_at_0_5']:.6f}")

    print(f"F1:        " f"{metrics['f1_at_0_5']:.6f}")

    print("Training seconds: " f"{training_seconds:.2f}")

    model_path = MODEL_DIR / "simplified_din.keras"

    model.save(model_path)

    print(f"Saved DIN model: {model_path}")

    history_frame = pd.DataFrame(history.history)

    history_frame.insert(
        0,
        "epoch",
        np.arange(
            1,
            len(history_frame) + 1,
        ),
    )

    save_training_history(history_frame)

    save_validation_predictions(
        validation_data,
        label_column,
        probabilities,
    )

    comparison = create_model_comparison(metrics)

    print()
    print("=" * 100)

    print("STAGE 3 DIN MODEL COMPARISON")

    print("=" * 100)

    print(comparison.to_string(index=False))

    best_model = str(comparison.iloc[0]["model"])

    best_pr_auc = float(comparison.iloc[0]["pr_auc"])

    summary_path = OUTPUT_DIR / "din_summary.txt"

    summary = (
        "Stage 3 Simplified DIN Summary\n"
        "==============================\n\n"
        "Primary selection metric: validation PR-AUC\n"
        "Imbalance strategy: 20:1 negative undersampling\n\n"
        "Methodological limitation:\n"
        "The available Stage 3 modeling table contains aggregated "
        "user-item features rather than a true timestamp-ordered "
        "user behavior history. Therefore this implementation is "
        "a simplified DIN-style baseline using user, item and "
        "category embeddings plus an attention-style interest "
        "interaction. It should not be interpreted as a complete "
        "industrial DIN implementation.\n\n"
        f"DIN validation PR-AUC: "
        f"{metrics['pr_auc']:.6f}\n"
        f"Best model in comparison: "
        f"{best_model}\n"
        f"Best PR-AUC: "
        f"{best_pr_auc:.6f}\n"
    )

    summary_path.write_text(
        summary,
        encoding="utf-8",
    )

    print(f"Saved summary: {summary_path}")

    del model

    gc.collect()

    tf.keras.backend.clear_session()

    print()
    print("=" * 80)

    print("STAGE 3 DIN TRAINING COMPLETED")

    print("=" * 80)


if __name__ == "__main__":
    main()
