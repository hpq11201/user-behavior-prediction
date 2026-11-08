# mypy: ignore_missing_imports=True, warn_return_any=False
"""Train Stage 3 LSTM and GRU sequence baselines."""

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

OUTPUT_DIR = PROJECT_ROOT / "reports" / "stage3" / "sequence_models"

MODEL_DIR = PROJECT_ROOT / "models" / "stage3"

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


def select_feature_columns(
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
    """Build integer embedding mapping."""
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


def encode_id_column(
    dataframe: pd.DataFrame,
    column: str,
    mapping: dict[Any, int],
) -> np.ndarray:
    """Encode IDs for embedding inputs."""
    if column not in dataframe.columns or not mapping:
        return np.zeros(
            len(dataframe),
            dtype=np.int32,
        )

    encoded = dataframe[column].map(mapping).fillna(0).astype(np.int32).to_numpy()

    return encoded


def prepare_numeric_features(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    StandardScaler,
]:
    """Scale numeric features and convert them to feature sequences."""
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

    train_sequence = np.expand_dims(
        train_scaled,
        axis=-1,
    )

    validation_sequence = np.expand_dims(
        validation_scaled,
        axis=-1,
    )

    print()
    print("Feature sequence shape:")

    print(f"Train: {train_sequence.shape}")

    print("Validation: " f"{validation_sequence.shape}")

    return (
        train_sequence,
        validation_sequence,
        scaler,
    )


def prepare_model_inputs(
    dataframe: pd.DataFrame,
    feature_sequence: np.ndarray,
    user_mapping: dict[Any, int],
    item_mapping: dict[Any, int],
    category_mapping: dict[Any, int],
) -> dict[str, np.ndarray]:
    """Prepare Keras model inputs."""
    return {
        "feature_sequence": feature_sequence,
        "user_id": encode_id_column(
            dataframe,
            "user_id",
            user_mapping,
        ),
        "item_id": encode_id_column(
            dataframe,
            "item_id",
            item_mapping,
        ),
        "category_id": encode_id_column(
            dataframe,
            "category_id",
            category_mapping,
        ),
    }


def embedding_dimension(
    vocabulary_size: int,
) -> int:
    """Choose a small embedding dimension."""
    if vocabulary_size <= 10:
        return 4

    if vocabulary_size <= 100:
        return 8

    if vocabulary_size <= 10_000:
        return 16

    return 24


def build_sequence_model(
    model_type: str,
    sequence_length: int,
    user_vocab_size: int,
    item_vocab_size: int,
    category_vocab_size: int,
) -> Any:
    """Build LSTM or GRU baseline model."""
    sequence_input = keras.Input(
        shape=(
            sequence_length,
            1,
        ),
        name="feature_sequence",
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

    if model_type == "LSTM":
        sequence_vector = layers.LSTM(
            32,
            dropout=0.20,
            recurrent_dropout=0.0,
            name="lstm",
        )(sequence_input)

    elif model_type == "GRU":
        sequence_vector = layers.GRU(
            32,
            dropout=0.20,
            recurrent_dropout=0.0,
            name="gru",
        )(sequence_input)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    user_embedding = layers.Embedding(
        input_dim=max(
            user_vocab_size,
            1,
        ),
        output_dim=embedding_dimension(user_vocab_size),
        name="user_embedding",
    )(user_input)

    item_embedding = layers.Embedding(
        input_dim=max(
            item_vocab_size,
            1,
        ),
        output_dim=embedding_dimension(item_vocab_size),
        name="item_embedding",
    )(item_input)

    category_embedding = layers.Embedding(
        input_dim=max(
            category_vocab_size,
            1,
        ),
        output_dim=embedding_dimension(category_vocab_size),
        name="category_embedding",
    )(category_input)

    user_embedding = layers.Flatten()(user_embedding)

    item_embedding = layers.Flatten()(item_embedding)

    category_embedding = layers.Flatten()(category_embedding)

    combined = layers.Concatenate()(
        [
            sequence_vector,
            user_embedding,
            item_embedding,
            category_embedding,
        ]
    )

    combined = layers.Dense(
        64,
        activation="relu",
    )(combined)

    combined = layers.Dropout(0.30)(combined)

    combined = layers.Dense(
        32,
        activation="relu",
    )(combined)

    combined = layers.Dropout(0.20)(combined)

    output = layers.Dense(
        1,
        activation="sigmoid",
        name="purchase_probability",
    )(combined)

    model = keras.Model(
        inputs={
            "feature_sequence": sequence_input,
            "user_id": user_input,
            "item_id": item_input,
            "category_id": category_input,
        },
        outputs=output,
        name=f"stage3_{model_type.lower()}",
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


def train_model(
    model_type: str,
    train_inputs: dict[str, np.ndarray],
    train_labels: np.ndarray,
    validation_inputs: dict[str, np.ndarray],
    validation_labels: np.ndarray,
    sequence_length: int,
    user_vocab_size: int,
    item_vocab_size: int,
    category_vocab_size: int,
) -> tuple[
    Any,
    pd.DataFrame,
    np.ndarray,
    dict[str, float],
    float,
]:
    """Train and evaluate one sequence model."""
    print()
    print("=" * 80)

    print(f"TRAINING {model_type}")

    print("=" * 80)

    model = build_sequence_model(
        model_type,
        sequence_length,
        user_vocab_size,
        item_vocab_size,
        category_vocab_size,
    )

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
    print(f"{model_type} VALIDATION RESULTS")

    print(f"ROC-AUC:   " f"{metrics['roc_auc']:.6f}")

    print(f"PR-AUC:    " f"{metrics['pr_auc']:.6f}")

    print(f"Precision: " f"{metrics['precision_at_0_5']:.6f}")

    print(f"Recall:    " f"{metrics['recall_at_0_5']:.6f}")

    print(f"F1:        " f"{metrics['f1_at_0_5']:.6f}")

    print("Training seconds: " f"{training_seconds:.2f}")

    history_frame = pd.DataFrame(history.history)

    history_frame.insert(
        0,
        "epoch",
        np.arange(
            1,
            len(history_frame) + 1,
        ),
    )

    return (
        model,
        history_frame,
        probabilities,
        metrics,
        training_seconds,
    )


def save_training_history(
    history: pd.DataFrame,
    model_name: str,
) -> None:
    """Save history table and chart."""
    csv_path = OUTPUT_DIR / f"{model_name}_training_history.csv"

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

    plt.title(f"{model_name.upper()} Training History")

    plt.legend()

    plt.tight_layout()

    chart_path = OUTPUT_DIR / f"{model_name}_training_history.png"

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
    lstm_probabilities: np.ndarray,
    gru_probabilities: np.ndarray,
) -> None:
    """Save validation model probabilities."""
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

    output["lstm_probability"] = lstm_probabilities

    output["gru_probability"] = gru_probabilities

    output_path = OUTPUT_DIR / "sequence_validation_predictions.parquet"

    output.to_parquet(
        output_path,
        index=False,
    )

    print(f"Saved predictions: {output_path}")


def create_comparison_chart(
    comparison: pd.DataFrame,
) -> None:
    """Create LSTM vs GRU comparison chart."""
    metrics = [
        "roc_auc",
        "pr_auc",
        "precision_at_0_5",
        "recall_at_0_5",
        "f1_at_0_5",
    ]

    positions = np.arange(len(comparison))

    width = 0.15

    plt.figure(figsize=(13, 7))

    for index, metric in enumerate(metrics):
        plt.bar(
            positions + index * width,
            comparison[metric],
            width=width,
            label=metric,
        )

    center_offset = width * (len(metrics) - 1) / 2

    plt.xticks(
        positions + center_offset,
        comparison["model"].astype(str).tolist(),
    )

    plt.ylabel("Metric Value")

    plt.xlabel("Sequence Model")

    plt.title("Stage 3 LSTM vs GRU Comparison")

    plt.legend()

    plt.tight_layout()

    output_path = OUTPUT_DIR / "sequence_model_comparison.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved chart: {output_path}")


def main() -> None:
    """Train Stage 3 LSTM and GRU baselines."""
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

    feature_columns = select_feature_columns(
        train_data,
        label_column,
    )

    print(f"Target column: {label_column}")

    print("Numeric sequence features: " f"{len(feature_columns)}")

    for feature in feature_columns:
        print(f"  - {feature}")

    train_data = undersample_training_data(
        train_data,
        label_column,
    )

    print()
    print("Building embedding mappings...")

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

    print("Category vocabulary: " f"{len(category_mapping) + 1:,}")

    (
        train_sequence,
        validation_sequence,
        _,
    ) = prepare_numeric_features(
        train_data,
        validation_data,
        feature_columns,
    )

    train_inputs = prepare_model_inputs(
        train_data,
        train_sequence,
        user_mapping,
        item_mapping,
        category_mapping,
    )

    validation_inputs = prepare_model_inputs(
        validation_data,
        validation_sequence,
        user_mapping,
        item_mapping,
        category_mapping,
    )

    train_labels = train_data[label_column].astype(np.float32).to_numpy()

    validation_labels = validation_data[label_column].astype(np.float32).to_numpy()

    sequence_length = len(feature_columns)

    (
        lstm_model,
        lstm_history,
        lstm_probabilities,
        lstm_metrics,
        lstm_seconds,
    ) = train_model(
        "LSTM",
        train_inputs,
        train_labels,
        validation_inputs,
        validation_labels,
        sequence_length,
        len(user_mapping) + 1,
        len(item_mapping) + 1,
        len(category_mapping) + 1,
    )

    lstm_path = MODEL_DIR / "lstm_sequence_baseline.keras"

    lstm_model.save(lstm_path)

    print(f"Saved LSTM model: {lstm_path}")

    save_training_history(
        lstm_history,
        "lstm",
    )

    del lstm_model

    gc.collect()

    tf.keras.backend.clear_session()

    (
        gru_model,
        gru_history,
        gru_probabilities,
        gru_metrics,
        gru_seconds,
    ) = train_model(
        "GRU",
        train_inputs,
        train_labels,
        validation_inputs,
        validation_labels,
        sequence_length,
        len(user_mapping) + 1,
        len(item_mapping) + 1,
        len(category_mapping) + 1,
    )

    gru_path = MODEL_DIR / "gru_sequence_baseline.keras"

    gru_model.save(gru_path)

    print(f"Saved GRU model: {gru_path}")

    save_training_history(
        gru_history,
        "gru",
    )

    save_validation_predictions(
        validation_data,
        label_column,
        lstm_probabilities,
        gru_probabilities,
    )

    comparison = pd.DataFrame(
        [
            {
                "model": "LSTM",
                **lstm_metrics,
                "training_seconds": lstm_seconds,
            },
            {
                "model": "GRU",
                **gru_metrics,
                "training_seconds": gru_seconds,
            },
        ]
    )

    comparison = comparison.sort_values(
        "pr_auc",
        ascending=False,
    ).reset_index(drop=True)

    comparison_path = OUTPUT_DIR / "sequence_model_comparison.csv"

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    create_comparison_chart(comparison)

    best_model = str(comparison.iloc[0]["model"])

    best_pr_auc = float(comparison.iloc[0]["pr_auc"])

    summary_path = OUTPUT_DIR / "sequence_model_summary.txt"

    summary = (
        "Stage 3 Sequence Model Summary\n"
        "==============================\n\n"
        "Models: LSTM and GRU\n"
        "Imbalance strategy: "
        "20:1 negative undersampling\n"
        "Primary selection metric: "
        "validation PR-AUC\n\n"
        "Important methodological note:\n"
        "The current Stage 3 dataset contains "
        "aggregated user-item features rather than "
        "a true timestamp-ordered interaction sequence. "
        "Therefore these LSTM/GRU models should be "
        "interpreted as simplified sequence baselines. "
        "They do not represent a full behavioral "
        "sequence model.\n\n"
        f"Best sequence baseline: {best_model}\n"
        f"Best validation PR-AUC: "
        f"{best_pr_auc:.6f}\n"
    )

    summary_path.write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("STAGE 3 SEQUENCE MODEL COMPARISON")
    print("=" * 100)

    print(comparison.to_string(index=False))

    print()
    print(f"Saved comparison: {comparison_path}")

    print(f"Saved summary: {summary_path}")

    print()
    print("=" * 80)
    print("STAGE 3 SEQUENCE MODEL TRAINING COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
