"""Build the SQLite database for user behavior analysis."""

import sqlite3
from pathlib import Path
from typing import cast

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "user_behavior_clean.parquet"

DATABASE_PATH = PROJECT_ROOT / "data" / "processed" / "user_behavior.db"

TABLE_NAME = "user_behavior"

INSERT_CHUNK_SIZE = 100_000


def load_clean_data(
    file_path: Path,
) -> pd.DataFrame:
    """Load the cleaned Parquet dataset.

    Args:
        file_path: Path to the cleaned Parquet file.

    Returns:
        Cleaned user behavior dataframe.

    Raises:
        FileNotFoundError: If the cleaned dataset does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Clean dataset not found: {file_path}")

    dataframe = cast(
        pd.DataFrame,
        pd.read_parquet(file_path),
    )

    return dataframe


def create_database_directory(
    database_path: Path,
) -> None:
    """Create the database parent directory if necessary.

    Args:
        database_path: Destination SQLite database path.
    """
    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def prepare_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare dataframe values for SQLite storage.

    Args:
        dataframe: Cleaned user behavior dataframe.

    Returns:
        Dataframe ready for SQLite insertion.
    """
    prepared = dataframe.copy()

    prepared["timestamp"] = prepared["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    prepared = cast(
        pd.DataFrame,
        prepared,
    )

    return prepared


def write_behavior_table(
    dataframe: pd.DataFrame,
    database_path: Path,
) -> None:
    """Write user behavior data into SQLite.

    Args:
        dataframe: Prepared user behavior dataframe.
        database_path: SQLite database path.
    """
    with sqlite3.connect(database_path) as connection:
        dataframe.to_sql(
            TABLE_NAME,
            connection,
            if_exists="replace",
            index=False,
            chunksize=INSERT_CHUNK_SIZE,
        )


def create_indexes(
    database_path: Path,
) -> None:
    """Create indexes used by common analytical queries.

    Args:
        database_path: SQLite database path.
    """
    index_statements = [
        """
        CREATE INDEX IF NOT EXISTS idx_behavior_user
        ON user_behavior(user_id)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_behavior_item
        ON user_behavior(item_id)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_behavior_time
        ON user_behavior(timestamp)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_behavior_type
        ON user_behavior(behavior_type)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_behavior_user_time
        ON user_behavior(user_id, timestamp)
        """,
    ]

    with sqlite3.connect(database_path) as connection:
        cursor = connection.cursor()

        for statement in index_statements:
            cursor.execute(statement)

        connection.commit()


def validate_database(
    database_path: Path,
) -> None:
    """Validate row count and table structure.

    Args:
        database_path: SQLite database path.
    """
    with sqlite3.connect(database_path) as connection:
        cursor = connection.cursor()

        cursor.execute("SELECT COUNT(*) FROM user_behavior")
        row_count = cursor.fetchone()[0]

        cursor.execute("PRAGMA table_info(user_behavior)")
        columns = cursor.fetchall()

    print()
    print("Database validation:")
    print(f"Rows in user_behavior: " f"{row_count:,}")

    print("Columns:")

    for column in columns:
        print(f"  - {column[1]} ({column[2]})")


def main() -> None:
    """Build the SQLite user behavior database."""
    print("Loading cleaned dataset...")

    dataframe = load_clean_data(CLEAN_DATA_PATH)

    print(f"Loaded {len(dataframe):,} records.")

    print("Preparing data for SQLite...")

    prepared_dataframe = prepare_dataframe(dataframe)

    create_database_directory(DATABASE_PATH)

    print("Writing user_behavior table...")

    write_behavior_table(
        dataframe=prepared_dataframe,
        database_path=DATABASE_PATH,
    )

    print("Creating indexes...")

    create_indexes(DATABASE_PATH)

    validate_database(DATABASE_PATH)

    print()
    print("SQLite database saved to: " f"{DATABASE_PATH}")


if __name__ == "__main__":
    main()
