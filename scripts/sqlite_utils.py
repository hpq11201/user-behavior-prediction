"""Reusable SQLite utility tools for the project."""

import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, cast

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "processed" / "user_behavior.db"


class SQLiteManager:
    """Reusable helper class for SQLite database operations."""

    def __init__(
        self,
        database_path: Path = DEFAULT_DATABASE_PATH,
    ) -> None:
        """Initialize the database manager.

        Args:
            database_path: Path to the SQLite database file.
        """
        self.database_path = database_path
        self.connection: Optional[sqlite3.Connection] = None

    def connect(self) -> sqlite3.Connection:
        """Open and return a SQLite connection."""
        if self.connection is None:
            self.connection = sqlite3.connect(self.database_path)

        return self.connection

    def close(self) -> None:
        """Close the current SQLite connection."""
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __enter__(self) -> "SQLiteManager":
        """Open connection when entering a context manager."""
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        """Close connection when leaving a context manager."""
        self.close()

    def execute(
        self,
        sql: str,
        parameters: Sequence[Any] = (),
        commit: bool = False,
    ) -> sqlite3.Cursor:
        """Execute one SQL statement.

        Args:
            sql: SQL statement.
            parameters: Optional SQL parameters.
            commit: Whether to commit after execution.

        Returns:
            SQLite cursor containing the execution result.
        """
        connection = self.connect()
        cursor = connection.cursor()

        cursor.execute(
            sql,
            parameters,
        )

        if commit:
            connection.commit()

        return cursor

    def execute_many(
        self,
        sql: str,
        parameters: Iterable[Sequence[Any]],
        commit: bool = True,
    ) -> None:
        """Execute one SQL statement for multiple parameter sets.

        Args:
            sql: SQL statement.
            parameters: Iterable of parameter sequences.
            commit: Whether to commit after execution.
        """
        connection = self.connect()
        cursor = connection.cursor()

        cursor.executemany(
            sql,
            parameters,
        )

        if commit:
            connection.commit()

    def query_one(
        self,
        sql: str,
        parameters: Sequence[Any] = (),
    ) -> Optional[tuple[Any, ...]]:
        """Return one row from a SQL query."""
        cursor = self.execute(
            sql,
            parameters,
        )

        row = cursor.fetchone()

        if row is None:
            return None

        return tuple(row)

    def query_all(
        self,
        sql: str,
        parameters: Sequence[Any] = (),
    ) -> list[tuple[Any, ...]]:
        """Return all rows from a SQL query."""
        cursor = self.execute(
            sql,
            parameters,
        )

        return [tuple(row) for row in cursor.fetchall()]

    def query_dataframe(
        self,
        sql: str,
        parameters: Sequence[Any] = (),
    ) -> pd.DataFrame:
        """Run SQL and return the result as a pandas DataFrame."""
        connection = self.connect()

        result = pd.read_sql_query(
            sql,
            connection,
            params=parameters,
        )

        return cast(
            pd.DataFrame,
            result,
        )

    def table_exists(
        self,
        table_name: str,
    ) -> bool:
        """Check whether a table exists."""
        row = self.query_one(
            """
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type = 'table'
              AND name = ?
            """,
            (table_name,),
        )

        if row is None:
            return False

        return bool(int(row[0]))

    def get_table_columns(
        self,
        table_name: str,
    ) -> list[str]:
        """Return column names for a table."""
        rows = self.query_all(f"""
            PRAGMA table_info({table_name})
            """)

        return [str(row[1]) for row in rows]

    def get_table_count(
        self,
        table_name: str,
    ) -> int:
        """Return total number of rows in a table."""
        row = self.query_one(f"""
            SELECT COUNT(*)
            FROM {table_name}
            """)

        if row is None:
            return 0

        return int(row[0])

    def get_database_tables(
        self,
    ) -> list[str]:
        """Return all user-created table names."""
        rows = self.query_all("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """)

        return [str(row[0]) for row in rows]

    def create_index(
        self,
        index_name: str,
        table_name: str,
        columns: Sequence[str],
        unique: bool = False,
    ) -> None:
        """Create an index if it does not already exist."""
        unique_sql = "UNIQUE " if unique else ""

        column_sql = ", ".join(columns)

        sql = f"""
        CREATE {unique_sql}INDEX IF NOT EXISTS
        {index_name}
        ON {table_name}({column_sql})
        """

        self.execute(
            sql,
            commit=True,
        )

    def drop_table(
        self,
        table_name: str,
    ) -> None:
        """Drop a table if it exists."""
        self.execute(
            f"""
            DROP TABLE IF EXISTS {table_name}
            """,
            commit=True,
        )


def main() -> None:
    """Run a simple validation of SQLiteManager."""
    print("Testing SQLiteManager...")

    with SQLiteManager() as database:
        tables = database.get_database_tables()

        print()
        print("=" * 60)
        print("SQLITE UTILITY VALIDATION")
        print("=" * 60)

        print(f"Database path: " f"{database.database_path}")

        print(f"Number of tables: " f"{len(tables)}")

        print()
        print("First 15 tables:")

        for table_name in tables[:15]:
            row_count = database.get_table_count(table_name)

            print(f"{table_name}: " f"{row_count:,} rows")

        print()

        user_behavior_columns = database.get_table_columns("user_behavior")

        print("user_behavior columns: " f"{user_behavior_columns}")

        user_behavior_count = database.get_table_count("user_behavior")

        print("user_behavior row count: " f"{user_behavior_count:,}")

    print()
    print("SQLiteManager validation completed.")


if __name__ == "__main__":
    main()
