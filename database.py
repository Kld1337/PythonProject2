import sqlite3
from pathlib import Path

import pandas as pd


DATABASE_PATH = Path(__file__).with_name(
    "bioassay_data.db"
)


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    return connection


def initialize_database() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                project_name TEXT NOT NULL,
                assay_type TEXT NOT NULL,
                experiment_date TEXT NOT NULL,
                measurement_type TEXT NOT NULL,
                measurement_unit TEXT,
                source_file TEXT NOT NULL,
                sheet_name TEXT NOT NULL,
                operator TEXT,
                timepoint TEXT,
                notes TEXT,
                created_at TEXT NOT NULL
                    DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS measurements (
                measurement_id INTEGER
                    PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                well TEXT NOT NULL,
                plate_row TEXT NOT NULL,
                plate_column INTEGER NOT NULL,
                sample_id TEXT,
                raw_value REAL,
                sample_status TEXT,
                FOREIGN KEY (experiment_id)
                    REFERENCES experiments(experiment_id)
                    ON DELETE CASCADE,
                UNIQUE (experiment_id, well)
            )
            """
        )


def experiment_exists(
    experiment_id: str,
) -> bool:
    with get_connection() as connection:
        result = connection.execute(
            """
            SELECT 1
            FROM experiments
            WHERE experiment_id = ?
            LIMIT 1
            """,
            (experiment_id,),
        ).fetchone()

    return result is not None


def save_experiment(
    metadata: dict,
    combined_data: pd.DataFrame,
) -> None:
    """
    Zapisuje metadane eksperymentu oraz wszystkie
    pomiary płytki w jednej transakcji SQLite.
    """

    experiment_id = metadata["experiment_id"]

    if experiment_exists(experiment_id):
        raise ValueError(
            "Eksperyment o identyfikatorze "
            f"'{experiment_id}' już istnieje w bazie."
        )

    if len(combined_data) != 96:
        raise ValueError(
            "Oczekiwano 96 pomiarów do zapisania, "
            f"ale otrzymano {len(combined_data)}."
        )

    required_columns = {
        "well",
        "plate_row",
        "plate_column",
        "sample_id",
        "sample_status",
    }

    missing_columns = (
        required_columns
        - set(combined_data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Dane nie zawierają wymaganych kolumn: "
            + ", ".join(sorted(missing_columns))
            + "."
        )

    if "raw_value" in combined_data.columns:
        value_column = "raw_value"

    elif "raw_absorbance" in combined_data.columns:
        value_column = "raw_absorbance"

    else:
        raise ValueError(
            "Dane nie zawierają kolumny raw_value "
            "ani raw_absorbance."
        )

    experiment_values = (
        experiment_id,
        metadata["project_name"],
        metadata["assay_type"],
        metadata["experiment_date"],
        metadata["measurement_type"],
        metadata.get("measurement_unit", ""),
        metadata["source_file"],
        metadata["sheet_name"],
        metadata.get("operator", ""),
        metadata.get("timepoint", ""),
        metadata.get("notes", ""),
    )

    measurement_values = []

    for row in combined_data.to_dict(
        orient="records"
    ):
        raw_value = row.get(value_column)

        measurement_values.append(
            (
                experiment_id,
                row["well"],
                row["plate_row"],
                int(row["plate_column"]),
                row.get("sample_id", ""),
                (
                    None
                    if pd.isna(raw_value)
                    else float(raw_value)
                ),
                row.get(
                    "sample_status",
                    "Unassigned",
                ),
            )
        )

    if len(measurement_values) != 96:
        raise RuntimeError(
            "Podczas przygotowania danych do zapisu "
            "nie utworzono 96 rekordów pomiarowych."
        )

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO experiments (
                experiment_id,
                project_name,
                assay_type,
                experiment_date,
                measurement_type,
                measurement_unit,
                source_file,
                sheet_name,
                operator,
                timepoint,
                notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            experiment_values,
        )

        connection.executemany(
            """
            INSERT INTO measurements (
                experiment_id,
                well,
                plate_row,
                plate_column,
                sample_id,
                raw_value,
                sample_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            measurement_values,
        )

        saved_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM measurements
            WHERE experiment_id = ?
            """,
            (experiment_id,),
        ).fetchone()[0]

        if saved_count != 96:
            raise RuntimeError(
                "Nie udało się zapisać pełnej płytki. "
                f"W bazie znaleziono {saved_count} "
                "pomiarów zamiast 96."
            )


def load_experiments() -> pd.DataFrame:
    with get_connection() as connection:
        return pd.read_sql_query(
            """
            SELECT *
            FROM experiments
            ORDER BY experiment_date DESC, created_at DESC
            """,
            connection,
        )


def load_measurements(
    experiment_id: str,
) -> pd.DataFrame:
    with get_connection() as connection:
        return pd.read_sql_query(
            """
            SELECT *
            FROM measurements
            WHERE experiment_id = ?
            ORDER BY plate_row, plate_column
            """,
            connection,
            params=(experiment_id,),
        )