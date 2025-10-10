"""Schema migration to align pre-existing Porto taxi database with the 2025-10 design.

Usage:
	python migrations/20251009_porto_schema_upgrade.py [--dry-run]

The script is idempotent. It checks for existing columns, indexes, and
constraints before applying changes so it can be rerun safely.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Iterable, Optional

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DbConnector import DbConnector


LOGGER = logging.getLogger(__name__)


CREATE_TABLE_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS taxis (
        taxi_id INT PRIMARY KEY,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS clients (
        client_id BIGINT PRIMARY KEY,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS stands (
        stand_id INT PRIMARY KEY,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS call_types (
        call_type CHAR(1) PRIMARY KEY,
        description VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS day_types (
        day_type CHAR(1) PRIMARY KEY,
        description VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
)

CALL_TYPE_LOOKUP = {
    "A": "Dispatched by the central",
    "B": "Hailed directly on the street",
    "C": "Picked up from a taxi stand",
}

DAY_TYPE_LOOKUP = {
    "A": "Workday",
    "B": "Weekend",
    "C": "Holiday or special day",
}

TRIP_COLUMNS = {
    "total_distance_m": "DOUBLE NULL",
    "duration_seconds": "INT NULL",
    "average_speed_kmh": "DOUBLE NULL",
    "point_count": "INT NOT NULL DEFAULT 0",
    "is_valid": "BOOLEAN NOT NULL DEFAULT TRUE",
    "is_outlier": "BOOLEAN NOT NULL DEFAULT FALSE",
    "start_longitude": "DOUBLE NULL",
    "start_latitude": "DOUBLE NULL",
    "end_longitude": "DOUBLE NULL",
    "end_latitude": "DOUBLE NULL",
    "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
}

TRIP_POINTS_COLUMNS = {
    "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
}

TRIP_INDEXES = {
    "idx_trips_taxi_start": "(taxi_id, start_time)",
    "idx_trips_call_type": "(call_type, day_type)",
    "idx_trips_start_end": "(start_time, end_time)",
    "idx_trips_start_coords": "(start_latitude, start_longitude)",
    "idx_trips_end_coords": "(end_latitude, end_longitude)",
    "idx_trips_valid": "(is_valid, is_outlier)",
    "idx_trips_client": "(client_id)",
    "idx_trips_stand": "(stand_id)",
}

TRIP_POINTS_INDEXES = {
    "idx_trip_points_time": "(point_time)",
    "idx_trip_points_lat_lon": "(latitude, longitude)",
}

TRIP_FOREIGN_KEYS = (
    ("fk_trips_taxi", "FOREIGN KEY (taxi_id) REFERENCES taxis(taxi_id)"),
    ("fk_trips_client", "FOREIGN KEY (client_id) REFERENCES clients(client_id)"),
    ("fk_trips_stand", "FOREIGN KEY (stand_id) REFERENCES stands(stand_id)"),
    ("fk_trips_call_type", "FOREIGN KEY (call_type) REFERENCES call_types(call_type)"),
    ("fk_trips_day_type", "FOREIGN KEY (day_type) REFERENCES day_types(day_type)"),
)

TRIP_POINTS_FOREIGN_KEYS = (
    ("fk_trip_points_trip", "FOREIGN KEY (trip_id) REFERENCES trips(trip_id) ON DELETE CASCADE"),
)


def execute(cursor, statement: str, params: Optional[Iterable] = None, dry_run: bool = False) -> None:
    preview = " ".join(statement.strip().split())[:120]
    LOGGER.debug("Statement preview: %s%s", preview, "..." if len(preview) == 120 else "")
    if dry_run:
        return
    cursor.execute(statement, params or ())


def column_exists(cursor, table: str, column: str) -> bool:
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = %s
          AND COLUMN_NAME = %s
        """,
        (table, column),
    )
    return cursor.fetchone()[0] > 0


def add_column(cursor, table: str, column: str, definition: str, dry_run: bool = False) -> None:
    if column_exists(cursor, table, column):
        LOGGER.debug("Column %s.%s already present", table, column)
        return
    LOGGER.info("Adding column %s.%s", table, column)
    execute(cursor, f"ALTER TABLE {table} ADD COLUMN {column} {definition};", dry_run=dry_run)


def index_exists(cursor, table: str, index: str) -> bool:
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM INFORMATION_SCHEMA.STATISTICS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = %s
          AND INDEX_NAME = %s
        """,
        (table, index),
    )
    return cursor.fetchone()[0] > 0


def add_index(cursor, table: str, index: str, definition: str, dry_run: bool = False) -> None:
    if index_exists(cursor, table, index):
        LOGGER.debug("Index %s on %s already present", index, table)
        return
    LOGGER.info("Creating index %s on %s", index, table)
    execute(cursor, f"ALTER TABLE {table} ADD INDEX {index} {definition};", dry_run=dry_run)


def constraint_exists(cursor, table: str, constraint: str) -> bool:
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = %s
          AND CONSTRAINT_NAME = %s
        """,
        (table, constraint),
    )
    return cursor.fetchone()[0] > 0


def add_constraint(cursor, table: str, name: str, definition: str, dry_run: bool = False) -> None:
    if constraint_exists(cursor, table, name):
        LOGGER.debug("Constraint %s on %s already present", name, table)
        return
    LOGGER.info("Adding constraint %s on %s", name, table)
    execute(cursor, f"ALTER TABLE {table} ADD CONSTRAINT {name} {definition};", dry_run=dry_run)


def seed_call_types(cursor, dry_run: bool = False) -> None:
    for code, description in CALL_TYPE_LOOKUP.items():
        LOGGER.debug("Ensuring call_type %s", code)
        if dry_run:
            continue
        cursor.execute(
            """
            INSERT INTO call_types (call_type, description)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE description = VALUES(description)
            """,
            (code, description),
        )


def seed_day_types(cursor, dry_run: bool = False) -> None:
    for code, description in DAY_TYPE_LOOKUP.items():
        LOGGER.debug("Ensuring day_type %s", code)
        if dry_run:
            continue
        cursor.execute(
            """
            INSERT INTO day_types (day_type, description)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE description = VALUES(description)
            """,
            (code, description),
        )


def normalise_trip_categories(cursor, dry_run: bool = False) -> None:
    statements = (
        "UPDATE trips SET call_type = NULL WHERE call_type IS NOT NULL AND TRIM(call_type) = '';",
        "UPDATE trips SET day_type = NULL WHERE day_type IS NOT NULL AND TRIM(day_type) = '';",
        "UPDATE trips SET call_type = UPPER(SUBSTRING(call_type, 1, 1)) WHERE call_type IS NOT NULL;",
        "UPDATE trips SET day_type = UPPER(SUBSTRING(day_type, 1, 1)) WHERE day_type IS NOT NULL;",
    )
    for statement in statements:
        execute(cursor, statement, dry_run=dry_run)


def backfill_dimensions(cursor, dry_run: bool = False) -> None:
    statements = (
        "INSERT IGNORE INTO taxis (taxi_id) SELECT DISTINCT taxi_id FROM trips WHERE taxi_id IS NOT NULL;",
        "INSERT IGNORE INTO clients (client_id) SELECT DISTINCT client_id FROM trips WHERE client_id IS NOT NULL;",
        "INSERT IGNORE INTO stands (stand_id) SELECT DISTINCT stand_id FROM trips WHERE stand_id IS NOT NULL;",
    )
    for statement in statements:
        execute(cursor, statement, dry_run=dry_run)


def recompute_metrics(cursor, dry_run: bool = False) -> None:
    statements = (
        """
        UPDATE trips
        SET average_speed_kmh = CASE
                WHEN total_distance_m IS NULL OR duration_seconds IS NULL OR duration_seconds <= 0 THEN NULL
                ELSE (total_distance_m / 1000.0) / (duration_seconds / 3600.0)
            END;
        """,
        """
        UPDATE trips
        SET is_outlier = (
                (total_distance_m IS NOT NULL AND total_distance_m > 200000)
                OR (duration_seconds IS NOT NULL AND duration_seconds > 21600)
                OR (average_speed_kmh IS NOT NULL AND average_speed_kmh > 160)
            );
        """,
    )
    for statement in statements:
        execute(cursor, statement, dry_run=dry_run)


def refresh_valid_trips_view(cursor, dry_run: bool = False) -> None:
    execute(cursor, "DROP VIEW IF EXISTS valid_trips;", dry_run=dry_run)
    execute(
        cursor,
        """
        CREATE VIEW valid_trips AS
        SELECT *
        FROM trips
        WHERE is_valid = TRUE AND is_outlier = FALSE;
        """,
        dry_run=dry_run,
    )


def ensure_trip_schema(cursor, dry_run: bool = False) -> None:
    for column, definition in TRIP_COLUMNS.items():
        add_column(cursor, "trips", column, definition, dry_run=dry_run)
    for column, definition in TRIP_POINTS_COLUMNS.items():
        add_column(cursor, "trip_points", column, definition, dry_run=dry_run)

    for name, definition in TRIP_INDEXES.items():
        add_index(cursor, "trips", name, definition, dry_run=dry_run)
    for name, definition in TRIP_POINTS_INDEXES.items():
        add_index(cursor, "trip_points", name, definition, dry_run=dry_run)

    # Foreign keys are applied later once dimension tables are populated.


def enforce_foreign_keys(cursor, dry_run: bool = False) -> None:
    for name, definition in TRIP_FOREIGN_KEYS:
        add_constraint(cursor, "trips", name, definition, dry_run=dry_run)
    for name, definition in TRIP_POINTS_FOREIGN_KEYS:
        add_constraint(cursor, "trip_points", name, definition, dry_run=dry_run)


def create_dimension_tables(cursor, dry_run: bool = False) -> None:
    for statement in CREATE_TABLE_STATEMENTS:
        execute(cursor, statement, dry_run=dry_run)


def configure_logging(level: int) -> None:
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s | %(message)s",
        level=level,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upgrade Porto taxi schema in-place")
    parser.add_argument("--dry-run", action="store_true", help="Log actions without executing SQL")
    parser.add_argument("--log-level", default="INFO", help="Python logging level (default: INFO)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(getattr(logging, args.log_level.upper(), logging.INFO))

    LOGGER.info("Starting schema upgrade (dry_run=%s)", args.dry_run)

    connector: Optional[DbConnector] = None
    try:
        connector = DbConnector()
        cursor = connector.cursor

        if cursor is None:
            raise RuntimeError("Database cursor could not be initialised")

        cursor.execute("SELECT DATABASE();")
        LOGGER.info("Connected to database: %s", cursor.fetchone()[0])

        create_dimension_tables(cursor, dry_run=args.dry_run)
        seed_call_types(cursor, dry_run=args.dry_run)
        seed_day_types(cursor, dry_run=args.dry_run)
        normalise_trip_categories(cursor, dry_run=args.dry_run)
        ensure_trip_schema(cursor, dry_run=args.dry_run)
        backfill_dimensions(cursor, dry_run=args.dry_run)
        enforce_foreign_keys(cursor, dry_run=args.dry_run)
        recompute_metrics(cursor, dry_run=args.dry_run)
        refresh_valid_trips_view(cursor, dry_run=args.dry_run)

        if connector.db_connection and not args.dry_run:
            connector.db_connection.commit()
            LOGGER.info("Schema upgrade committed")
        else:
            LOGGER.info("Dry run complete - no changes committed")

    finally:
        if connector is not None:
            connector.close_connection()


if __name__ == "__main__":
    main()
