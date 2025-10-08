"""Utilities for cleaning and loading the Porto taxi trajectory dataset into MySQL.

The script reads raw CSV files where each row represents a taxi trip with a nested
`POLYLINE` column that contains the sequence of GPS coordinates sampled every
15 seconds. The data is transformed into two relational tables:

- `trips`: Trip level metadata, derived metrics (distance, duration) and summary
  coordinates.
- `trip_points`: Individual GPS points with timestamps derived from the trip
  start time and the 15 second sampling cadence.

The code is intentionally chunked to support very large CSV files while keeping
memory usage bounded. Use the CLI to customise chunk size, limit the number of
rows processed, or enable a dry-run mode that skips database writes.
"""

from __future__ import annotations

import argparse
import ast
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from haversine import Unit, haversine

from DbConnector import DbConnector


# ---------------------------------------------------------------------------
# Configuration constants

POINT_INTERVAL_SECONDS = 15
DEFAULT_CHUNK_SIZE = 10_000


# ---------------------------------------------------------------------------
# Data containers


@dataclass(frozen=True)
class TripRecord:
	trip_id: int
	taxi_id: int
	call_type: Optional[str]
	client_id: Optional[int]
	stand_id: Optional[int]
	start_time: datetime
	end_time: Optional[datetime]
	day_type: Optional[str]
	missing_data: bool
	total_distance_m: Optional[float]
	duration_seconds: Optional[int]
	point_count: int
	is_valid: bool
	start_longitude: Optional[float]
	start_latitude: Optional[float]
	end_longitude: Optional[float]
	end_latitude: Optional[float]


@dataclass(frozen=True)
class TripPointRecord:
	trip_id: int
	point_seq: int
	point_time: datetime
	latitude: float
	longitude: float


# ---------------------------------------------------------------------------
# Logging setup


def configure_logging(verbosity: int) -> None:
	level = logging.WARNING
	if verbosity == 1:
		level = logging.INFO
	elif verbosity >= 2:
		level = logging.DEBUG

	logging.basicConfig(
		format="[%(asctime)s] %(levelname)s | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
		level=level,
	)


# ---------------------------------------------------------------------------
# Schema management


TRIPS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS trips (
	trip_id BIGINT PRIMARY KEY,
	taxi_id INT NOT NULL,
	call_type CHAR(1),
	client_id BIGINT NULL,
	stand_id INT NULL,
	start_time DATETIME NOT NULL,
	end_time DATETIME NULL,
	day_type CHAR(1),
	missing_data BOOLEAN NOT NULL,
	total_distance_m DOUBLE NULL,
	duration_seconds INT NULL,
	point_count INT NOT NULL,
	is_valid BOOLEAN NOT NULL,
	start_longitude DOUBLE NULL,
	start_latitude DOUBLE NULL,
	end_longitude DOUBLE NULL,
	end_latitude DOUBLE NULL,
	created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	INDEX idx_trips_taxi_start (taxi_id, start_time),
	INDEX idx_trips_call_type (call_type, day_type),
	INDEX idx_trips_start_end (start_time, end_time),
	INDEX idx_trips_start_coords (start_latitude, start_longitude),
	INDEX idx_trips_end_coords (end_latitude, end_longitude)
);
"""


TRIP_POINTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS trip_points (
	trip_id BIGINT NOT NULL,
	point_seq SMALLINT UNSIGNED NOT NULL,
	point_time DATETIME NOT NULL,
	latitude DOUBLE NOT NULL,
	longitude DOUBLE NOT NULL,
	PRIMARY KEY (trip_id, point_seq),
	INDEX idx_trip_points_time (point_time),
	INDEX idx_trip_points_lat_lon (latitude, longitude),
	CONSTRAINT fk_trip_points_trip FOREIGN KEY (trip_id)
		REFERENCES trips(trip_id)
		ON DELETE CASCADE
);
"""


def ensure_schema(cursor) -> None:
	"""Create the trips and trip_points tables if they don't already exist."""

	logging.info("Ensuring database schema is present")
	cursor.execute(TRIPS_TABLE_SQL)
	cursor.execute(TRIP_POINTS_TABLE_SQL)


# ---------------------------------------------------------------------------
# Transformation helpers


def _safe_int(value: str) -> Optional[int]:
	if value is None:
		return None
	value = str(value).strip()
	if value == "" or value.upper() == "NA":
		return None
	try:
		return int(value)
	except ValueError:
		return None


def parse_polyline(polyline_str: str) -> List[Tuple[float, float]]:
	"""Parse the POLYLINE column into a list of (longitude, latitude)."""

	if not isinstance(polyline_str, str):
		return []
	polyline_str = polyline_str.strip()
	if not polyline_str:
		return []

	try:
		raw_points = ast.literal_eval(polyline_str)
	except (ValueError, SyntaxError):
		logging.debug("Failed to parse POLYLINE: %s", polyline_str[:128])
		return []

	points: List[Tuple[float, float]] = []
	for entry in raw_points:
		if (
			isinstance(entry, (list, tuple))
			and len(entry) == 2
			and all(isinstance(v, (int, float)) for v in entry)
		):
			longitude, latitude = float(entry[0]), float(entry[1])
			points.append((longitude, latitude))
	return points


def compute_distance_and_duration(points: Sequence[Tuple[float, float]]) -> Tuple[Optional[float], Optional[int]]:
	"""Calculate total travelled distance (meters) and duration (seconds)."""

	point_count = len(points)
	if point_count < 2:
		return (0.0 if point_count == 1 else None, 0 if point_count == 1 else None)

	total_distance = 0.0
	prev = points[0]
	for current in points[1:]:
		# haversine expects (lat, lon)
		total_distance += haversine((prev[1], prev[0]), (current[1], current[0]), unit=Unit.METERS)
		prev = current

	duration_seconds = (point_count - 1) * POINT_INTERVAL_SECONDS
	return total_distance, duration_seconds


def make_trip_record(row) -> Tuple[TripRecord, List[TripPointRecord]]:
	points = parse_polyline(row.POLYLINE)
	point_count = len(points)

	start_timestamp = _safe_int(row.TIMESTAMP) or 0
	start_time = datetime.fromtimestamp(start_timestamp, tz=timezone.utc)

	total_distance_m, duration_seconds = compute_distance_and_duration(points)

	end_time: Optional[datetime] = None
	if point_count >= 1:
		end_time = start_time + timedelta(seconds=(point_count - 1) * POINT_INTERVAL_SECONDS)

	trip_record = TripRecord(
		trip_id=int(row.TRIP_ID),
		taxi_id=_safe_int(row.TAXI_ID) or 0,
		call_type=(row.CALL_TYPE or None),
		client_id=_safe_int(row.ORIGIN_CALL),
		stand_id=_safe_int(row.ORIGIN_STAND),
		start_time=start_time.replace(tzinfo=None),  # store as naive UTC
		end_time=end_time.replace(tzinfo=None) if end_time else None,
		day_type=(row.DAY_TYPE or None),
		missing_data=str(row.MISSING_DATA).strip().lower() == "true",
		total_distance_m=total_distance_m,
		duration_seconds=duration_seconds,
		point_count=point_count,
		is_valid=point_count >= 3,
		start_longitude=points[0][0] if point_count else None,
		start_latitude=points[0][1] if point_count else None,
		end_longitude=points[-1][0] if point_count else None,
		end_latitude=points[-1][1] if point_count else None,
	)

	point_records: List[TripPointRecord] = []
	if points:
		for seq, (longitude, latitude) in enumerate(points):
			point_time = start_time + timedelta(seconds=seq * POINT_INTERVAL_SECONDS)
			point_records.append(
				TripPointRecord(
					trip_id=trip_record.trip_id,
					point_seq=seq,
					point_time=point_time.replace(tzinfo=None),
					latitude=latitude,
					longitude=longitude,
				)
			)

	return trip_record, point_records


# ---------------------------------------------------------------------------
# Database writing helpers


def chunk_iterable(items: Sequence, size: int) -> Iterable[Sequence]:
	for index in range(0, len(items), size):
		yield items[index : index + size]


def upsert_trips(cursor, trip_records: Sequence[TripRecord]) -> None:
	sql = """
	INSERT INTO trips (
		trip_id, taxi_id, call_type, client_id, stand_id,
		start_time, end_time, day_type, missing_data,
		total_distance_m, duration_seconds, point_count, is_valid,
		start_longitude, start_latitude, end_longitude, end_latitude
	) VALUES (
		%(trip_id)s, %(taxi_id)s, %(call_type)s, %(client_id)s, %(stand_id)s,
		%(start_time)s, %(end_time)s, %(day_type)s, %(missing_data)s,
		%(total_distance_m)s, %(duration_seconds)s, %(point_count)s, %(is_valid)s,
		%(start_longitude)s, %(start_latitude)s, %(end_longitude)s, %(end_latitude)s
	)
	ON DUPLICATE KEY UPDATE
		taxi_id = VALUES(taxi_id),
		call_type = VALUES(call_type),
		client_id = VALUES(client_id),
		stand_id = VALUES(stand_id),
		start_time = VALUES(start_time),
		end_time = VALUES(end_time),
		day_type = VALUES(day_type),
		missing_data = VALUES(missing_data),
		total_distance_m = VALUES(total_distance_m),
		duration_seconds = VALUES(duration_seconds),
		point_count = VALUES(point_count),
		is_valid = VALUES(is_valid),
		start_longitude = VALUES(start_longitude),
		start_latitude = VALUES(start_latitude),
		end_longitude = VALUES(end_longitude),
		end_latitude = VALUES(end_latitude);
	"""

	data = [
		{
			"trip_id": trip.trip_id,
			"taxi_id": trip.taxi_id,
			"call_type": trip.call_type,
			"client_id": trip.client_id,
			"stand_id": trip.stand_id,
			"start_time": trip.start_time,
			"end_time": trip.end_time,
			"day_type": trip.day_type,
			"missing_data": trip.missing_data,
			"total_distance_m": trip.total_distance_m,
			"duration_seconds": trip.duration_seconds,
			"point_count": trip.point_count,
			"is_valid": trip.is_valid,
			"start_longitude": trip.start_longitude,
			"start_latitude": trip.start_latitude,
			"end_longitude": trip.end_longitude,
			"end_latitude": trip.end_latitude,
		}
		for trip in trip_records
	]

	cursor.executemany(sql, data)


def delete_existing_points(cursor, trip_ids: Sequence[int]) -> None:
	if not trip_ids:
		return

	delete_sql = "DELETE FROM trip_points WHERE trip_id IN ({})"
	for chunk in chunk_iterable(list(trip_ids), 500):
		placeholder = ",".join(["%s"] * len(chunk))
		sql = delete_sql.format(placeholder)
		cursor.execute(sql, tuple(chunk))


def insert_trip_points(cursor, point_records: Sequence[TripPointRecord]) -> None:
	if not point_records:
		return

	sql = """
	INSERT INTO trip_points (
		trip_id, point_seq, point_time, latitude, longitude
	) VALUES (
		%(trip_id)s, %(point_seq)s, %(point_time)s, %(latitude)s, %(longitude)s
	)
	ON DUPLICATE KEY UPDATE
		point_time = VALUES(point_time),
		latitude = VALUES(latitude),
		longitude = VALUES(longitude);
	"""

	data = [
		{
			"trip_id": point.trip_id,
			"point_seq": point.point_seq,
			"point_time": point.point_time,
			"latitude": point.latitude,
			"longitude": point.longitude,
		}
		for point in point_records
	]

	cursor.executemany(sql, data)


# ---------------------------------------------------------------------------
# Chunk processing


def process_dataframe(chunk: pd.DataFrame) -> Tuple[List[TripRecord], List[TripPointRecord]]:
	trip_records: List[TripRecord] = []
	point_records: List[TripPointRecord] = []

	for row in chunk.itertuples(index=False):
		try:
			trip, points = make_trip_record(row)
		except Exception as exc:  # noqa: BLE001
			logging.exception("Skipping row due to processing error: %s", exc)
			continue

		trip_records.append(trip)
		point_records.extend(points)

	return trip_records, point_records


def read_csv_in_chunks(
	csv_path: str,
	chunk_size: int,
	limit: Optional[int] = None,
) -> Iterable[pd.DataFrame]:
	row_count = 0
	chunk_iter = pd.read_csv(
		csv_path,
		chunksize=chunk_size,
		dtype="string",
		na_filter=False,
	)
	for chunk in chunk_iter:
		if limit is not None and row_count >= limit:
			break

		if limit is not None and row_count + len(chunk) > limit:
			chunk = chunk.iloc[: limit - row_count]

		yield chunk
		row_count += len(chunk)


# ---------------------------------------------------------------------------
# Orchestration


def load_porto_csv(
	csv_path: str,
	chunk_size: int = DEFAULT_CHUNK_SIZE,
	limit: Optional[int] = None,
	dry_run: bool = False,
) -> None:
	logging.info(
		"Starting ETL for %s (chunk_size=%s, limit=%s, dry_run=%s)",
		csv_path,
		chunk_size,
		limit,
		dry_run,
	)

	connector: Optional[DbConnector] = None
	cursor = None

	if not dry_run:
		connector = DbConnector()
		cursor = connector.cursor
		ensure_schema(cursor)
		connector.db_connection.commit()

	total_trips = 0
	total_points = 0

	for chunk_index, chunk in enumerate(read_csv_in_chunks(csv_path, chunk_size, limit), start=1):
		logging.info("Processing chunk %s containing %s rows", chunk_index, len(chunk))

		trip_records, point_records = process_dataframe(chunk)
		logging.debug(
			"Chunk %s produced %s trips and %s points",
			chunk_index,
			len(trip_records),
			len(point_records),
		)

		total_trips += len(trip_records)
		total_points += len(point_records)

		if dry_run:
			continue

		if not trip_records:
			continue

		trip_ids = [trip.trip_id for trip in trip_records]

		upsert_trips(cursor, trip_records)
		delete_existing_points(cursor, trip_ids)
		insert_trip_points(cursor, point_records)
		connector.db_connection.commit()

	logging.info("Finished processing: %s trips, %s points", total_trips, total_points)

	if connector is not None:
		connector.close_connection()


# ---------------------------------------------------------------------------
# CLI


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Clean and load Porto taxi data into MySQL")
	parser.add_argument("--csv", default="porto.csv", help="Path to porto.csv (default: porto.csv)")
	parser.add_argument(
		"--chunk-size",
		type=int,
		default=DEFAULT_CHUNK_SIZE,
		help="Number of rows to process per chunk (default: 10000)",
	)
	parser.add_argument(
		"--limit",
		type=int,
		default=None,
		help="Limit the number of rows processed (useful for testing)",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Process data but skip database writes",
	)
	parser.add_argument(
		"-v",
		"--verbose",
		action="count",
		default=2,
		help="Increase logging verbosity (-v, -vv)",
	)
	return parser


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()

	configure_logging(args.verbose)

	load_porto_csv(
		csv_path=args.csv,
		chunk_size=args.chunk_size,
		limit=args.limit,
		dry_run=args.dry_run,
	)


if __name__ == "__main__":
	main()
