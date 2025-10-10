"""Identify taxi pairs that came within a specified distance and time window.

The script streams `trip_points` from MySQL in chronological windows, groups
points into small spatial buckets, and only performs precise Haversine distance
checks for candidates that are both spatially and temporally close. Results are
persisted as soon as new pairs are discovered, making the run resumable and
providing continuous feedback through a progress bar.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Deque, Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DbConnector import DbConnector  # noqa: E402

EARTH_RADIUS_M = 6_371_000.0
PORTO_REFERENCE_LAT_DEG = 41.15
METERS_PER_DEG_LAT = 111_132.0
METERS_PER_DEG_LON = 111_320.0 * math.cos(math.radians(PORTO_REFERENCE_LAT_DEG))


@dataclass
class Encounter:
    taxi_a: int
    taxi_b: int
    first_met_at: datetime


@dataclass
class Point:
    idx: int
    taxi_id: int
    trip_id: int
    point_time: datetime
    epoch: float
    lat_rad: float
    lon_rad: float
    cell: Tuple[int, int]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find taxi pairs that met closely")
    parser.add_argument(
        "--radius-m",
        type=float,
        default=5.0,
        help="Maximum distance between taxis in metres (default: 5)",
    )
    parser.add_argument(
        "--time-window-s",
        type=int,
        default=5,
        help="Maximum time difference in seconds between points (default: 5)",
    )
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=5,
        help="Duration (minutes) of each processing window (default: 5)",
    )
    parser.add_argument(
        "--output",
        default="close_taxi_pairs.csv",
        help="CSV output file for unique taxi pairs (default: ./close_taxi_pairs.csv)",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="ISO timestamp; skip windows ending before this time (optional)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="How many windows between checkpoint writes (default: 10)",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s | %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def get_time_bounds(cursor) -> Tuple[datetime, datetime]:
    cursor.execute("SELECT MIN(point_time), MAX(point_time) FROM trip_points")
    min_time, max_time = cursor.fetchone()
    if min_time is None or max_time is None:
        raise RuntimeError("trip_points table is empty")
    return min_time, max_time


def iter_windows(
    start: datetime,
    end: datetime,
    window: timedelta,
    resume: Optional[datetime] = None,
) -> Iterator[Tuple[int, datetime, datetime]]:
    current = start
    index = 0
    while current < end:
        window_end = min(current + window, end)
        if resume is None or window_end > resume:
            yield index, current, window_end
        index += 1
        current = window_end


def fetch_points(
    connection,
    window_start: datetime,
    window_end: datetime,
) -> pd.DataFrame:
    sql = """
        SELECT tp.trip_id, tp.point_time, tp.latitude, tp.longitude, tr.taxi_id
        FROM trip_points tp
        JOIN trips tr ON tr.trip_id = tp.trip_id
        WHERE tp.point_time >= %s AND tp.point_time < %s
        ORDER BY tp.point_time ASC
    """
    return pd.read_sql(
        sql,
        con=connection,
        params=(window_start, window_end),
        parse_dates=["point_time"],
    )


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    sin_dlat = math.sin(dlat / 2.0)
    sin_dlon = math.sin(dlon / 2.0)
    a = sin_dlat * sin_dlat + math.cos(lat1) * math.cos(lat2) * sin_dlon * sin_dlon
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_M * c


def point_to_cell(latitude: float, longitude: float, cell_size_m: float) -> Tuple[int, int]:
    x_m = longitude * METERS_PER_DEG_LON
    y_m = latitude * METERS_PER_DEG_LAT
    return int(x_m // cell_size_m), int(y_m // cell_size_m)


def process_window(
    df: pd.DataFrame,
    radius_m: float,
    time_window_s: int,
    next_point_idx: int,
    pairs: Dict[Tuple[int, int], datetime],
) -> int:
    if df.empty:
        return next_point_idx

    df = df.sort_values("point_time")
    points: List[Point] = []
    cell_size_m = max(radius_m, 5.0)

    for row in df.itertuples(index=False):
        point_time: datetime = row.point_time.to_pydatetime() if hasattr(row.point_time, "to_pydatetime") else row.point_time
        lat_rad = math.radians(row.latitude)
        lon_rad = math.radians(row.longitude)
        cell = point_to_cell(row.latitude, row.longitude, cell_size_m)
        points.append(
            Point(
                idx=next_point_idx,
                taxi_id=int(row.taxi_id),
                trip_id=int(row.trip_id),
                point_time=point_time,
                epoch=point_time.timestamp(),
                lat_rad=lat_rad,
                lon_rad=lon_rad,
                cell=cell,
            )
        )
        next_point_idx += 1

    if not points:
        return next_point_idx

    active_points: Deque[Point] = deque()
    active_cells: Dict[Tuple[int, int], Deque[Point]] = defaultdict(deque)

    for point in points:
        cutoff = point.epoch - time_window_s
        while active_points and active_points[0].epoch < cutoff:
            old_point = active_points.popleft()
            cell_queue = active_cells.get(old_point.cell)
            if cell_queue:
                while cell_queue and cell_queue[0].idx != old_point.idx:
                    cell_queue.popleft()
                if cell_queue and cell_queue[0].idx == old_point.idx:
                    cell_queue.popleft()
                if not cell_queue:
                    active_cells.pop(old_point.cell, None)

        cell_x, cell_y = point.cell
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbour_cell = (cell_x + dx, cell_y + dy)
                for other in active_cells.get(neighbour_cell, ()):  # pragma: no branch
                    if other.taxi_id == point.taxi_id:
                        continue
                    if abs(point.epoch - other.epoch) > time_window_s:
                        continue
                    pair_key = (min(other.taxi_id, point.taxi_id), max(other.taxi_id, point.taxi_id))
                    if pair_key in pairs:
                        continue
                    distance = haversine_distance(point.lat_rad, point.lon_rad, other.lat_rad, other.lon_rad)
                    if distance <= radius_m:
                        first_met = min(point.point_time, other.point_time)
                        pairs[pair_key] = first_met

        active_points.append(point)
        active_cells[point.cell].append(point)

    return next_point_idx


def write_pairs_to_csv(pairs: Dict[Tuple[int, int], datetime], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["taxi_id_a", "taxi_id_b", "first_met_at"])
        for (taxi_a, taxi_b), met_time in sorted(pairs.items(), key=lambda item: (item[1], item[0][0], item[0][1])):
            writer.writerow([taxi_a, taxi_b, met_time.isoformat(sep=" ")])


def main() -> None:
    args = parse_arguments()
    configure_logging(args.log_level)

    resume_from: Optional[datetime] = None
    if args.resume_from:
        resume_from = datetime.fromisoformat(args.resume_from)

    connector: Optional[DbConnector] = None
    try:
        connector = DbConnector()
        cursor = connector.cursor
        if cursor is None or connector.db_connection is None:
            raise RuntimeError("Database connection not available")

        start_time, end_time = get_time_bounds(cursor)
        window_td = timedelta(minutes=args.window_minutes)
        total_windows = math.ceil((end_time - start_time) / window_td)
        logging.info(
            "Processing trip_points from %s to %s in %s-minute windows (%s total windows)",
            start_time, end_time, args.window_minutes, total_windows,
        )

        pairs: Dict[Tuple[int, int], datetime] = {}
        next_point_idx = 0
        progress = tqdm(total=total_windows, desc="Windows", unit="window")

        for index, window_start, window_end in iter_windows(start_time, end_time, window_td, resume_from):
            progress.update(1)
            if resume_from and window_end <= resume_from:
                logging.debug("Skipping window ending %s due to resume-from", window_end)
                continue

            df = fetch_points(connector.db_connection, window_start, window_end)
            before_count = len(pairs)
            next_point_idx = process_window(df, args.radius_m, args.time_window_s, next_point_idx, pairs)
            new_pairs = len(pairs) - before_count

            logging.info(
                "Window %s processed (%s rows, %s new pairs, total pairs %s)",
                index, len(df), new_pairs, len(pairs),
            )

            if len(pairs) and (index + 1) % args.checkpoint_every == 0:
                logging.info("Writing checkpoint to %s", args.output)
                write_pairs_to_csv(pairs, Path(args.output))

        progress.close()

        if pairs:
            logging.info("Writing final results to %s", args.output)
            write_pairs_to_csv(pairs, Path(args.output))
        else:
            logging.info("No taxi pairs found within the specified thresholds")

    finally:
        if connector is not None:
            connector.close_connection()


if __name__ == "__main__":
    main()
