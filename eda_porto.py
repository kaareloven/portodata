"""Exploratory data analysis for the Porto Taxi Trajectory dataset.

The script processes the large CSV file in manageable chunks, computes
summary statistics about the trips, and writes the results to the
`eda_reports/` directory. Plots are generated for common distributions
(point counts, trip durations, and trip distances) using a down-sampled
subset to keep memory requirements modest.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocess import (
    POINT_INTERVAL_SECONDS,
    compute_distance_and_duration,
    parse_polyline,
)

# ---------------------------------------------------------------------------
# Helpers


def configure_logging(verbosity: int) -> None:
    """Configure logging level based on verbosity flag."""

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


def parse_bool(value: object) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() == "true"


def normalize_str(value: object, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip() or default


def read_csv_chunks(csv_path: str, chunk_size: int, limit: Optional[int] = None) -> Iterable[pd.DataFrame]:
    """Yield chunks from the CSV file while respecting an optional row limit."""

    total = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, dtype="string", na_filter=False):
        if limit is not None and total >= limit:
            break
        if limit is not None and total + len(chunk) > limit:
            chunk = chunk.iloc[: limit - total]
        yield chunk
        total += len(chunk)


# ---------------------------------------------------------------------------
# EDA core


def run_eda(csv_path: str, chunk_size: int, limit: Optional[int], output_dir: Path) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    total_trips = 0
    total_points = 0
    missing_trips = 0
    invalid_trips = 0

    call_type_counter: Counter[str] = Counter()
    daytype_counter: Counter[str] = Counter()
    hour_counter: Counter[int] = Counter()

    unique_taxis: set[str] = set()
    unique_clients: set[str] = set()
    unique_stands: set[str] = set()

    point_counts: List[int] = []
    distances_m: List[float] = []
    durations_s: List[int] = []

    point_count_by_missing: defaultdict[bool, List[int]] = defaultdict(list)

    for chunk_index, chunk in enumerate(read_csv_chunks(csv_path, chunk_size, limit), start=1):
        logging.info("Processing chunk %s with %s rows", chunk_index, len(chunk))
        for row in chunk.itertuples(index=False):
            total_trips += 1

            call_type = normalize_str(row.CALL_TYPE, "Unknown")
            call_type_counter[call_type] += 1

            day_type = normalize_str(row.DAY_TYPE, "Unknown")
            daytype_counter[day_type] += 1

            missing = parse_bool(row.MISSING_DATA)
            if missing:
                missing_trips += 1

            taxi_id = normalize_str(row.TAXI_ID)
            if taxi_id:
                unique_taxis.add(taxi_id)

            origin_call = normalize_str(row.ORIGIN_CALL)
            if origin_call:
                unique_clients.add(origin_call)

            origin_stand = normalize_str(row.ORIGIN_STAND)
            if origin_stand:
                unique_stands.add(origin_stand)

            try:
                timestamp = int(str(row.TIMESTAMP))
            except ValueError:
                timestamp = 0
            start_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            hour_counter[start_time.hour] += 1

            points = parse_polyline(str(row.POLYLINE))
            point_count = len(points)
            total_points += point_count
            point_counts.append(point_count)
            point_count_by_missing[missing].append(point_count)

            if point_count < 3:
                invalid_trips += 1

            distance, duration = compute_distance_and_duration(points)
            if distance is not None and distance > 0:
                distances_m.append(distance)
            if duration is not None and duration > 0:
                durations_s.append(duration)

    summary = build_summary(
        csv_path=csv_path,
        total_trips=total_trips,
        total_points=total_points,
        missing_trips=missing_trips,
        invalid_trips=invalid_trips,
        call_type_counter=call_type_counter,
        daytype_counter=daytype_counter,
        hour_counter=hour_counter,
        unique_taxis=unique_taxis,
        unique_clients=unique_clients,
        unique_stands=unique_stands,
        point_counts=point_counts,
        distances_m=distances_m,
        durations_s=durations_s,
        point_count_by_missing=point_count_by_missing,
    )

    write_summary(summary, output_dir)
    generate_plots(point_counts, distances_m, durations_s, output_dir)

    return summary


def summarise_numeric(values: Sequence[float | int], precision: int = 2) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "p90": None}

    array = np.array(values, dtype=float)
    return {
        "count": int(array.size),
        "min": round(float(array.min()), precision),
        "max": round(float(array.max()), precision),
        "mean": round(float(array.mean()), precision),
        "median": round(float(np.median(array)), precision),
        "p90": round(float(np.percentile(array, 90)), precision),
    }


def build_summary(
    *,
    csv_path: str,
    total_trips: int,
    total_points: int,
    missing_trips: int,
    invalid_trips: int,
    call_type_counter: Counter[str],
    daytype_counter: Counter[str],
    hour_counter: Counter[int],
    unique_taxis: set[str],
    unique_clients: set[str],
    unique_stands: set[str],
    point_counts: List[int],
    distances_m: List[float],
    durations_s: List[int],
    point_count_by_missing: defaultdict[bool, List[int]],
) -> Dict:
    dataset_size_bytes = os.path.getsize(csv_path)

    point_summary = summarise_numeric(point_counts, precision=0)
    distance_summary = summarise_numeric([d / 1000.0 for d in distances_m], precision=3)
    duration_summary = summarise_numeric([d / 60.0 for d in durations_s], precision=2)

    average_points_per_trip = round(total_points / total_trips, 2) if total_trips else 0

    missing_share = round(missing_trips / total_trips * 100, 2) if total_trips else 0
    invalid_share = round(invalid_trips / total_trips * 100, 2) if total_trips else 0

    missing_vs_complete = {
        "missing_true": summarise_numeric(point_count_by_missing[True], precision=0),
        "missing_false": summarise_numeric(point_count_by_missing[False], precision=0),
    }

    summary = {
        "dataset": {
            "path": csv_path,
            "size_mb": round(dataset_size_bytes / (1024 ** 2), 2),
            "total_trips": total_trips,
            "total_points": total_points,
            "unique_taxis": len(unique_taxis),
            "unique_clients": len(unique_clients),
            "unique_stands": len(unique_stands),
            "average_points_per_trip": average_points_per_trip,
        },
        "quality": {
            "missing_trips": missing_trips,
            "missing_share_percent": missing_share,
            "invalid_trips": invalid_trips,
            "invalid_share_percent": invalid_share,
            "point_counts_missing_vs_complete": missing_vs_complete,
        },
    "call_type_distribution": dict(call_type_counter),
    "day_type_distribution": dict(daytype_counter),
    "start_hour_distribution": dict(hour_counter),
        "point_count_summary": point_summary,
        "distance_km_summary": distance_summary,
        "duration_minutes_summary": duration_summary,
    }
    return summary


def write_summary(summary: Dict, output_dir: Path) -> None:
    json_path = output_dir / "summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md_path = output_dir / "summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Porto Taxi Dataset – EDA Summary\n\n")
        dataset = summary["dataset"]
        f.write("## Dataset overview\n")
        f.write("- File: `{}`\n".format(dataset["path"]))
        f.write("- Size: {} MB\n".format(dataset["size_mb"]))
        f.write("- Trips: {}\n".format(dataset["total_trips"]))
        f.write("- GPS points: {}\n".format(dataset["total_points"]))
        f.write("- Unique taxis: {}\n".format(dataset["unique_taxis"]))
        f.write("- Unique clients: {}\n".format(dataset["unique_clients"]))
        f.write("- Unique stands: {}\n".format(dataset["unique_stands"]))
        f.write("- Avg points per trip: {}\n\n".format(dataset["average_points_per_trip"]))

        quality = summary["quality"]
        f.write("## Data quality\n")
        f.write("- Missing trips: {} ({}%)\n".format(quality["missing_trips"], quality["missing_share_percent"]))
        f.write("- Invalid trips (<3 points): {} ({}%)\n\n".format(quality["invalid_trips"], quality["invalid_share_percent"]))

        def write_counter(title: str, counter: Dict[str, int]) -> None:
            if not counter:
                return
            f.write(f"## {title}\n")
            total = sum(counter.values()) or 1
            for key, value in sorted(counter.items(), key=lambda item: item[1], reverse=True):
                share = round(value / total * 100, 2)
                f.write(f"- {key or 'Unknown'}: {value} ({share}%)\n")
            f.write("\n")

        write_counter("Call type distribution", summary["call_type_distribution"])
        write_counter("Day type distribution", summary["day_type_distribution"])

        hour_counter_dict = summary["start_hour_distribution"]
        if hour_counter_dict:
            f.write("## Start hour distribution\n")
            total = sum(hour_counter_dict.values()) or 1
            for hour in sorted(hour_counter_dict.keys()):
                count = hour_counter_dict[hour]
                share = round(count / total * 100, 2)
                f.write(f"- {hour:02d}:00–{hour:02d}:59 → {count} trips ({share}%)\n")
            f.write("\n")

        def write_stat_block(title: str, stats: Dict[str, Optional[float]]) -> None:
            f.write(f"## {title}\n")
            for key in ["count", "min", "max", "mean", "median", "p90"]:
                f.write(f"- {key}: {stats.get(key)}\n")
            f.write("\n")

        write_stat_block("Point count distribution", summary["point_count_summary"])
        write_stat_block("Distance (km) distribution", summary["distance_km_summary"])
        write_stat_block("Duration (minutes) distribution", summary["duration_minutes_summary"])


def generate_plots(point_counts: List[int], distances_m: List[float], durations_s: List[int], output_dir: Path) -> None:
    def maybe_sample(values: Sequence[float | int], max_samples: int = 200_000) -> np.ndarray:
        if not values:
            return np.array([])
        array = np.array(values)
        if array.size <= max_samples:
            return array
        idx = np.random.choice(array.size, size=max_samples, replace=False)
        return array[idx]

    plots = [
        (maybe_sample(point_counts), "Point count per trip", "Number of GPS points", "point_count_distribution_log.png"),
        (maybe_sample([d / 1000.0 for d in distances_m]), "Trip distance", "Distance (km)", "distance_distribution_log.png"),
        (maybe_sample([d / 60.0 for d in durations_s]), "Trip duration", "Duration (minutes)", "duration_distribution_log.png"),
    ]

    for data, title, xlabel, filename in plots:
        if data.size == 0:
            continue
        plt.figure(figsize=(8, 5))
        plt.hist(data, bins=50, color="#1976d2", edgecolor="white")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.yscale('log')
        plt.ylabel("Trips")
        plt.tight_layout()
        plt.savefig(output_dir / filename)
        plt.close()


# ---------------------------------------------------------------------------
# CLI


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EDA on the Porto Taxi dataset")
    parser.add_argument("--csv", default="porto.csv", help="Path to porto.csv")
    parser.add_argument("--chunk-size", type=int, default=100_000, help="Rows per chunk (default: 100000)")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quicker experiments")
    parser.add_argument(
        "--output-dir",
        default="eda_reports",
        help="Directory where summary files and plots are stored (default: eda_reports)",
    )
    parser.add_argument("-v", "--verbose", action="count", default=1, help="Increase logging verbosity")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    configure_logging(args.verbose)
    output_dir = Path(f"{args.output_dir}/y_log_scale")

    summary = run_eda(
        csv_path=args.csv,
        chunk_size=args.chunk_size,
        limit=args.limit,
        output_dir=output_dir,
    )

    logging.info("EDA complete: processed %s trips", summary["dataset"]["total_trips"])


if __name__ == "__main__":
    main()
