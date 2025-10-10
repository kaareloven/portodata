-- Migration script to align existing Porto taxi database with the 2025-10 schema refresh.
-- Execute this script after taking a full backup. It is idempotent: rerunning has no harmful effect.

/* ----------------------------------------------------------------------
   0. Safety first: ensure we are in the expected database
---------------------------------------------------------------------- */
SELECT CONCAT('Running migration against database: ', DATABASE()) AS info_message;

/* ----------------------------------------------------------------------
   1. Create dimension/lookup tables (no-op if they already exist)
---------------------------------------------------------------------- */
CREATE TABLE IF NOT EXISTS taxis (
    taxi_id INT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS clients (
    client_id BIGINT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS stands (
    stand_id INT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS call_types (
    call_type CHAR(1) PRIMARY KEY,
    description VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS day_types (
    day_type CHAR(1) PRIMARY KEY,
    description VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

/* ----------------------------------------------------------------------
   2. Seed lookup values for call/day types
---------------------------------------------------------------------- */
INSERT INTO call_types (call_type, description)
VALUES
    ('A', 'Dispatched by the central'),
    ('B', 'Hailed directly on the street'),
    ('C', 'Picked up from a taxi stand')
ON DUPLICATE KEY UPDATE description = VALUES(description);

INSERT INTO day_types (day_type, description)
VALUES
    ('A', 'Workday'),
    ('B', 'Weekend'),
    ('C', 'Holiday or special day')
ON DUPLICATE KEY UPDATE description = VALUES(description);

/* ----------------------------------------------------------------------
   3. Normalise categorical values in the trips table so FK constraints succeed
---------------------------------------------------------------------- */
UPDATE trips
SET call_type = NULL
WHERE call_type IS NOT NULL AND TRIM(call_type) = '';

UPDATE trips
SET day_type = NULL
WHERE day_type IS NOT NULL AND TRIM(day_type) = '';

UPDATE trips
SET call_type = UPPER(SUBSTRING(call_type, 1, 1))
WHERE call_type IS NOT NULL;

UPDATE trips
SET day_type = UPPER(SUBSTRING(day_type, 1, 1))
WHERE day_type IS NOT NULL;

/* ----------------------------------------------------------------------
   4. Add new columns (no-op when they already exist)
---------------------------------------------------------------------- */
ALTER TABLE trips
    ADD COLUMN IF NOT EXISTS total_distance_m DOUBLE NULL AFTER missing_data,
    ADD COLUMN IF NOT EXISTS duration_seconds INT NULL AFTER total_distance_m,
    ADD COLUMN IF NOT EXISTS average_speed_kmh DOUBLE NULL AFTER duration_seconds,
    ADD COLUMN IF NOT EXISTS point_count INT NOT NULL DEFAULT 0 AFTER average_speed_kmh,
    ADD COLUMN IF NOT EXISTS is_valid BOOLEAN NOT NULL DEFAULT TRUE AFTER point_count,
    ADD COLUMN IF NOT EXISTS is_outlier BOOLEAN NOT NULL DEFAULT FALSE AFTER is_valid,
    ADD COLUMN IF NOT EXISTS start_longitude DOUBLE NULL AFTER is_outlier,
    ADD COLUMN IF NOT EXISTS start_latitude DOUBLE NULL AFTER start_longitude,
    ADD COLUMN IF NOT EXISTS end_longitude DOUBLE NULL AFTER start_latitude,
    ADD COLUMN IF NOT EXISTS end_latitude DOUBLE NULL AFTER end_longitude,
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP AFTER end_latitude;

ALTER TABLE trip_points
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP AFTER longitude;

/* ----------------------------------------------------------------------
   5. Backfill dimension tables from existing trip facts
---------------------------------------------------------------------- */
INSERT IGNORE INTO taxis (taxi_id)
SELECT DISTINCT taxi_id
FROM trips
WHERE taxi_id IS NOT NULL;

INSERT IGNORE INTO clients (client_id)
SELECT DISTINCT client_id
FROM trips
WHERE client_id IS NOT NULL;

INSERT IGNORE INTO stands (stand_id)
SELECT DISTINCT stand_id
FROM trips
WHERE stand_id IS NOT NULL;

/* ----------------------------------------------------------------------
   6. (Re)compute summary metrics to populate the new analytical columns
---------------------------------------------------------------------- */
UPDATE trips
SET average_speed_kmh = CASE
        WHEN total_distance_m IS NULL OR duration_seconds IS NULL OR duration_seconds <= 0 THEN NULL
        ELSE (total_distance_m / 1000.0) / (duration_seconds / 3600.0)
    END;

UPDATE trips
SET is_outlier = (
        (total_distance_m IS NOT NULL AND total_distance_m > 200000)
        OR (duration_seconds IS NOT NULL AND duration_seconds > 21600)
        OR (average_speed_kmh IS NOT NULL AND average_speed_kmh > 160)
    );

/* ----------------------------------------------------------------------
   7. Create or refresh supporting indexes (use IF NOT EXISTS to remain idempotent)
---------------------------------------------------------------------- */
ALTER TABLE trips
    ADD INDEX IF NOT EXISTS idx_trips_taxi_start (taxi_id, start_time),
    ADD INDEX IF NOT EXISTS idx_trips_call_type (call_type, day_type),
    ADD INDEX IF NOT EXISTS idx_trips_start_end (start_time, end_time),
    ADD INDEX IF NOT EXISTS idx_trips_start_coords (start_latitude, start_longitude),
    ADD INDEX IF NOT EXISTS idx_trips_end_coords (end_latitude, end_longitude),
    ADD INDEX IF NOT EXISTS idx_trips_valid (is_valid, is_outlier),
    ADD INDEX IF NOT EXISTS idx_trips_client (client_id),
    ADD INDEX IF NOT EXISTS idx_trips_stand (stand_id);

ALTER TABLE trip_points
    ADD INDEX IF NOT EXISTS idx_trip_points_time (point_time),
    ADD INDEX IF NOT EXISTS idx_trip_points_lat_lon (latitude, longitude);

/* ----------------------------------------------------------------------
   8. Apply foreign keys now that data quality is guaranteed
---------------------------------------------------------------------- */
ALTER TABLE trips
    ADD CONSTRAINT fk_trips_taxi FOREIGN KEY IF NOT EXISTS (taxi_id)
        REFERENCES taxis(taxi_id),
    ADD CONSTRAINT fk_trips_client FOREIGN KEY IF NOT EXISTS (client_id)
        REFERENCES clients(client_id),
    ADD CONSTRAINT fk_trips_stand FOREIGN KEY IF NOT EXISTS (stand_id)
        REFERENCES stands(stand_id),
    ADD CONSTRAINT fk_trips_call_type FOREIGN KEY IF NOT EXISTS (call_type)
        REFERENCES call_types(call_type),
    ADD CONSTRAINT fk_trips_day_type FOREIGN KEY IF NOT EXISTS (day_type)
        REFERENCES day_types(day_type);

ALTER TABLE trip_points
    ADD CONSTRAINT fk_trip_points_trip FOREIGN KEY IF NOT EXISTS (trip_id)
        REFERENCES trips(trip_id)
        ON DELETE CASCADE;

/* ----------------------------------------------------------------------
   9. Expose helper view for downstream analytics (drops + recreates)
---------------------------------------------------------------------- */
DROP VIEW IF EXISTS valid_trips;

CREATE VIEW valid_trips AS
SELECT *
FROM trips
WHERE is_valid = TRUE AND is_outlier = FALSE;

/* ----------------------------------------------------------------------
Migration complete
---------------------------------------------------------------------- */
SELECT 'Schema upgrade completed successfully.' AS completion_message;
