SELECT DISTINCT 
    t.trip_id,
    t.taxi_id,
    t.start_time,
    t.total_distance_m
FROM trip_points tp
JOIN trips t ON tp.trip_id = t.trip_id
WHERE 
    tp.longitude BETWEEN (-8.62911-0.0012) AND (-8.62911+0.0012)
    AND tp.latitude BETWEEN (41.15794-0.0009) AND (41.15794+0.0009)
ORDER BY t.start_time