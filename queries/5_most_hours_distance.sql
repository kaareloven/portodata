SELECT 
    taxi_id,
    SUM(duration_seconds) / 3600.0 as total_hours,
    SUM(total_distance_m) / 1000.0 as total_distance_km
FROM trips
GROUP BY taxi_id
ORDER BY total_hours DESC
LIMIT 20