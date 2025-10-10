SELECT 
    COUNT(*) / COUNT(DISTINCT taxi_id) as avg_trips_per_taxi
FROM trips