SELECT 
    taxi_id,
    COUNT(*) as trip_count
FROM trips
GROUP BY taxi_id
ORDER BY trip_count DESC
LIMIT 20