SELECT 
    taxi_id,
    call_type,
    trip_count
FROM (
    SELECT 
        taxi_id,
        call_type,
        COUNT(*) as trip_count,
        ROW_NUMBER() OVER (PARTITION BY taxi_id ORDER BY COUNT(*) DESC) as rn
    FROM trips
    GROUP BY taxi_id, call_type
) ranked
WHERE rn = 1
ORDER BY trip_count DESC
LIMIT 20
