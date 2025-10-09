"""
Porto taxi data analysis queries.
Each function corresponds to a query from the assignment.
"""

from DbConnector import DbConnector
from tabulate import tabulate
import os
import time


class PortoQueryAnalysis:
    """Class to run analytical queries on the Porto taxi database."""
    
    def __init__(self):
        # Set the correct password for Docker MySQL
        os.environ['DB_PASSWORD'] = 'test1234'
        
        self.connection = DbConnector()
        self.db_connection = self.connection.db_connection
        self.cursor = self.connection.cursor
    
    def query_1_basic_statistics(self):
        """
        Query 1: How many taxis, trips, and total GPS points are there?
        """
        start_time = time.time()
        
        print("\n" + "=" * 80)
        print("QUERY 1: Basic Statistics")
        print("=" * 80 + "\n")
        
        # Count distinct taxis
        self.cursor.execute("SELECT COUNT(DISTINCT TAXI_ID) FROM TRIP")
        num_taxis = self.cursor.fetchone()[0]
        
        # Count trips
        self.cursor.execute("SELECT COUNT(*) FROM TRIP")
        num_trips = self.cursor.fetchone()[0]
        
        # Count total GPS points
        self.cursor.execute("SELECT COUNT(*) FROM TRIP_POINT")
        num_points = self.cursor.fetchone()[0]
        
        # Create result table
        results = [
            ["Number of Taxis", f"{num_taxis:,}"],
            ["Number of Trips", f"{num_trips:,}"],
            ["Total GPS Points", f"{num_points:,}"],
        ]
        
        print(tabulate(results, headers=["Metric", "Count"], tablefmt="grid"))
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Query execution time: {elapsed:.2f} seconds")
        print()
        
        return {
            "taxis": num_taxis,
            "trips": num_trips,
            "points": num_points
        }
    
    def query_2_average_trips_per_taxi(self):
        """
        Query 2: What is the average number of trips per taxi?
        """
        start_time = time.time()
        
        print("\n" + "=" * 80)
        print("QUERY 2: Average Number of Trips per Taxi")
        print("=" * 80 + "\n")
        
        # Calculate average trips per taxi
        query = """
            SELECT 
                COUNT(*) / COUNT(DISTINCT TAXI_ID) as avg_trips_per_taxi
            FROM TRIP
        """
        self.cursor.execute(query)
        avg_trips = self.cursor.fetchone()[0]
        
        print(f"Average trips per taxi: {avg_trips:.2f}")
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Query execution time: {elapsed:.2f} seconds")
        print()
        
        return avg_trips
    
    def query_3_top_20_taxis_most_trips(self):
        """
        Query 3: List the top 20 taxis with the most trips.
        """
        start_time = time.time()
        
        print("\n" + "=" * 80)
        print("QUERY 3: Top 20 Taxis with Most Trips")
        print("=" * 80 + "\n")
        
        query = """
            SELECT 
                TAXI_ID,
                COUNT(*) as trip_count
            FROM TRIP
            GROUP BY TAXI_ID
            ORDER BY trip_count DESC
            LIMIT 20
        """
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        
        # Format results for tabulate
        formatted_results = [
            [i+1, taxi_id, f"{trip_count:,}"] 
            for i, (taxi_id, trip_count) in enumerate(results)
        ]
        
        print(tabulate(formatted_results, 
                      headers=["Rank", "Taxi ID", "Trip Count"], 
                      tablefmt="grid"))
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Query execution time: {elapsed:.2f} seconds")
        print()
        
        return results
    
    def query_4_call_type_analysis(self):
        """
        Query 4: Call type analysis
        a) What is the most used call type per taxi?
        b) For each call type, compute average trip duration and distance,
           and share of trips in time bands (00-06, 06-12, 12-18, 18-24)
        """
        start_time = time.time()
        
        print("\n" + "=" * 80)
        print("QUERY 4: Call Type Analysis")
        print("=" * 80 + "\n")
        
        # Part a) Most used call type per taxi
        print("Part a) Most used call type per taxi (showing first 20 taxis):")
        print("-" * 80)
        
        query_a = """
            SELECT 
                TAXI_ID,
                CALL_TYPE,
                trip_count
            FROM (
                SELECT 
                    TAXI_ID,
                    CALL_TYPE,
                    COUNT(*) as trip_count,
                    ROW_NUMBER() OVER (PARTITION BY TAXI_ID ORDER BY COUNT(*) DESC) as rn
                FROM TRIP
                GROUP BY TAXI_ID, CALL_TYPE
            ) ranked
            WHERE rn = 1
            ORDER BY TAXI_ID
            LIMIT 20
        """
        self.cursor.execute(query_a)
        results_a = self.cursor.fetchall()
        
        formatted_a = [
            [taxi_id, call_type, f"{count:,}"] 
            for taxi_id, call_type, count in results_a
        ]
        
        print(tabulate(formatted_a, 
                      headers=["Taxi ID", "Most Used Call Type", "Trip Count"], 
                      tablefmt="grid"))
        print()
        
        # Part b) For each call type, compute averages and time band distribution
        print("Part b) Call type statistics:")
        print("-" * 80)
        
        query_b = """
            SELECT 
                CALL_TYPE,
                COUNT(*) as total_trips,
                AVG(DURATION_SEC) as avg_duration_sec,
                AVG(DISTANCE_KM) as avg_distance_km,
                SUM(CASE WHEN HOUR(START_TIMESTAMP) >= 0 AND HOUR(START_TIMESTAMP) < 6 THEN 1 ELSE 0 END) as band_00_06,
                SUM(CASE WHEN HOUR(START_TIMESTAMP) >= 6 AND HOUR(START_TIMESTAMP) < 12 THEN 1 ELSE 0 END) as band_06_12,
                SUM(CASE WHEN HOUR(START_TIMESTAMP) >= 12 AND HOUR(START_TIMESTAMP) < 18 THEN 1 ELSE 0 END) as band_12_18,
                SUM(CASE WHEN HOUR(START_TIMESTAMP) >= 18 AND HOUR(START_TIMESTAMP) < 24 THEN 1 ELSE 0 END) as band_18_24
            FROM TRIP
            GROUP BY CALL_TYPE
            ORDER BY total_trips DESC
        """
        self.cursor.execute(query_b)
        results_b = self.cursor.fetchall()
        
        formatted_b = []
        for row in results_b:
            call_type, total, avg_dur, avg_dist, b1, b2, b3, b4 = row
            formatted_b.append([
                call_type,
                f"{total:,}",
                f"{avg_dur:.2f}s",
                f"{avg_dist:.2f} km",
                f"{(b1/total*100):.1f}%",
                f"{(b2/total*100):.1f}%",
                f"{(b3/total*100):.1f}%",
                f"{(b4/total*100):.1f}%"
            ])
        
        print(tabulate(formatted_b, 
                      headers=["Call Type", "Total Trips", "Avg Duration", "Avg Distance",
                              "00-06", "06-12", "12-18", "18-24"], 
                      tablefmt="grid"))
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Query execution time: {elapsed:.2f} seconds")
        print()
        
        return {"part_a": results_a, "part_b": results_b}
    
    def query_5_taxis_most_hours_and_distance(self):
        """
        Query 5: Find the taxis with the most total hours driven as well as total distance driven.
        List them in order of total hours.
        """
        start_time = time.time()
        
        print("\n" + "=" * 80)
        print("QUERY 5: Taxis with Most Total Hours and Distance Driven")
        print("=" * 80 + "\n")
        
        query = """
            SELECT 
                TAXI_ID,
                SUM(DURATION_SEC) / 3600.0 as total_hours,
                SUM(DISTANCE_KM) as total_distance_km
            FROM TRIP
            GROUP BY TAXI_ID
            ORDER BY total_hours DESC
            LIMIT 20
        """
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        
        # Format results for tabulate
        formatted_results = [
            [i+1, taxi_id, f"{hours:.2f}", f"{distance:.2f}"] 
            for i, (taxi_id, hours, distance) in enumerate(results)
        ]
        
        print(tabulate(formatted_results, 
                      headers=["Rank", "Taxi ID", "Total Hours", "Total Distance (km)"], 
                      tablefmt="grid"))
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Query execution time: {elapsed:.2f} seconds")
        print()
        
        return results
    
    def query_6_trips_near_city_hall(self):
        """
        Query 6: Find the trips that passed within 100 m of Porto City Hall.
        (longitude, latitude) = (-8.62911, 41.15794)
        
        Simplified: Using bounding box approximation (~100m radius).
        This is much faster than precise distance calculation.
        """
        start_time = time.time()
        
        print("\n" + "=" * 80)
        print("QUERY 6: Trips Near Porto City Hall")
        print("=" * 80 + "\n")
        print("🔄 Running query (may take 10-30 seconds)...")
        
        # Porto City Hall coordinates
        city_hall_lon = -8.62911
        city_hall_lat = 41.15794
        
        # Bounding box for ~100m radius at Porto's latitude (~41°)
        # 1° latitude ≈ 111km, 1° longitude ≈ 85km at 41° latitude
        lon_offset = 0.0012  # ~100m in longitude
        lat_offset = 0.0009  # ~100m in latitude
        
        # Simplified query using only bounding box (no expensive distance calculation)
        query = """
            SELECT DISTINCT 
                t.TRIP_ID,
                t.TAXI_ID,
                t.START_TIMESTAMP,
                t.DISTANCE_KM
            FROM TRIP_POINT tp
            JOIN TRIP t ON tp.TRIP_ID = t.TRIP_ID
            WHERE 
                tp.LONGITUDE BETWEEN %s AND %s
                AND tp.LATITUDE BETWEEN %s AND %s
            ORDER BY t.START_TIMESTAMP
        """
        
        self.cursor.execute(query, (
            city_hall_lon - lon_offset, city_hall_lon + lon_offset,
            city_hall_lat - lat_offset, city_hall_lat + lat_offset
        ))
        results = self.cursor.fetchall()
        
        print(f"\n✅ Total trips passing near Porto City Hall: {len(results):,}")
        print(f"   (within bounding box of ~100m radius)\n")
        
        if results:
            # Show first 20 trips
            display_results = results[:20]
            formatted_results = [
                [trip_id, taxi_id, start_time, f"{distance:.2f}"] 
                for trip_id, taxi_id, start_time, distance in display_results
            ]
            
            print(f"Showing first {len(display_results)} trips:")
            print(tabulate(formatted_results, 
                          headers=["Trip ID", "Taxi ID", "Start Time", "Distance (km)"], 
                          tablefmt="grid"))
            
            if len(results) > 20:
                print(f"\n... and {len(results) - 20} more trips")
        else:
            print("No trips found near Porto City Hall.")
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Query execution time: {elapsed:.2f} seconds")
        print()
        
        return results
    
    def query_7_invalid_trips(self):
        """
        Query 7: Identify the number of invalid trips.
        An invalid trip is defined as a trip with fewer than 3 GPS points.
        """
        start_time = time.time()
        
        print("\n" + "=" * 80)
        print("QUERY 7: Invalid Trips (< 3 GPS Points)")
        print("=" * 80 + "\n")
        
        query = """
            SELECT COUNT(*) as invalid_count
            FROM TRIP
            WHERE N_POINTS < 3
        """
        
        self.cursor.execute(query)
        invalid_count = self.cursor.fetchone()[0]
        
        print(f"Number of invalid trips (< 3 GPS points): {invalid_count:,}")
        
        # Show some examples if there are any
        if invalid_count > 0:
            query_examples = """
                SELECT 
                    TRIP_ID,
                    TAXI_ID,
                    START_TIMESTAMP,
                    N_POINTS
                FROM TRIP
                WHERE N_POINTS < 3
                ORDER BY N_POINTS, TRIP_ID
                LIMIT 20
            """
            self.cursor.execute(query_examples)
            examples = self.cursor.fetchall()
            
            print(f"\nShowing first {len(examples)} invalid trips:")
            formatted = [
                [trip_id, taxi_id, start_time, n_points]
                for trip_id, taxi_id, start_time, n_points in examples
            ]
            print(tabulate(formatted,
                          headers=["Trip ID", "Taxi ID", "Start Time", "GPS Points"],
                          tablefmt="grid"))
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Query execution time: {elapsed:.2f} seconds")
        print()
        
        return invalid_count
    
    def query_8_nearby_taxi_pairs(self):
        """
        Query 8: Find pairs of different taxis that were within 5m and within 5 seconds
        of each other at least once.
        
        Note: This is a complex query that may take several minutes.
        """
        start_time = time.time()
        
        print("\n" + "=" * 80)
        print("QUERY 8: Taxi Pairs Within 5m and 5 Seconds")
        print("=" * 80 + "\n")
        print("⚠️  NOTE: This query involves self-join on 40M rows.")
        print("🔄 Running query (may take 2-5 minutes)...")
        print("   Please be patient...")
        print()
        
        # Simplified approach: use bounding box and time filters first
        query = """
            SELECT DISTINCT
                t1.TAXI_ID as taxi_1,
                t2.TAXI_ID as taxi_2,
                COUNT(*) as encounter_count
            FROM TRIP_POINT tp1
            JOIN TRIP_POINT tp2 ON 
                tp1.TRIP_ID < tp2.TRIP_ID
                AND ABS(TIMESTAMPDIFF(SECOND, tp1.POINT_TIME, tp2.POINT_TIME)) <= 5
                AND ABS(tp1.LATITUDE - tp2.LATITUDE) <= 0.00005
                AND ABS(tp1.LONGITUDE - tp2.LONGITUDE) <= 0.00005
                AND ST_Distance_Sphere(tp1.GPS_POINT, tp2.GPS_POINT) <= 5
            JOIN TRIP t1 ON tp1.TRIP_ID = t1.TRIP_ID
            JOIN TRIP t2 ON tp2.TRIP_ID = t2.TRIP_ID
            WHERE t1.TAXI_ID < t2.TAXI_ID
            GROUP BY t1.TAXI_ID, t2.TAXI_ID
            ORDER BY encounter_count DESC
            LIMIT 20
        """
        
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        
        print(f"\n✅ Found {len(results)} taxi pairs that were within 5m and 5s")
        print(f"   (showing top 20 by encounter count)\n")
        
        if results:
            formatted = [
                [taxi1, taxi2, f"{count:,}"]
                for taxi1, taxi2, count in results
            ]
            print(tabulate(formatted,
                          headers=["Taxi 1", "Taxi 2", "Encounters"],
                          tablefmt="grid"))
        else:
            print("No taxi pairs found within 5m and 5 seconds of each other.")
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Query execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        print()
        
        return results
    
    def query_9_midnight_crossers(self):
        """
        Query 9: Find trips that started on one calendar day and ended on the next
        (midnight crossers).
        """
        start_time = time.time()
        
        print("\n" + "=" * 80)
        print("QUERY 9: Trips Crossing Midnight")
        print("=" * 80 + "\n")
        
        query = """
            SELECT 
                TRIP_ID,
                TAXI_ID,
                START_TIMESTAMP,
                DATE_ADD(START_TIMESTAMP, INTERVAL DURATION_SEC SECOND) as end_timestamp,
                DURATION_SEC / 60 as duration_minutes
            FROM TRIP
            WHERE DATE(START_TIMESTAMP) != DATE(DATE_ADD(START_TIMESTAMP, INTERVAL DURATION_SEC SECOND))
            ORDER BY START_TIMESTAMP
            LIMIT 20
        """
        
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        
        # Get total count
        count_query = """
            SELECT COUNT(*)
            FROM TRIP
            WHERE DATE(START_TIMESTAMP) != DATE(DATE_ADD(START_TIMESTAMP, INTERVAL DURATION_SEC SECOND))
        """
        self.cursor.execute(count_query)
        total_count = self.cursor.fetchone()[0]
        
        print(f"Total trips crossing midnight: {total_count:,}\n")
        
        if results:
            print(f"Showing first {len(results)} trips:")
            formatted = [
                [trip_id, taxi_id, start_time, end_time, f"{duration:.1f}"]
                for trip_id, taxi_id, start_time, end_time, duration in results
            ]
            print(tabulate(formatted,
                          headers=["Trip ID", "Taxi ID", "Start Time", "End Time", "Duration (min)"],
                          tablefmt="grid"))
        else:
            print("No trips found crossing midnight.")
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Query execution time: {elapsed:.2f} seconds")
        print()
        
        return results
    
    def query_10_circular_trips(self):
        """
        Query 10: Find trips whose start and end points are within 50m of each other
        (circular trips).
        """
        start_time = time.time()
        
        print("\n" + "=" * 80)
        print("QUERY 10: Circular Trips (Start/End Within 50m)")
        print("=" * 80 + "\n")
        
        query = """
            SELECT 
                TRIP_ID,
                TAXI_ID,
                START_TIMESTAMP,
                DISTANCE_KM,
                DURATION_SEC / 60 as duration_minutes,
                ST_Distance_Sphere(START_POINT, END_POINT) as start_end_distance
            FROM TRIP
            WHERE ST_Distance_Sphere(START_POINT, END_POINT) <= 50
            ORDER BY start_end_distance
            LIMIT 20
        """
        
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        
        # Get total count
        count_query = """
            SELECT COUNT(*)
            FROM TRIP
            WHERE ST_Distance_Sphere(START_POINT, END_POINT) <= 50
        """
        self.cursor.execute(count_query)
        total_count = self.cursor.fetchone()[0]
        
        print(f"Total circular trips (start/end within 50m): {total_count:,}\n")
        
        if results:
            print(f"Showing first {len(results)} trips (sorted by start-end distance):")
            formatted = [
                [trip_id, taxi_id, start_time, f"{dist_km:.2f}", f"{duration:.1f}", f"{se_dist:.1f}"]
                for trip_id, taxi_id, start_time, dist_km, duration, se_dist in results
            ]
            print(tabulate(formatted,
                          headers=["Trip ID", "Taxi ID", "Start Time", "Trip Dist (km)", 
                                  "Duration (min)", "Start-End (m)"],
                          tablefmt="grid"))
        else:
            print("No circular trips found.")
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Query execution time: {elapsed:.2f} seconds")
        print()
        
        return results
    
    def query_11_highest_idle_time(self):
        """
        Query 11: For each taxi, compute the average idle time between consecutive trips.
        List the top 20 taxis with the highest average idle time.
        """
        start_time = time.time()
        
        print("\n" + "=" * 80)
        print("QUERY 11: Top 20 Taxis with Highest Average Idle Time")
        print("=" * 80 + "\n")
        
        query = """
            WITH trip_pairs AS (
                SELECT 
                    TAXI_ID,
                    START_TIMESTAMP,
                    DATE_ADD(START_TIMESTAMP, INTERVAL DURATION_SEC SECOND) as end_timestamp,
                    LEAD(START_TIMESTAMP) OVER (PARTITION BY TAXI_ID ORDER BY START_TIMESTAMP) as next_start
                FROM TRIP
            ),
            idle_times AS (
                SELECT 
                    TAXI_ID,
                    TIMESTAMPDIFF(SECOND, end_timestamp, next_start) as idle_seconds
                FROM trip_pairs
                WHERE next_start IS NOT NULL
                AND TIMESTAMPDIFF(SECOND, end_timestamp, next_start) >= 0
            )
            SELECT 
                TAXI_ID,
                COUNT(*) as num_gaps,
                AVG(idle_seconds) as avg_idle_seconds,
                AVG(idle_seconds) / 3600 as avg_idle_hours,
                MAX(idle_seconds) / 3600 as max_idle_hours
            FROM idle_times
            GROUP BY TAXI_ID
            HAVING COUNT(*) >= 2
            ORDER BY avg_idle_seconds DESC
            LIMIT 20
        """
        
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        
        print(f"Top 20 taxis with highest average idle time:\n")
        
        if results:
            formatted = [
                [i+1, taxi_id, f"{gaps:,}", f"{avg_sec/60:.1f}", f"{avg_hrs:.2f}", f"{max_hrs:.2f}"]
                for i, (taxi_id, gaps, avg_sec, avg_hrs, max_hrs) in enumerate(results)
            ]
            print(tabulate(formatted,
                          headers=["Rank", "Taxi ID", "# Gaps", "Avg Idle (min)", 
                                  "Avg Idle (hrs)", "Max Idle (hrs)"],
                          tablefmt="grid"))
        else:
            print("No results found.")
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Query execution time: {elapsed:.2f} seconds")
        print()
        
        return results


def display_menu():
    """Display the query selection menu."""
    print("\n" + "🚕" * 40)
    print("PORTO TAXI DATA ANALYSIS - QUERY MENU")
    print("🚕" * 40)
    print("\nAvailable Queries:")
    print("-" * 80)
    print("  1. How many taxis, trips, and total GPS points are there?")
    print("  2. What is the average number of trips per taxi?")
    print("  3. List the top 20 taxis with the most trips")
    print("  4. Call type analysis (most used per taxi, averages, time bands)")
    print("  5. Taxis with most total hours and distance driven")
    print("  6. Trips within 100m of Porto City Hall")
    print("  7. Number of invalid trips (< 3 GPS points)")
    print("  8. Pairs of taxis within 5m and 5 seconds")
    print("  9. Trips crossing midnight")
    print(" 10. Circular trips (start/end within 50m)")
    print(" 11. Top 20 taxis with highest average idle time")
    print("-" * 80)
    print("  0. Run ALL queries")
    print("  q. Quit")
    print("=" * 80)


def main():
    """Main function with interactive menu."""
    program = None
    
    try:
        program = PortoQueryAnalysis()
        
        # Dictionary mapping choices to query functions
        query_functions = {
            '1': program.query_1_basic_statistics,
            '2': program.query_2_average_trips_per_taxi,
            '3': program.query_3_top_20_taxis_most_trips,
            '4': program.query_4_call_type_analysis,
            '5': program.query_5_taxis_most_hours_and_distance,
            '6': program.query_6_trips_near_city_hall,
            '7': program.query_7_invalid_trips,
            '8': program.query_8_nearby_taxi_pairs,
            '9': program.query_9_midnight_crossers,
            '10': program.query_10_circular_trips,
            '11': program.query_11_highest_idle_time,
        }
        
        while True:
            display_menu()
            choice = input("\nEnter your choice: ").strip().lower()
            
            if choice == 'q':
                print("\n👋 Goodbye!")
                break
            
            elif choice == '0':
                # Run all implemented queries
                print("\n" + "=" * 80)
                print("RUNNING ALL QUERIES")
                print("=" * 80)
                for key in sorted(query_functions.keys()):
                    try:
                        query_functions[key]()
                    except Exception as e:
                        print(f"\n❌ ERROR in query {key}: {e}")
                        import traceback
                        traceback.print_exc()
                
                print("\n" + "=" * 80)
                print("All Queries Complete!")
                print("=" * 80)
                
                input("\nPress Enter to return to menu...")
            
            elif choice in query_functions:
                try:
                    query_functions[choice]()
                except Exception as e:
                    print(f"\n❌ ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                
                input("\nPress Enter to return to menu...")
            
            else:
                print("\n⚠️  Invalid choice! Please try again.")
                input("Press Enter to continue...")
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ ERROR: Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if program:
            program.connection.close_connection()


if __name__ == '__main__':
    main()

