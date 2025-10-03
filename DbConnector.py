import os
import mysql.connector as mysql
from mysql.connector import Error

# Optional: load .env file when present. This requires python-dotenv in requirements.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # If python-dotenv isn't available, we still allow environment variables set in the environment.
    pass


class DbConnector:
    """DbConnector reads DB connection parameters from environment variables.

    Environment variables (see `.env` or OS environment):
      - DB_HOST (default: 127.0.0.1)
      - DB_DATABASE (default: porto)
      - DB_PORT (default: 3306)
      - DB_USER (default: root)
      - DB_PASSWORD (default: password)

    The constructor will attempt to establish a connection and create a cursor.
    On failure it raises the underlying exception.
    """

    def __init__(self,
                 HOST: str = None,
                 DATABASE: str = None,
                 PORT: int = None,
                 USER: str = None,
                 PASSWORD: str = None):
        # Read from provided args or environment variables with sensible defaults
        host = HOST or os.getenv("DB_HOST", "127.0.0.1")
        database = DATABASE or os.getenv("DB_DATABASE", "porto")
        port_str = PORT or os.getenv("DB_PORT", os.getenv("PORT", "3306"))
        user = USER or os.getenv("DB_USER", "root")
        password = PASSWORD or os.getenv("DB_PASSWORD", "password")

        # Ensure port is an int
        try:
            port = int(port_str)
        except (TypeError, ValueError):
            port = 3306

        self.db_connection = None
        self.cursor = None

        # Connect to the database
        try:
            self.db_connection = mysql.connect(host=host, database=database, user=user, password=password, port=port)
            # Get the db cursor
            self.cursor = self.db_connection.cursor()

            server_info = None
            try:
                server_info = self.db_connection.get_server_info()
            except Exception:
                # get_server_info may fail on some connection errors
                server_info = "(unknown)"

            print("Connected to:", server_info)
            # get database information
            try:
                self.cursor.execute("select database();")
                database_name = self.cursor.fetchone()
                print("You are connected to the database:", database_name)
            except Exception:
                # Ignore database() query errors
                pass

            print("-----------------------------------------------\n")

        except Error as e:
            # Surface a clearer error and re-raise so callers can handle it
            print("ERROR: Failed to connect to db:", e)
            raise

    def close_connection(self):
        """Close cursor and DB connection if open."""
        if self.cursor:
            try:
                self.cursor.close()
            except Exception:
                pass
        if self.db_connection:
            try:
                # Retrieve server info for message before closing
                try:
                    server_info = self.db_connection.get_server_info()
                except Exception:
                    server_info = "(unknown)"
                self.db_connection.close()
            except Exception:
                pass

        print("\n-----------------------------------------------")
        print("Connection closed")
