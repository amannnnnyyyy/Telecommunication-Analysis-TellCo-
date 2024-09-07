import psycopg2
from psycopg2 import sql
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

database_Obj = os.environ.get('DB_Object')


# os.getenv

def returnFromDb():
    # dbName = 'telecom'
    # Database connection parameters
    conn = psycopg2.connect(
        host= os.environ.get("DB_OBJECT_HOST"),
        database= os.environ.get("DB_OBJECT_DATABASE"),
        user= os.environ.get("DB_OBJECT_USER"),
        password= os.environ.get("DB_OBJECT_PASSWORD")

    )

    # Open a cursor to perform database operations
    cur = conn.cursor()

    # Define your query
    query = "SELECT * FROM xdr_data;"

    # Execute the query
    try:
        cur.execute(query)
        # Fetch all results
        rows = cur.fetchall()
        # Print results
        colnames = [desc[0] for desc in cur.description]
        data = pd.DataFrame(rows, columns=colnames)
    except Exception as e:
        print(f"An error occurred while executing the query: {e}")
        data = pd.DataFrame()
    # Close communication with the database
    cur.close()
    conn.close()

    return data

