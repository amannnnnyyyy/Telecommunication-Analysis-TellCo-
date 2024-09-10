import psycopg2
from psycopg2 import sql
import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# database_Obj = os.environ.get('DB_Object')


# os.getenv

def returnFromDb():
    # dbName = 'telecom'
    # Database connection parameters
    conn = psycopg2.connect(
        host= os.environ.get("DB_OBJECT_HOST"),
        database= os.environ.get("DB_OBJECT_DATABASE"),
        user= os.environ.get("DB_OBJECT_USER"),
        password= os.environ.get("DB_OBJECT_PASSWORD"),
        port = 19773

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

def insert_data_to_postgres(df, table_name):
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(
            host=os.environ.get("DB_OBJECT_HOST"),
            port='5433',
            database=os.environ.get("DB_OBJECT_DATABASE"),
            user=os.environ.get("DB_OBJECT_USER"),
            password=os.environ.get("DB_OBJECT_PASSWORD")
        )
        cur = conn.cursor()

        # Create table if it does not exist
        create_table_query = sql.SQL("""
            CREATE TABLE IF NOT EXISTS {table} (
                index SERIAL PRIMARY KEY,
                Cluster INT,
                Satisfaction_Score FLOAT,
                Experience_Score FLOAT
            )
        """).format(table=sql.Identifier(table_name))

        cur.execute(create_table_query)

        # Insert data into the table
        for _, row in df.iterrows():
            # Convert NumPy types to Python native types using .item()
            insert_query = sql.SQL("""
                INSERT INTO {table} (Cluster, Satisfaction_Score, Experience_Score)
                VALUES (%s, %s, %s)
            """).format(table=sql.Identifier(table_name))

            cur.execute(insert_query, (int(row['Cluster']), float(row['Satisfaction Score'].item()), float(row['Experience Score'].item())))

        # Commit changes and close the connection
        conn.commit()
        cur.close()
        conn.close()

        print(f"Data successfully inserted into {table_name} table.")

    except Exception as e:
        print(f"An error occurred: {e}")
