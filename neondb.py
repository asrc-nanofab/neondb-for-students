# File: neondb.py
# Author: Samantha Roberts
# Created on: 6/10/2024
# Last modified by:
# Last modified on:
#
# Description:
# ------------------------------------------------------------
# This module provides functionalities for interacting with a PostgreSQL database.
# It includes functions to connect to the database, create and manage tables,
# insert and retrieve data, and perform vector-based searches using the pgvector
# extension. This module is used primarily for operations related to the
# 'labnetwork' table, which stores various pieces of information and embeddings
# for the MIT Labnetwork email forum


import os
import ast
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values
import numpy as np
import dotenv

dotenv.load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


########################################################################


def connect_db():
    """Connect to the PostgreSQL database and ensure pgvector is available."""
    conn = psycopg2.connect(DATABASE_URL)
    # Ensure extension exists and register adapters
    with conn.cursor() as cur:
        try:
            enable_pgvector_extension(cur)
        except Exception:
            pass
        try:
            register_vector(conn)
        except psycopg2.ProgrammingError:
            # If extension wasn't available earlier, try enabling then register again
            enable_pgvector_extension(cur)
            register_vector(conn)
    return conn


########################################################################
# Functions needed for creating and filling the table


def enable_pgvector_extension(cur):
    """Enable the pgvector extension when creating table"""
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    print("pgvector extension enabled")


def create_labnetwork_table():
    """Creates labnetwork table if it does not already exist"""
    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            enable_pgvector_extension(cur)
            # Register after extension exists
            try:
                register_vector(conn)
            except psycopg2.ProgrammingError:
                enable_pgvector_extension(cur)
                register_vector(conn)

            create_table_query = """
            CREATE TABLE IF NOT EXISTS labnetwork (
                id SERIAL PRIMARY KEY,
                sender TEXT,
                email TEXT,
                subject TEXT,
                body TEXT,
                message_id TEXT,
                thread_id TEXT,
                date_time TIMESTAMP,
                cleaned_body TEXT,
                embed VECTOR(1536)
            );
            """
            cur.execute(create_table_query)
            # Create an index for faster vector search (requires pgvector)
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS labnetwork_embed_idx
                ON labnetwork USING ivfflat (embed vector_cosine_ops)
                WITH (lists = 100)
                """
            )
            conn.commit()
    print("Table created successfully.")


def _ensure_vector(value):
    """Coerce various Python representations to a pgvector-compatible numpy array."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, list):
        return np.array(value, dtype=float)
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return np.array(parsed, dtype=float)
        except Exception:
            pass
    return value


def insert_data_to_db(df):
    """Insert data from a DataFrame into the labnetwork table in PostgreSQL."""
    insert_query = """
    INSERT INTO labnetwork (
        sender, email, subject, body,
        message_id, thread_id, date_time,
        cleaned_body, embed
    ) VALUES %s
    """

    data_tuples = [
        (
            row.get("sender"),
            row.get("email"),
            row.get("subject"),
            row.get("body"),
            row.get("message_id"),
            row.get("thread_id"),
            row.get("date_time"),
            row.get("cleaned_body"),
            _ensure_vector(row.get("embed")),
        )
        for index, row in df.iterrows()
    ]

    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            try:
                register_vector(conn)
            except psycopg2.ProgrammingError:
                enable_pgvector_extension(cur)
                register_vector(conn)
            execute_values(cur, insert_query, data_tuples)
            conn.commit()
    print("Data uploaded successfully.")


def check_labnetwork_table_exists():
    """Check if the labnetwork table exists in the database.
    Returns:
        Boolean -- True if exists, False if not
    """
    conn = connect_db()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'labnetwork'
            );
        """)
        exists = cur.fetchone()[0]
    return exists


def delete_labnetwork_table():
    """Deletes the labnetwork table and its associated index from the database.

    WARNING: This will permanently delete all data in the labnetwork table.
    Use with caution!
    """
    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            # Drop the index first (if it exists)
            cur.execute("DROP INDEX IF EXISTS labnetwork_embed_idx")

            # Drop the table
            cur.execute("DROP TABLE IF EXISTS labnetwork")

            conn.commit()
    print("labnetwork table and index deleted successfully.")


def truncate_labnetwork_table():
    """Truncates the labnetwork table, removing all data but keeping the table structure.

    This is faster than DELETE for removing all rows and resets the auto-increment counter.
    """
    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE labnetwork RESTART IDENTITY")
            conn.commit()
    print("labnetwork table truncated successfully.")


###############################################################################
# Index the vectors in the labnetwork database
# This may not be the best method, or lead to the fastest searching
# could be updated in the future to be more efficient


def get_number_of_records_in_table(cur):
    """Get the number of records in the labnetwork table"""
    cur.execute("SELECT COUNT(*) as cnt FROM labnetwork;")
    num_records = cur.fetchone()[0]
    return num_records


###############################################################################
# This is used by the vector search nanobot Labnetwork page


def get_top_k_similar_docs(query_embedding, k):
    """Query the labnetwork table for the top k similar posts

    Args:
        query_embedding (list): embedding of the query vector (currently 1536 d)
        k (int): number of similar posts to return
    Returns:
            top_k_docs (list): list of tuples of the top k similar posts
    """
    try:
        conn = connect_db()
        with conn.cursor() as cur:
            # === Return everything in the table but the vectors ===
            query = """
                SELECT sender, email, subject, body, message_id,
                       thread_id, date_time, cleaned_body
                FROM labnetwork
                ORDER BY embed <=> %s
                LIMIT %s
            """
            # === Must convert vector to array for pgvector to work ===
            cur.execute(query, (np.array(query_embedding), k))
            top_k_docs = cur.fetchall()
        return top_k_docs
    except (psycopg2.OperationalError, psycopg2.InterfaceError):
        print("Connection lost. Reconnecting...")
        return get_top_k_similar_docs(query_embedding, k)


##############################################################################
# Functions for Execututing SQL query
# These are used when calling queeries created by openai LLM
# These are broken down so that a failure can throw an exception and be caught
# by the retry loop
