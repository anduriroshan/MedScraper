from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer
import mysql.connector
import configparser
from datetime import datetime, timedelta
import re
import spacy
from src.logger import logging

# Load Configurations
config = configparser.ConfigParser()
config.read("config/config.ini")

MYSQL_CONFIG = {
    "host": config["MYSQL"]["host"],
    "user": config["MYSQL"]["user"],
    "password": config["MYSQL"]["password"],
    "database": config["MYSQL"]["database"]
}

# Load English NLP Model for Query Parsing
nlp = spacy.load("en_core_web_sm")


def parse_advanced_date_from_query(query):
    """
    Advanced date parsing with support for more complex time expressions.

    Input:
        - query (str): The natural language query containing time-related expressions (e.g., "last week", "2023 year").

    Output:
        - tuple: A tuple containing the start and end dates derived from the query (start_date, end_date).

    Purpose:
        This function parses natural language time expressions such as "last week", "yesterday", "this month", "2023 year", etc.
        and returns the corresponding date range. It provides support for both relative and year-based queries.
    """
    today = datetime.today()
    query = query.lower()

    # Patterns for year-based queries
    year_patterns = [
        r'in (\d{4})',  # "in 2023"
        r'year (\d{4})',  # "year 2023"
        r'(\d{4}) year',  # "2023 year"
    ]

    # Check for year-specific queries first
    for pattern in year_patterns:
        match = re.search(pattern, query)
        if match:
            year = int(match.group(1))
            return datetime(year, 1, 1).date(), datetime(year, 12, 31).date()

    # Existing time-based parsing from original function
    if "last week" in query:
        start_date = today - timedelta(days=7)
    elif "yesterday" in query:
        start_date = today - timedelta(days=1)
    elif "this month" in query:
        start_date = today.replace(day=1)
    elif "last month" in query:
        first_day_of_this_month = today.replace(day=1)
        start_date = first_day_of_this_month - timedelta(days=1)
        start_date = start_date.replace(day=1)
    else:
        # Default to a wide range if no specific time is found
        start_date = today - timedelta(days=30)

    return start_date.date(), today.date()


def fetch_articles_from_mysql(ids, start_date, end_date):
    """
    Fetch articles by their IDs and filter by publication date.

    Input:
        - ids (list): List of article IDs to fetch from MySQL.
        - start_date (date): The start date for filtering articles.
        - end_date (date): The end date for filtering articles.

    Output:
        - list: A list of tuples containing article data (id, title, pub_date).

    Purpose:
        This function retrieves articles from the MySQL database based on the specified article IDs and filters them
        by publication date within the given range (start_date to end_date).
    """
    connection = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = connection.cursor()

    placeholders = ", ".join(["%s"] * len(ids))  # Prepare placeholders for IN clause
    query = f"""
        SELECT id, title, pub_date 
        FROM articles 
        WHERE id IN ({placeholders}) AND pub_date BETWEEN %s AND %s
    """

    cursor.execute(query, ids + [start_date, end_date])
    articles = cursor.fetchall()
    connection.close()

    return articles


def search_articles(query):
    """
    Enhanced search function with improved date parsing and semantic understanding.

    Input:
        - query (str): The natural language query for searching articles (e.g., "Give me the journals published last week").

    Output:
        - list: A list of articles that match the search criteria (filtered by date and relevance).

    Purpose:
        This function takes a natural language query, parses the date range (using the `parse_advanced_date_from_query` function),
        and then performs a semantic search in Milvus to find articles that match the query. It retrieves article IDs from Milvus,
        fetches the corresponding articles from MySQL, and returns the results filtered by date.
    """
    try:
        # Step 1: Parse the date range from the query
        start_date, end_date = parse_advanced_date_from_query(query)
        logging.info(f"Searching for articles published between {start_date} and {end_date}")

        # Step 2: Connect to Milvus and perform semantic search
        connections.connect(host=config["MILVUS"]["host"], port=config["MILVUS"]["port"])
        collection = Collection(config["MILVUS"]["collection_name"])
        collection.load()

        # Use a more advanced embedding model for better semantic understanding
        model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")  # More powerful model
        embedding = model.encode([query])[0]

        # More flexible search parameters
        search_params = {
            "metric_type": "L2",
            "params": {
                "nprobe": 20,  # Increased for broader search
                "ef": 100  # Extended search range
            }
        }

        # Increased search limit to capture more potential matches
        results = collection.search([embedding], anns_field="embedding", param=search_params, limit=50)

        # Step 3: Retrieve IDs from Milvus results
        article_ids = [hit.id for hit in results[0]]
        if not article_ids:
            logging.info("No relevant articles found in Milvus.")
            return []

        # Step 4: Fetch details from MySQL and filter by date
        logging.info("Fetching article details from MySQL...")
        articles = fetch_articles_from_mysql(article_ids, start_date, end_date)

        # Step 5: Display and return results
        if articles:
            print("\nSearch Results:")
            for article in articles:
                print(f"ID: {article[0]}, Title: {article[1]}, Date: {article[2]}")
            return articles
        else:
            logging.info("No articles match the specified date range.")
            return []

    except Exception as e:
        logging.error(f"An error occurred during search: {e}")
        return []


def main():
    """
    Main function to allow user input for searching articles.

    Input: None
    Output: None

    Purpose:
        This function runs an interactive loop allowing the user to input a natural language query and retrieve articles
        that match the query (filtered by date). The user can continue searching or exit the program based on input.
    """
    while True:
        # Prompt user for search query
        query = input("Enter a search query (e.g., 'Give me the journals published last week'): ")

        if query.strip().lower() == "exit":
            print("Exiting the program.")
            break

        # Perform the search based on user input
        search_articles(query)

        # Ask the user if they want to search again or exit
        user_choice = input("\nDo you want to search again? (yes/no): ").strip().lower()
        if user_choice != "yes":
            print("Exiting the program.")
            break


if __name__ == "__main__":
    main()
