from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer
import mysql.connector
import configparser
from datetime import datetime, timedelta
import re
import spacy
from src.logger import logging


class AdvancedArticleSearch:
    def __init__(self, config_path="config/config.ini"):
        """
        Initialize search with configuration and setup
        """
        # Load Configurations
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

        # MySQL Configuration
        self.mysql_config = {
            "host": self.config["MYSQL"]["host"],
            "user": self.config["MYSQL"]["user"],
            "password": self.config["MYSQL"]["password"],
            "database": self.config["MYSQL"]["database"]
        }

        # Load NLP Model
        self.nlp = spacy.load("en_core_web_sm")

        # Initialize Sentence Transformer
        self.model = SentenceTransformer("multi-qa-mpnet-base-cos-v1")

    def parse_date_from_query(self, query):
        """
        Advanced date parsing with support for complex time expressions
        """
        today = datetime.today()
        query = query.lower()

        # Expanded date parsing patterns
        date_patterns = [
            # Year-based queries
            (r'in (\d{4})', lambda match: (
                datetime(int(match.group(1)), 1, 1).date(),
                datetime(int(match.group(1)), 12, 31).date()
            )),
            # Month and year queries
            (r'(\w+) (\d{4})', lambda match: self._parse_month_year(match.group(1), int(match.group(2)))),
            # Relative time queries
            ('last week', lambda _: (today - timedelta(days=7), today)),
            ('last month', lambda _: (
                (today.replace(day=1) - timedelta(days=1)).replace(day=1),
                today.replace(day=1) - timedelta(days=1)
            )),
            ('this month', lambda _: (today.replace(day=1), today)),
            ('yesterday', lambda _: (today - timedelta(days=1), today - timedelta(days=1)))
        ]

        # Check patterns
        for pattern, date_func in date_patterns:
            match = re.search(pattern, query)
            if match:
                return date_func(match)

        # Default to last 30 days
        return (today - timedelta(days=30), today)

    def _parse_month_year(self, month_str, year):
        """
        Helper method to parse month and year
        """
        month_map = {
            'january': 1, 'jan': 1,
            'february': 2, 'feb': 2,
            'march': 3, 'mar': 3,
            'april': 4, 'apr': 4,
            'may': 5,
            'june': 6, 'jun': 6,
            'july': 7, 'jul': 7,
            'august': 8, 'aug': 8,
            'september': 9, 'sep': 9,
            'october': 10, 'oct': 10,
            'november': 11, 'nov': 11,
            'december': 12, 'dec': 12
        }

        month = month_map.get(month_str.lower())
        if month:
            first_day = datetime(year, month, 1).date()
            last_day = (first_day.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            return first_day, last_day

        return datetime(year, 1, 1).date(), datetime(year, 12, 31).date()

    def search_articles(self, query, top_k=50):
        """
        Comprehensive article search using Milvus and MySQL
        """
        try:
            # Parse date from query
            start_date, end_date = self.parse_date_from_query(query)
            logging.info(f"Searching for articles published between {start_date} and {end_date}")

            # Connect to Milvus
            connections.connect(
                host=self.config["MILVUS"]["host"],
                port=self.config["MILVUS"]["port"]
            )
            collection = Collection(self.config["MILVUS"]["collection_name"])
            collection.load()

            # Generate query embedding
            embedding = self.model.encode([query])[0]

            # Flexible search parameters
            search_params = {
                "metric_type": "L2",
                "params": {
                    "nprobe": 20,  # Increased for broader search
                    "ef": 100  # Extended search range
                }
            }

            # Perform semantic search
            results = collection.search(
                data=[embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k
            )

            # Extract article IDs
            article_ids = [hit.id for hit in results[0]]

            if not article_ids:
                logging.info("No relevant articles found in Milvus.")
                return []

            # Fetch and filter articles from MySQL
            connection = mysql.connector.connect(**self.mysql_config)
            cursor = connection.cursor()

            # Prepare placeholders for IN clause
            placeholders = ", ".join(["%s"] * len(article_ids))
            query = f"""
                SELECT id, title, pub_date 
                FROM articles 
                WHERE id IN ({placeholders}) 
                AND pub_date BETWEEN %s AND %s
                ORDER BY pub_date DESC
            """

            cursor.execute(query, article_ids + [start_date, end_date])
            articles = cursor.fetchall()
            connection.close()

            # Display results
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
    Main function to allow user input for searching articles
    """
    search_engine = AdvancedArticleSearch()

    while True:
        # Prompt user for search query
        query = input("Enter a search query (e.g., 'Give me the journals published in 2023'): ")

        if query.strip().lower() == "exit":
            print("Exiting the program.")
            break

        # Perform the search based on user input
        search_engine.search_articles(query)

        # Ask the user if they want to search again or exit
        user_choice = input("\nDo you want to search again? (yes/no): ").strip().lower()
        if user_choice != "yes":
            print("Exiting the program.")
            break


if __name__ == "__main__":
    main()