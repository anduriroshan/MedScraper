from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from sentence_transformers import SentenceTransformer
import mysql.connector
import configparser
from src.logger import logging

# Configurations
config = configparser.ConfigParser()
config.read('config/config.ini')

MILVUS_CONFIG = {"host": config["MILVUS"]["host"], "port": config["MILVUS"]["port"]}
MYSQL_CONFIG = {key: config["MYSQL"][key] for key in config["MYSQL"]}
COLLECTION_NAME = config["MILVUS"]["collection_name"]


def connect_milvus():
    """
    Establish a connection to the Milvus server.

    Input: None
    Output: None

    Purpose:
    This function connects to the Milvus vector database using the configuration settings defined in the 'config/config.ini'.
    It establishes the connection to the Milvus server with the provided host and port settings.
    """
    connections.connect(**MILVUS_CONFIG)


def create_collection():
    """
    Create a new collection in Milvus to store the embeddings.

    Input: None
    Output: Collection object

    Purpose:
    This function creates a new collection in the Milvus vector database. The collection schema includes two fields:
    - 'id' (INT64): A primary key for each article.
    - 'embedding' (FLOAT_VECTOR): The vector representation of the article titles.

    The collection is indexed using 'IVF_FLAT' with 'L2' distance metric, and 'nlist' is set to 128.
    """
    schema = CollectionSchema([
        FieldSchema("id", DataType.INT64, is_primary=True),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=384),  # 384 dimensions for the sentence embeddings
    ])
    collection = Collection(COLLECTION_NAME, schema)
    collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
    return collection


def insert_embeddings():
    """
    Fetch article titles from MySQL and insert their embeddings into Milvus.

    Input: None
    Output: None

    Purpose:
    This function fetches article titles from the 'articles' table in the MySQL database, generates embeddings for each
    article title using the SentenceTransformer model, and inserts the embeddings into the Milvus collection.
    The embeddings are stored as vectors for future similarity search.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")  # Load pre-trained model for sentence embeddings
    connection = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = connection.cursor()
    cursor.execute("SELECT id, title FROM articles")  # Query to fetch article titles
    records = cursor.fetchall()

    embeddings = [model.encode(title) for _, title in records]  # Generate embeddings for each article title
    ids = [record[0] for record in records]  # Extract the article IDs

    collection = Collection(COLLECTION_NAME)  # Access the Milvus collection
    collection.insert([ids, embeddings])  # Insert article IDs and embeddings into Milvus
    logging.info("Embeddings inserted into Milvus.")


if __name__ == "__main__":
    connect_milvus()  # Connect to Milvus
    create_collection()  # Create a collection in Milvus
    insert_embeddings()  # Insert article embeddings into Milvus
