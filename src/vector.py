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
    connections.connect(**MILVUS_CONFIG)

def create_collection():
    """Create Milvus collection."""
    schema = CollectionSchema([
        FieldSchema("id", DataType.INT64, is_primary=True),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=384),
    ])
    collection = Collection(COLLECTION_NAME, schema)
    collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
    return collection

def insert_embeddings():
    """Fetch data from MySQL and insert embeddings into Milvus."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    connection = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = connection.cursor()
    cursor.execute("SELECT id, title FROM articles")
    records = cursor.fetchall()

    embeddings = [model.encode(title) for _, title in records]
    ids = [record[0] for record in records]

    collection = Collection(COLLECTION_NAME)
    collection.insert([ids, embeddings])
    logging.info("Embeddings inserted into Milvus.")

if __name__ == "__main__":
    connect_milvus()
    create_collection()
    insert_embeddings()
