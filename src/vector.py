from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from sentence_transformers import SentenceTransformer
import mysql.connector

# MySQL Configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",  # Replace with your MySQL username
    "password": "root",  # Replace with your MySQL password
    "database": "nature_articles"  # Replace with your database name
}

# Step 1: Connect to MySQL and Fetch Titles
def fetch_titles_from_mysql():
    """Fetch journal titles from MySQL database."""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()
        query = "SELECT title FROM articles"  # Replace 'articles' with your table name
        cursor.execute(query)
        titles = [row[0] for row in cursor.fetchall()]  # Fetch all titles
        print(f"Fetched {len(titles)} titles from MySQL.")
        return titles

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return []

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

# Step 2: Connect to Milvus
def connect_to_milvus():
    """Connect to Milvus server."""
    connections.connect(host="localhost", port="19530")
    print("Connected to Milvus!")

# Step 3: Create Collection in Milvus
def create_collection(collection_name):
    """Define the schema and create a collection in Milvus."""
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),  # Vector dimension
    ]
    schema = CollectionSchema(fields, description="Journal Title Embeddings")
    collection = Collection(name=collection_name, schema=schema)
    print(f"Collection '{collection_name}' created.")
    return collection

# Step 4: Insert Data into Milvus
def insert_titles_to_milvus(collection, titles):
    """Generate embeddings for journal titles and insert into Milvus."""
    model = SentenceTransformer("all-MiniLM-L6-v2")  # Pre-trained Sentence-BERT model
    embeddings = [model.encode(title) for title in titles]  # Generate embeddings

    # Prepare data for Milvus
    entities = [
        titles,  # Journal Titles
        embeddings  # Corresponding Embeddings
    ]

    # Insert data into Milvus
    collection.insert([None, entities[0], entities[1]])
    print(f"Inserted {len(titles)} titles into Milvus.")

# Step 5: Main Execution
if __name__ == "__main__":
    # Connect to Milvus
    connect_to_milvus()

    # Define collection name
    collection_name = "nature_articles"

    # Create a Milvus collection
    collection = create_collection(collection_name)

    # Fetch titles from MySQL
    titles = fetch_titles_from_mysql()

    # Insert titles into Milvus
    if titles:
        insert_titles_to_milvus(collection, titles)
    else:
        print("No titles to insert into Milvus.")
