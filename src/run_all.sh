#!/bin/bash
echo "Running crawler..."
python crawler_application.py

echo "Running summarization..."
python summarization.py

echo "Inserting embeddings into Milvus..."
python vector_milvus.py

echo "Running query search..."
python query_search.py
