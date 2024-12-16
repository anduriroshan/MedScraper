# Article Search and Summarization with Milvus, MySQL, and Hugging Face

## Project Overview

This project is designed to crawl articles from Nature's Oncology section, process them by generating embeddings, and then allow the user to search for articles using natural language queries. The system integrates multiple technologies for article storage, vector search, summarization, and keyword extraction.

The key functionality of the project includes:
1. **Article Crawling**: Scraping articles from Nature's Oncology section.
2. **Article Storage**: Storing crawled articles in MySQL database.
3. **Text Summarization and Keyword Extraction**: Using Hugging Face transformers and TF-IDF to summarize articles and extract keywords.
4. **Vector Search**: Storing embeddings of article titles in Milvus and searching them based on user queries.
5. **Natural Language Query Processing**: Interpreting user queries for specific time frames (e.g., "last week", "2023") and returning relevant articles.

## Tech Stack

### 1. **Python**:
   - Python is the core language used in this project due to its simplicity and wide support for data manipulation, machine learning, and natural language processing.
   - Libraries such as `requests`, `BeautifulSoup`, and `mysql.connector` provide powerful functionalities for web scraping, database interaction, and more.

### 2. **Milvus**:
   - **Why Milvus?**: Milvus is a vector database optimized for similarity search and storage of high-dimensional data, which is essential for working with embeddings generated from text (such as article titles).
   - It is used in this project to store and search article embeddings (created using the SentenceTransformer model).
   - Milvus enables fast, scalable retrieval of semantically similar articles based on user queries.

### 3. **Sentence Transformers**:
   - **Why Sentence Transformers?**: Sentence Transformers is a library that allows the creation of high-quality sentence embeddings using pre-trained transformer models. This is used to convert article titles into embeddings, which are stored and searched within Milvus.
   - Models like `multi-qa-MiniLM-L6-cos-v1` and `all-MiniLM-L6-v2` provide good accuracy for semantic search tasks.

### 4. **MySQL**:
   - **Why MySQL?**: MySQL is used to store the crawled article data, including article titles, publication dates, and abstracts. It allows for relational data storage and retrieval.
   - The database also stores metadata such as summaries and extracted keywords, enabling easy query filtering and retrieval.

### 5. **Hugging Face Transformers**:
   - **Why Hugging Face Transformers?**: Hugging Face provides state-of-the-art models for text generation tasks such as summarization. Models like `t5-small` and `distilbart-cnn-12-6` are used to generate concise summaries of the article abstracts.
   - The `pipeline("summarization")` interface is used to quickly summarize articles with minimal setup, improving processing speed and performance.

### 6. **Spacy**:
   - **Why Spacy?**: Spacy is a popular NLP library used for advanced text processing tasks such as entity recognition and linguistic analysis. It is used in this project for parsing natural language queries and extracting date-related entities (like "last week", "2023 year").

### 7. **ThreadPoolExecutor** (for parallel processing):
   - **Why ThreadPoolExecutor?**: This Python class from the `concurrent.futures` module is used to parallelize text summarization tasks. By splitting the summarization of articles into batches and processing them concurrently, the performance of the system is improved, allowing it to handle large datasets more efficiently.

## Project Features

### 1. **Article Crawling**:
   - The `crawl_articles` function crawls articles from the "Oncology" section of Nature's website. Each article is stored in MySQL with its title, publication date, and abstract.

### 2. **Text Summarization**:
   - The `summarize_articles` function uses the Hugging Face `summarization` pipeline to summarize the abstracts of articles.
   - Summaries are stored back in the MySQL database, along with extracted keywords from the articles using TF-IDF.

### 3. **Vector Search**:
   - The `insert_embeddings` function generates embeddings for article titles using the SentenceTransformer model and inserts them into Milvus for efficient semantic search.
   - The `search_articles` function performs semantic search in Milvus using user queries, retrieving similar articles based on their embeddings and filtering results by publication date.

### 4. **Natural Language Query Processing**:
   - The `parse_advanced_date_from_query` function interprets natural language time-related expressions like "last week", "this month", or specific years like "2023".
   - The system allows the user to input queries like "Give me the journals published last week", and the relevant articles are fetched from MySQL and Milvus.

## Flow of Process

The following flow illustrates the sequence of events that occur from crawling articles to executing a search query:

1. **Crawling**: 
   - The system starts by crawling the "Oncology" section of Nature's website using the `crawl_articles` function. 
   - Articles are retrieved with their titles, publication dates, and abstracts.
   - The data is stored in the MySQL database for later use.

2. **Text Summarization and Keyword Extraction**: 
   - Once articles are stored, the `summarize_articles` function processes the abstracts of each article.
   - It uses the Hugging Face `summarization` model to create concise summaries of each article and extracts keywords using the TF-IDF method.
   - Summaries and keywords are then saved back into MySQL.

3. **Embedding Generation and Insertion**:
   - Using the `SentenceTransformer`, embeddings are generated for each article title.
   - These embeddings are inserted into the Milvus database for efficient vector-based search.

4. **Search Query**:
   - When a user submits a natural language query (e.g., "Give me the journals published last week"), the query is processed using Spacy to extract date-related information.
   - The query is also converted into an embedding using the `SentenceTransformer`, and Milvus is used to perform a semantic search on the article embeddings.
   - The search is filtered based on the parsed date range, and relevant articles are retrieved from MySQL.

5. **Displaying Results**: 
   - The relevant articles are displayed to the user, including their titles and publication dates.
   - The system then prompts the user if they wish to perform another search or exit the program.

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
