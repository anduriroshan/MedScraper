import mysql.connector
import configparser
from transformers import pipeline  # Hugging Face pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from src.logger import logging
from concurrent.futures import ThreadPoolExecutor

# Load Configurations
config = configparser.ConfigParser()
config.read('config/config.ini')

MYSQL_CONFIG = {
    "host": config["MYSQL"]["host"],
    "user": config["MYSQL"]["user"],
    "password": config["MYSQL"]["password"],
    "database": config["MYSQL"]["database"]
}

# Use a smaller, faster model for CPU
summarizer = pipeline("summarization", model="t5-small", device=-1)  # Use CPU

def extract_keywords_tfidf(text, num_keywords=5):
    """
    Extract top keywords using TF-IDF.
    :param text: The text to analyze.
    :param num_keywords: Number of keywords to extract.
    :return: A comma-separated string of keywords.
    """
    vectorizer = TfidfVectorizer(stop_words="english", max_features=num_keywords)
    tfidf_matrix = vectorizer.fit_transform([text])
    keywords = vectorizer.get_feature_names_out()
    return ", ".join(keywords)


def summarize_texts_in_batch(texts, batch_size=10):
    """
    Summarize a list of texts in batches using Hugging Face Transformers with dynamic length calculation.
    :param texts: List of texts to summarize.
    :param batch_size: Number of texts to process in a batch.
    :return: List of summaries.
    """
    summaries = []
    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Dynamically calculate max_length and min_length for each text
            max_lengths = [max(20, int(len(text.split()) * 0.5)) for text in batch]
            min_lengths = [max(10, int(len(text.split()) * 0.3)) for text in batch]

            # Summarize each batch
            results = []
            for idx, text in enumerate(batch):
                max_len = max_lengths[idx]
                min_len = min_lengths[idx]
                result = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
                results.append(result[0]["summary_text"])

            summaries.extend(results)
            print(f"Processed {len(summaries)} articles so far...")
    except Exception as e:
        logging.error(f"Error during batch summarization: {e}")
        summaries.extend(["Error"] * len(texts))  # Placeholder for failed batches
    return summaries


def parallel_summarize(texts, batch_size=10):
    """
    Parallelize text summarization using ThreadPoolExecutor.
    """
    summaries = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_summary = [executor.submit(summarize_texts_in_batch, texts[i:i + batch_size])
                             for i in range(0, len(texts), batch_size)]
        for future in future_to_summary:
            summaries.extend(future.result())
    return summaries


def summarize_articles():
    """
    Summarize article abstracts and extract keywords in batches using Hugging Face and TF-IDF.
    """
    connection = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = connection.cursor()

    # Add summary and keywords columns if they don't exist
    try:
        cursor.execute("""
            ALTER TABLE articles 
            ADD COLUMN IF NOT EXISTS summary TEXT;
        """)
        cursor.execute("""
            ALTER TABLE articles 
            ADD COLUMN IF NOT EXISTS keywords TEXT;
        """)
    except Exception as e:
        logging.error(f"Error altering table: {e}")

    # Fetch article abstracts
    cursor.execute("SELECT id, abstract FROM articles")
    articles = cursor.fetchall()

    # Process abstracts in batches
    batch_size = 10  # Adjust for your hardware
    article_ids, abstracts = zip(*articles)

    # Print the number of articles being processed
    print(f"Processing {len(articles)} articles...")

    summaries = parallel_summarize(abstracts, batch_size=10)

    # Extract keywords and update database
    for idx, (article_id, abstract) in enumerate(zip(article_ids, abstracts)):
        try:
            keywords = extract_keywords_tfidf(abstract)
            cursor.execute("UPDATE articles SET summary=%s, keywords=%s WHERE id=%s",
                           (summaries[idx], keywords, article_id))
            logging.info(f"Processed article ID: {article_id}")
            print(f"Processed article ID: {article_id}")  # Print for debugging
        except Exception as e:
            logging.error(f"Error processing article ID {article_id}: {e}")
            print(f"Error processing article ID {article_id}")  # Print for debugging

    connection.commit()
    cursor.close()
    connection.close()
    logging.info("Summaries and keywords added to MySQL.")
    print("Summaries and keywords added to MySQL.")

if __name__ == "__main__":
    summarize_articles()
