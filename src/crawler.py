import requests
from bs4 import BeautifulSoup
import mysql.connector
import configparser
import datetime
from src.logger import logging  # Import logging from the custom logger module

# Load Configurations
config = configparser.ConfigParser()
config.read('config\config.ini')

# MySQL Connection Configuration
MYSQL_CONFIG = {
    "host": config["MYSQL"]["host"],
    "user": config["MYSQL"]["user"],
    "password": config["MYSQL"]["password"],
    "database": config["MYSQL"]["database"]
}


def connect_mysql():
    """
    Establish a connection to MySQL database.

    Input: None
    Output: MySQL connection object

    This function connects to the MySQL database using credentials from the config file. If the connection fails,
    it logs the error and raises an exception.
    """
    try:
        connection = mysql.connector.connect(**MYSQL_CONFIG)
        logging.info("Successfully connected to MySQL.")
        return connection
    except mysql.connector.Error as e:
        logging.error(f"Error connecting to MySQL: {e}")
        raise


def crawl_articles(pages=5):
    """
    Crawl article details (title, publication date, abstract) from Nature's Oncology section.

    Input:
        - pages (int): The number of pages to scrape (default is 5).

    Output:
        - articles (list): A list of dictionaries, where each dictionary contains article details like title,
          publication date, and abstract.

    This function fetches article titles, publication dates, and abstracts by crawling through multiple pages of
    Nature's Oncology section based on the given number of pages. The articles are parsed from the HTML and
    stored in a list.
    """
    BASE_URL = config["SCRAPER"]["BASE_URL"]
    subject = config["SCRAPER"]["subject"]
    article_type = config["SCRAPER"]["article_type"]

    articles = []
    for page in range(1, int(pages) + 1):
        params = {
            "subject": subject,
            "article_type": article_type,
            "page": page
        }

        logging.info(f"Crawling page {page}...")
        response = requests.get(BASE_URL, params=params)

        if response.status_code != 200:
            logging.error(f"Failed to fetch page {page}, status code: {response.status_code}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        article_items = soup.find_all("li", class_="app-article-list-row__item")

        for item in article_items:
            try:
                title_tag = item.find("h3", class_="c-card__title")
                title = title_tag.get_text(strip=True) if title_tag else "No Title"
                link = "https://www.nature.com" + item.find("a")["href"]
                pub_date = item.find("time")["datetime"] if item.find("time") else "No Date"
                abstract = fetch_abstract(link)

                articles.append({
                    "title": title,
                    "pub_date": pub_date,
                    "abstract": abstract
                })
                logging.info(f"Fetched article: {title}")
            except Exception as e:
                logging.error(f"Error parsing article on page {page}: {e}")

    return articles


def fetch_abstract(url):
    """
    Fetch the abstract of an article from its detail page.

    Input:
        - url (str): The URL of the article page.

    Output:
        - abstract (str): The abstract of the article or "No Abstract" if not found.

    This function fetches the abstract by making an HTTP request to the article's URL, parses the page's HTML,
    and extracts the abstract text. If the abstract is not found, it returns "No Abstract".
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        abstract_section = soup.find("div", class_="c-article-section__content")
        return abstract_section.get_text(strip=True) if abstract_section else "No Abstract"
    except Exception as e:
        logging.error(f"Error fetching abstract for {url}: {e}")
        return "No Abstract"


def save_to_mysql(articles):
    """
    Save the crawled articles into the MySQL database.

    Input:
        - articles (list): A list of dictionaries containing article details (title, pub_date, abstract).

    Output:
        - None

    This function saves the article data into the MySQL database. It first checks if the 'articles' table exists,
    and creates it if necessary. Then, it inserts each article into the table.
    """
    try:
        connection = connect_mysql()
        cursor = connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INT AUTO_INCREMENT PRIMARY KEY,
                title VARCHAR(500),
                pub_date DATE,
                abstract TEXT
            )
        """)
        logging.info("Verified or created 'articles' table in MySQL.")

        insert_query = "INSERT INTO articles (title, pub_date, abstract) VALUES (%s, %s, %s)"
        for article in articles:
            cursor.execute(insert_query, (article["title"], article["pub_date"], article["abstract"]))

        connection.commit()
        logging.info(f"Successfully inserted {len(articles)} articles into MySQL.")
    except mysql.connector.Error as e:
        logging.error(f"Error saving articles to MySQL: {e}")
    finally:
        if connection:
            connection.close()
            logging.info("MySQL connection closed.")


if __name__ == "__main__":
    try:
        # Start logging for the process
        logging.info("Starting article crawling process...")

        # Load the number of pages to scrape from the configuration
        pages_to_scrape = int(config["SCRAPER"]["PAGES"])
        logging.info(f"Configured to crawl {pages_to_scrape} pages.")

        # Step 1: Crawl articles
        articles = crawl_articles(pages=pages_to_scrape)
        logging.info(f"Total articles fetched: {len(articles)}")

        # Step 2: Save articles to MySQL
        if articles:
            save_to_mysql(articles)
            logging.info("All articles have been successfully saved to the MySQL database.")
        else:
            logging.warning("No articles were fetched during the crawling process.")

        # End of process
        logging.info("Article crawling process completed successfully.")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
