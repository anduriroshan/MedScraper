import requests
from bs4 import BeautifulSoup
import mysql.connector
import datetime
from src.logger import logging

BASE_URL = "https://www.nature.com/search"
PARAMS = {
    "subject": "oncology",
    "article_type": "protocols,research,reviews",
    "page": 1
}

# Database Configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",  # Replace with your MySQL password
    "database": "nature_articles"
}

def fetch_article_links():
    """Fetch article links from the main search page."""
    page = 1
    article_links = []

    while page < 2:
        logging.info(f"Scraping search page {page}...")
        PARAMS["page"] = page
        response = requests.get(BASE_URL, params=PARAMS)
        response.encoding = 'utf-8'

        if response.status_code != 200:
            logging.info("Failed to fetch page. Stopping.")
            break

        soup = BeautifulSoup(response.text, "html.parser")
        article_items = soup.find_all("li", class_="app-article-list-row__item")

        if not article_items:
            logging.info("No more articles found. Stopping.")
            break

        for article in article_items:
            # Extract article link
            title_tag = article.find("h3", class_="c-card__title").find("a", href=True)
            if title_tag and title_tag["href"]:
                article_links.append("https://www.nature.com" + title_tag["href"])

        page += 1

    return article_links


def fetch_article_details(article_url):
    """Fetch article details (title, authors, date, abstract, and keywords) from the article page."""
    response = requests.get(article_url)
    response.encoding = 'utf-8'
    if response.status_code != 200:
        logging.info(f"Failed to fetch article page: {article_url}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract the article title
    title_tag = soup.find("h1", class_="c-article-title")
    title = title_tag.text.strip() if title_tag else "No Title"

    # Extract the authors
    authors = []
    author_tags = soup.find_all("li", class_="c-article-author-list__item")
    for author in author_tags:
        name_tag = author.find("a", attrs={"data-test": "author-name"})
        if name_tag:
            authors.append(name_tag.text.strip())

    # Extract the publication date
    date = "No Date"
    identifiers = soup.find("ul", class_="c-article-identifiers")
    if identifiers:
        for item in identifiers.find_all("li", class_="c-article-identifiers__item"):
            time_tag = item.find("time")
            if time_tag:
                date = time_tag.text.strip()
                break

    # Extract the abstract
    abstract = "No Abstract"
    abstract_header = soup.find("h2", {"id": "Abs1"})
    if abstract_header:
        abstract_section = abstract_header.find_next_sibling("div", class_="c-article-section__content")
        if abstract_section:
            abstract = abstract_section.text.strip()

    # Extract the keywords
    keywords = []
    keywords_header = soup.find("h3", class_="c-article__sub-heading", string="Keywords")
    if keywords_header:
        keyword_list = keywords_header.find_next_sibling("ul", class_="c-article-subject-list")
        if keyword_list:
            keywords = [kw.text.strip() for kw in keyword_list.find_all("li")]

    # Format date
    try:
        pub_date = datetime.datetime.strptime(date, "%d %B %Y").date()
    except ValueError:
        pub_date = None

    return {
        "title": title,
        "authors": ", ".join(authors),
        "pub_date": pub_date,
        "abstract": abstract,
        "keywords": ", ".join(keywords)
    }


def insert_into_mysql(data):
    """Insert article data into MySQL."""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()

        insert_query = """
            INSERT INTO articles (title, authors, pub_date, abstract, keywords)
            VALUES (%s, %s, %s, %s, %s)
        """

        for article in data:
            cursor.execute(insert_query, (
                article["title"],
                article["authors"],
                article["pub_date"],
                article["abstract"],
                article["keywords"]
            ))

        connection.commit()
        logging.info("Data successfully inserted into MySQL.")

    except mysql.connector.Error as err:
        logging.error(f"Error: {err}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


if __name__ == "__main__":
    # Step 1: Get all article links from the search pages
    article_links = fetch_article_links()
    logging.info(f"Found {len(article_links)} articles.\n")

    # Step 2: Visit each article page and extract details
    articles_data = []
    for idx, url in enumerate(article_links):
        logging.info(f"Fetching details for article {idx + 1}...")
        details = fetch_article_details(url)
        if details:
            articles_data.append(details)

    # Step 3: Insert data into MySQL
    insert_into_mysql(articles_data)
    logging.info("Process completed successfully.")
