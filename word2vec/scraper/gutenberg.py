import requests

from tqdm import tqdm
from pathlib import Path
from bs4 import BeautifulSoup


class GutenbergSpanishScraper:
    """
    Scraper class to download Spanish books that are available in the
    Gutenberg project's page
    """
    results_url = "https://www.gutenberg.org/ebooks/results/?author=&title=&subject=&lang=es&category=&locc=&filetype=txt.utf-8&submit_search=Search&pageno="
    file_base_url = "https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.book_ids = list()

        self.data_dir.mkdir(parents=True, exist_ok=True)

    def scrape_book_ids(self) -> None:
        target_urls = [f"{self.results_url}{page}" for page in range(1, 10)]

        for url in tqdm(target_urls, desc="Scraping Spanish book IDs"):
            self._scrape_book_id(url)

    def _scrape_book_id(self, url) -> str:
        res = requests.get(url)
        soup = BeautifulSoup(res.content, "html.parser")
        table = soup.select_one("#content table")
        rows = table.find_all('tr')[1:]
        for row in rows:
            cols = row.find_all('td')
            if cols:
                book_id = cols[0].text.strip()
                self.book_ids.append(book_id)

    def download_books(self):
        if len(self.book_ids) == 0:
            raise ValueError("No book id's to download, did you call `scrape_book_ids` first?")
        
        for book_id in tqdm(self.book_ids, desc="Downloading Spanish books"):
            target_url = self.file_base_url.format(book_id=book_id)
            target_file = self.data_dir / f"{book_id}.txt"

            res = requests.get(target_url)
            with open(target_file, "wb") as f:
                f.write(res.content)
