import httpx
import asyncio

from tqdm import tqdm
from pathlib import Path
from bs4 import BeautifulSoup


class GutenbergSpanishScraper:
    results_url = "https://www.gutenberg.org/ebooks/results/?author=&title=&subject=&lang=es&category=&locc=&filetype=txt.utf-8&submit_search=Search&pageno="
    file_base_url = "https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"

    def __init__(self, data_dir: str, timeout: int = 60):
        self.data_dir = Path(data_dir)
        self.book_ids = list()
        self.timeout = timeout

        self.data_dir.mkdir(parents=True, exist_ok=True)

    async def scrape_book_ids(self):
        target_urls = [f"{self.results_url}{page}" for page in range(1, 10)]

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            with tqdm(
                total=len(target_urls), desc="Scraping Spanish book IDs"
            ) as progress_bar:
                tasks = [
                    self._scrape_book_id(client, url, progress_bar)
                    for url in target_urls
                ]
                await asyncio.gather(*tasks)

    async def _scrape_book_id(self, client, url, progress_bar):
        res = await client.get(url)
        soup = BeautifulSoup(res.content, "html.parser")
        table = soup.select_one("#content table")
        rows = table.find_all("tr")[1:]
        for row in rows:
            cols = row.find_all("td")
            if cols:
                book_id = cols[0].text.strip()
                self.book_ids.append(book_id)
        progress_bar.update(1)

    async def download_books(self):
        if len(self.book_ids) == 0:
            raise ValueError(
                "No book ids to download, did you call `scrape_book_ids` first?"
            )

        async with httpx.AsyncClient(
            follow_redirects=True, timeout=self.timeout
        ) as client:
            with tqdm(
                total=len(self.book_ids), desc="Downloading Spanish books"
            ) as progress_bar:
                tasks = [
                    self._download_book(client, book_id, progress_bar)
                    for book_id in self.book_ids
                ]
                await asyncio.gather(*tasks)

    async def _download_book(self, client, book_id, progress_bar):
        target_url = self.file_base_url.format(book_id=book_id)
        target_file = self.data_dir / f"{book_id}.txt"

        res = await client.get(target_url)
        with open(target_file, "wb") as f:
            f.write(res.content)
        progress_bar.update(1)
