import httpx
import pickle
import asyncio

from tqdm import tqdm
from pathlib import Path


class GutenbergSpanishScraper:
    results_url = "https://www.gutenberg.org/ebooks/results/?author=&title=&subject=&lang=es&category=&locc=&filetype=txt.utf-8&submit_search=Search&pageno="
    file_base_url = "https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"

    def __init__(self, data_dir: str, timeout: int = 60):
        self.data_dir = Path(data_dir)
        self.timeout = timeout

        self.data_dir.mkdir(parents=True, exist_ok=True)

        book_ids_path = Path(__file__).parent / "spanish_books_ids.pkl"
        with open(book_ids_path, "rb") as f:
            self.book_ids = pickle.load(f)


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
