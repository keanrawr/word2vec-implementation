import httpx
import pickle
import asyncio

from tqdm import tqdm
from pathlib import Path


class GutenbergSpanishScraper:
    file_base_url = "https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"

    def __init__(self, data_dir: str, timeout: int = 60):
        self.data_dir = Path(data_dir)
        self.timeout = timeout

        self.data_dir.mkdir(parents=True, exist_ok=True)

        book_ids_path = Path(__file__).parent / "spanish_books_ids.pkl"
        with open(book_ids_path, "rb") as f:
            self.book_ids = pickle.load(f)

        existing_books = set(item.stem for item in self.data_dir.iterdir() if item.is_file())
        remaining_books = set(self.book_ids) - existing_books
        self.remaining_books = list(remaining_books)

    async def download_books(self):
        if len(self.remaining_books) == 0:
            raise ValueError(
                "No book ids to download, all of them are downloaded or something went wrong"
            )

        async with httpx.AsyncClient(
            follow_redirects=True, timeout=self.timeout
        ) as client:
            with tqdm(
                total=len(self.remaining_books), desc="Downloading Spanish books"
            ) as progress_bar:
                tasks = [
                    self._download_book(client, book_id, progress_bar)
                    for book_id in self.remaining_books
                ]
                await asyncio.gather(*tasks)

    async def _download_book(self, client, book_id, progress_bar):
        target_url = self.file_base_url.format(book_id=book_id)
        target_file = self.data_dir / f"{book_id}.txt"

        res = await client.get(target_url)
        with open(target_file, "wb") as f:
            f.write(res.content)
        progress_bar.update(1)
