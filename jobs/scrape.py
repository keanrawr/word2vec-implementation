import asyncio
from word2vec.scraper import GutenbergSpanishScraper

from jobs.settings import settings


scraper = GutenbergSpanishScraper(
    data_dir=settings.scrape.books_dir,
    timeout=settings.scrape.timeout
)
async def run_scraper():
    await scraper.download_books()

asyncio.run(run_scraper())
