import asyncio
from word2vec.scraper import GutenbergSpanishScraper

scraper = GutenbergSpanishScraper("data/books", timeout=120)
async def run_scraper():
    await scraper.scrape_book_ids()
    await scraper.download_books()

asyncio.run(run_scraper())
