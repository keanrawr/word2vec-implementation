import pickle

from tqdm import tqdm
from pathlib import Path
from collections import Counter
from word2vec.process import GutenbergTextProcessor

from jobs.settings import settings

processor = GutenbergTextProcessor()
books_dir = Path(settings.process.books_dir)

all_tokens = list()

for book in tqdm(books_dir.glob("*.txt"), desc="processing books"):
    tokens_without_stopwords = processor.process_text(book)
    all_tokens.extend(tokens_without_stopwords)

word_counts = Counter(all_tokens)
threshold = settings.process.min_word_count
filtered_tokens = [token for token in all_tokens if word_counts[token] >= threshold]


word_counts = Counter(filtered_tokens)
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
indexed_tokens = [word_to_idx[token] for token in filtered_tokens]

vocab_dir = Path(settings.process.vocab_dir)
vocab_dir.mkdir(parents=True, exist_ok=True)
vocab_file = vocab_dir / settings.process.vocab_file_name
with open(vocab_file, "wb") as f:
    pickle.dump(word_to_idx, f)

corpus_dir = Path(settings.process.corpus_dir)
corpus_dir.mkdir(parents=True, exist_ok=True)
corpus_file = corpus_dir / settings.process.corpus_file_name
with open(corpus_file, "wb") as f:
    pickle.dump(indexed_tokens, f)
