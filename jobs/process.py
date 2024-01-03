import pickle

from tqdm import tqdm
from pathlib import Path
from collections import Counter
from word2vec.process import GutenbergTextProcessor

processor = GutenbergTextProcessor()
books_dir = Path("data/books")

all_tokens = list()

for book in tqdm(books_dir.glob("*.txt"), desc="processing books"):
    tokens_without_stopwords = processor.process_text(book)
    all_tokens.extend(tokens_without_stopwords)

word_counts = Counter(all_tokens)
threshold = 900
filtered_tokens = [token for token in all_tokens if word_counts[token] >= threshold]


word_counts = Counter(filtered_tokens)
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
indexed_tokens = [word_to_idx[token] for token in filtered_tokens]

vocab_dir = Path("data/vocab")
vocab_dir.mkdir(parents=True, exist_ok=True)
with open(vocab_dir / "spanish-vocab.pkl", "wb") as f:
    pickle.dump(word_to_idx, f)

corpus_dir = Path("data/corpus")
corpus_dir.mkdir(parents=True, exist_ok=True)
with open(corpus_dir / "spanish-corpus.pkl", "wb") as f:
    pickle.dump(indexed_tokens, f)
