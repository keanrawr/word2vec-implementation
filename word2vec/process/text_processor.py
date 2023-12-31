import re
import nltk

from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class GutenbergTextProcessor:
    def __init__(self, nltk_dir: str = "data/nltk_data"):
        self.nltk_dir = Path(nltk_dir)
        self.nltk_dir.mkdir(parents=True, exist_ok=True)
        nltk.data.path.append(str(self.nltk_dir))

        nltk.download("punkt", download_dir=self.nltk_dir)
        nltk.download("stopwords", download_dir=self.nltk_dir)
        self.spanish_stopwords = set(stopwords.words("spanish"))

    def process_text(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        start_idx, end_idx = self.find_gutenberg_delimiters(text)
        main_text = (
            text[start_idx:end_idx] if start_idx != -1 and end_idx != -1 else text
        )

        cleaned_text = re.sub(r"[^a-zA-Z\s]", " ", main_text)
        tokens = word_tokenize(cleaned_text)
        normalized_tokens = [token.lower() for token in tokens]

        return [
            token for token in normalized_tokens if token not in self.spanish_stopwords
        ]

    def find_gutenberg_delimiters(self, text):
        # Function to find the start and end index of the main content in a Gutenberg text
        def find_index_start(text):
            intro_phrases = [
                "START OF THIS PROJECT GUTENBERG EBOOK",
                "START OF THE PROJECT GUTENBERG EBOOK",
                "START OF THE PROJECT GUTENBERG",
                "START OF PROJECT GUTENBERG",
                "THE PROJECT GUTENBERG EBOOK",
            ]
            for phrase in intro_phrases:
                index = text.find(phrase)
                if index != -1:
                    end_of_line_index = text.find("\n", index)
                    return end_of_line_index + 1 if end_of_line_index != -1 else index
            return -1

        def find_index_end(text):
            concluding_phrases = [
                "END OF THIS PROJECT GUTENBERG EBOOK",
                "END OF THE PROJECT GUTENBERG EBOOK",
                "END OF PROJECT GUTENBERG",
                "END OF THE PROJECT GUTENBERG",
            ]
            for phrase in concluding_phrases:
                index = text.find(phrase)
                if index != -1:
                    return index
            return -1

        return find_index_start(text), find_index_end(text)
