import pickle

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


class Word2VecDataset(Dataset):
    def __init__(self, corpus_file: str, window_size: int = 2):
        with open(corpus_file, "rb") as f:
            self.words = pickle.load(f)

        self.window_size = window_size
        self.context_target_pairs = self.generate_pairs()

    def generate_pairs(self):
        context_target_pairs = []

        # Skip the first and last "window_size" words to ensure consistent context size
        for i in tqdm(range(self.window_size, len(self.words) - self.window_size)):
            target_word = self.words[i]
            context_words = (
                self.words[i - self.window_size : i]
                + self.words[i + 1 : i + self.window_size + 1]
            )
            for context_word in context_words:
                context_target_pairs.append((context_word, target_word))
        return context_target_pairs

    def __len__(self):
        return len(self.context_target_pairs)

    def __getitem__(self, idx):
        context, target = self.context_target_pairs[idx]
        return context, target


class Word2VecDataModule(LightningDataModule):
    def __init__(self, corpus_file: str, window_size: int = 2, batch_size: int = 64):
        super().__init__()
        self.corpus_file = corpus_file
        self.window_size = window_size
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = Word2VecDataset(self.corpus_file, self.window_size)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
