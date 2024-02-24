import pickle

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from lightning.pytorch.profilers import AdvancedProfiler
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

from jobs.settings import settings
from word2vec.models import SkipGramModel
from word2vec.dataset import Word2VecDataModule


with open(settings.train.vocab_path, "rb") as f:
    word_2_idx = pickle.load(f)

vocab_size = len(word_2_idx)
embedding_dim = settings.train.embedding_size
data_module = Word2VecDataModule(
    settings.train.corpus_path,
    batch_size=settings.train.batch_size,
    window_size=settings.train.window_size,
)
model = SkipGramModel(vocab_size, embedding_dim)
print(
    f"""Setting up model with:
    vocab size: {vocab_size:,}
    embedding dim: {embedding_dim}
"""
)

mlflow_logger = MLFlowLogger(**settings.train.mlflow_kwargs)
profiler = AdvancedProfiler(dirpath=".", filename="perf_logs_2")

trainer = Trainer(
    max_epochs=settings.train.trainer.max_epochs,
    logger=mlflow_logger,
    callbacks=[DeviceStatsMonitor(), ModelCheckpoint("model", monitor="train_loss")],
    profiler=profiler,
)
trainer.fit(model, data_module)
