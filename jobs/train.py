import pickle

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from lightning.pytorch.profilers import AdvancedProfiler
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

from word2vec.models import SkipGramModel
from word2vec.dataset import Word2VecDataModule


with open("data/vocab/spanish-vocab.pkl", "rb") as f:
    word_2_idx = pickle.load(f)

vocab_size = len(word_2_idx)
embedding_dim = 100
data_module = Word2VecDataModule(
    "data/corpus/spanish-corpus.pkl", batch_size=1_000, window_size=2
)
model = SkipGramModel(vocab_size, embedding_dim)
print(
    f"""Setting up model with:
    vocab size: {vocab_size:,}
    embedding dim: {embedding_dim}
"""
)

mlflow_logger = MLFlowLogger(
    experiment_name="pytorch-word2vec",
    tracking_uri="http://localhost:5000",
    log_model="all",
)
profiler = AdvancedProfiler(dirpath=".", filename="perf_logs_2")

trainer = Trainer(
    max_epochs=1,
    logger=mlflow_logger,
    callbacks=[DeviceStatsMonitor(), ModelCheckpoint("model", monitor="train_loss")],
    profiler=profiler,
)
trainer.fit(model, data_module)
