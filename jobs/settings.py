from pydantic_settings import BaseSettings
from omegaconf import OmegaConf
from icecream import ic

class ScrapeSettings(BaseSettings):
    books_dir: str
    timeout: int


class ProcessSettings(BaseSettings):
    books_dir: str
    min_word_count: int
    vocab_dir: str
    vocab_file_name: str
    corpus_dir: str
    corpus_file_name: str


class TrainSettings(BaseSettings):
    class _Trainer(BaseSettings):
        max_epochs: int

    class _MlflowLogger(BaseSettings):
        experiment_name: str
        tracking_uri: str
        log_model: str

    embedding_size: int
    vocab_path: str
    corpus_path: str
    trainer: _Trainer
    mlflow_logger: _MlflowLogger


class JobSettings(BaseSettings):
    scrape: ScrapeSettings
    process: ProcessSettings
    train: TrainSettings

raw_conf = OmegaConf.load("jobs/config.yaml")
settings = JobSettings(**raw_conf)
