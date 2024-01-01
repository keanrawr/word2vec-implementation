import torch
import torch.nn as nn

from icecream import ic
from pytorch_lightning import LightningModule


class SkipGramModel(LightningModule):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Embedding(vocab_size, embedding_dim),
            nn.Linear(embedding_dim, vocab_size),
            nn.LogSoftmax(dim=1),
        )
        self.loss_function = nn.NLLLoss()

    def training_step(self, batch, batch_idx):
        context_indices, target_indices = batch

        target_indices = target_indices.view(-1)

        log_probs = self.net(target_indices)
        loss = self.loss_function(log_probs, context_indices)

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
