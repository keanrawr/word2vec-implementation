import torch
import torch.nn as nn

from pytorch_lightning import LightningModule

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)  # additional linear layer
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_words):
        embeds = self.embeddings(input_words)
        out = self.linear(embeds)
        return self.log_softmax(out)

class SkipGramModel(LightningModule):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.model = SkipGram(vocab_size, embedding_dim)
        self.loss_function = nn.NLLLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        context_indices, target_indices = batch
        loss = 0
        for context in context_indices:
            context = torch.tensor(context).to(self.device)
            log_probs = self(context)
            loss += self.loss_function(log_probs, target_indices)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())
