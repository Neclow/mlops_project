from lightning.pytorch import LightningModule

import torch

from torch import nn, optim
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class MLP(nn.Module):
    def __init__(self, emb_dim, out_dim):
        super().__init__()

        self.emb_dim = emb_dim

        self.out_dim = out_dim

        self.classifier = nn.Linear(self.emb_dim, self.out_dim)

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # Feature extraction: B x T* --> B x N x D

        # 2D: B x D --> B X C
        # 3D: B x N x D --> B x N x C
        x = self.classifier(x)

        # 2D: B x C --> B x C (logits)
        # 3D: B x N x C --> B x N x C (logits)
        x = self.log_softmax(x)

        if x.ndim == 3:
            # B x N x C --> B x C
            x = x.mean(1)

        return x


class LightningMLP(LightningModule):
    def __init__(self, feature_extractor, num_classes, loss_fn, lr, weight_decay):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.feature_extractor.eval()

        self.classifier = MLP(
            emb_dim=self.feature_extractor.emb_dim, out_dim=num_classes
        )
        self.classifier.train()

        self.loss_fn = loss_fn

        metric = MulticlassAccuracy(num_classes=num_classes, average="micro")

        # TODO: add more metrics
        # metric = MetricCollection(
        #     [
        #         MulticlassAccuracy(num_classes=num_classes, average="micro"),
        #         MulticlassF1Score(num_classes=num_classes, average="micro"),
        #         MulticlassF1Score(num_classes=num_classes, average="macro"),
        #     ]
        # )

        self.train_metric = metric

        self.valid_metric = metric.clone()

        self.test_metric = metric.clone()

        self.lr = lr

        self.weight_decay = weight_decay

        self.save_hyperparameters(ignore=["feature_extractor"])

    def training_step(self, *args, **kwargs):
        batch = args[0]

        loss, score = self._get_loss_and_scores(batch, self.train_metric)

        self.log_dict(
            {
                "train_loss": loss,
                "train_metric": score,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, *args, **kwargs):
        batch = args[0]

        loss, score = self._get_loss_and_scores(batch, self.valid_metric)

        self.log_dict(
            {
                "valid_loss": loss,
                "valid_metric": score,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def test_step(self, *args, **kwargs):
        batch = args[0]

        _, score = self._get_loss_and_scores(batch, self.test_metric)

        self.log("test_metric", score, on_step=False, on_epoch=True)

    def _get_loss_and_scores(self, batch, metric):
        self.feature_extractor.eval()
        # Forward pass
        X_batch, y_batch = batch["input"].data, batch["label"]

        with torch.no_grad():
            emb_batch = self.feature_extractor(X_batch)

        o_batch = self.classifier(emb_batch)

        # Compute losses
        loss = self.loss_fn(o_batch, y_batch)

        # Update metrics
        score = metric(o_batch, y_batch)

        return loss, score

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        return optimizer
