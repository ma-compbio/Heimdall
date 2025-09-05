from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn


@dataclass
class TransformerOutput:
    logits: torch.Tensor
    # predictions: torch.Tensor
    sequence_embeddings: torch.Tensor
    # pooled_embeddings: torch.Tensor
    cls_embeddings: torch.Tensor

    @property
    def device(self):
        return self.logits.device

    def to(self, device):
        for key, val in self.__dict__.items():
            self.__dict__[key] = val.to(device)

    @classmethod
    def reduce(cls, outputs: list["TransformerOutput"], reduction: Callable = torch.sum):
        keys = cls.__dict__["__annotations__"].keys()
        reduced_output = TransformerOutput(
            **{
                key: reduction(
                    torch.stack([getattr(output, key) for output in outputs], axis=0),
                    axis=0,
                )
                for key in keys
            },
        )
        return reduced_output

    def __post_init__(self):
        # ensure output tensors are in float32 format
        for k, v in self.__dict__.items():
            setattr(self, k, v.float())


class CellPredHeadMixin:
    def forward(self, encoder_output) -> TransformerOutput:
        cls_emb = encoder_output[:, 0, :]
        logits = self.decoder(cls_emb.unsqueeze(1)).squeeze(1)
        return TransformerOutput(
            logits=logits,
            sequence_embeddings=encoder_output,
            cls_embeddings=cls_emb,
        )


class SeqPredHeadMixin:
    def forward(self, encoder_output) -> TransformerOutput:
        cls_emb = encoder_output[:, 0, :]
        logits = self.decoder(encoder_output[:, 1:, :])
        return TransformerOutput(
            logits=logits,
            sequence_embeddings=encoder_output,
            cls_embeddings=cls_emb,
        )


class ExpressionPredHeadMixin:
    def forward(self, encoder_output) -> TransformerOutput:
        logits = self.decoder(encoder_output)
        logits = logits.squeeze(1)
        return TransformerOutput(
            logits=logits,
            sequence_embeddings=logits,
            cls_embeddings=logits,
        )


class SeqHeadPredHeadMixin:
    def forward(self, encoder_output) -> TransformerOutput:
        logits = self.decoder(encoder_output)
        logits = logits.squeeze(1)
        return TransformerOutput(
            logits=logits,
            sequence_embeddings=logits,
            cls_embeddings=logits,
        )


class LinearDecoderMixin(nn.Module):
    def __init__(self, dim_in: int, dim_out: Optional[int] = None, dropout: float = 0.0, **kwargs):
        super().__init__()
        if dim_out is None:
            dim_out = dim_in
        self.decoder = nn.Sequential(
            nn.Linear(dim_in, dim_out, **kwargs),
            nn.Dropout(dropout),
        )


class LinearCellPredHead(CellPredHeadMixin, LinearDecoderMixin):
    """Linear cell prediction head."""


class ExpressionOnlyCellPredHead(ExpressionPredHeadMixin, LinearDecoderMixin):
    """Logistic regression prediction head.

    Put expression be the input

    """


class LinearSeqPredHead(SeqPredHeadMixin, LinearDecoderMixin):
    """Linear sequence prediction head."""
