import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):  # , dropout: float = 0.1
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = torch.einsum("sbe->bse", pe)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Forward function.

        Args:
            x: Tensor, shape ``[batch_size , seq_len, embedding_dim]``

        """
        # x = x + self.pe[:x.size(0)]
        x = x + self.pe[:, : x.size(1)]  # Broadcasting to match input shape
        x = self.dropout(x)
        return x


class FlexibleTypeLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dtype = self.weight.dtype

    def forward(self, inputs: torch.Tensor):
        return super().forward(inputs.type(self.dtype).unsqueeze(-1))


class FlexibleTypeEmbedding(nn.Embedding):
    def forward(self, idx: torch.Tensor):
        return super().forward(idx.type(torch.long))


# https://github.com/bowang-lab/scGPT/blob/7301b51a72f5db321fccebb51bc4dd1380d99023/scgpt/model/model.py#L795
class ScGPTCategoryValueEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.long()
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class FlexibleTypeEmbeddingAndProjection(nn.Module):
    def __init__(
        self,
        embeddings: Tensor,
        d_model: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings)
        self.linear = nn.Linear(embeddings.shape[1], d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = x.long()
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.linear(x)
        return x


class TwoLayerNN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.nonlinear = nn.ReLU()
        self.linear2 = nn.Linear(out_features, out_features)
        self.dtype = self.linear1.weight.dtype

    def forward(self, inputs: torch.Tensor):
        return self.linear2(self.nonlinear(self.linear1(inputs.type(self.dtype).unsqueeze(-1))))
