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


class GaussianInitEmbedding(FlexibleTypeEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.normal_(self.weight, mean=0.0, std=1.0)


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


# https://github.com/biomap-research/scFoundation/blob/main/model/pretrainmodels/mae_autobin.py#L18-L77
class AutoDiscretizationEmbedding2(nn.Module):
    def __init__(self, dim, bin_num, bin_alpha, mask_token_id=None, pad_token_id=None):
        super().__init__()

        self.dim = dim
        self.bin_num = bin_num
        self.bin_alpha = bin_alpha

        self.mlp = nn.Linear(1, self.bin_num)
        self.mlp2 = nn.Linear(self.bin_num, self.bin_num)
        self.LeakyReLU = nn.LeakyReLU(0.1)
        self.Softmax = nn.Softmax(dim=-1)
        self.emb = nn.Embedding(self.bin_num, self.dim)

        self.emb_mask = nn.Embedding(1, self.dim)
        self.emb_pad = nn.Embedding(1, self.dim)

        self.bin_num_idx = torch.tensor(range(self.bin_num))
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        # print('self.bin_num_idx',self.bin_num_idx, self.bin_num_idx.shape)

        self.tensor0 = torch.tensor(0, dtype=torch.long)

    def forward(self, x, output_weight=False):
        x = x.unsqueeze(-1)
        # x_mask_idx = (x==self.mask_token_id).nonzero()
        # x_pad_idx = (x==self.pad_token_id).nonzero()
        # print("x_mask",x_mask_idx.shape,x_mask_idx)

        x = self.mlp(x)  # [B,N,1] -> [B,N,H]
        x = self.LeakyReLU(x)  # [B,N,H]
        x_crosslayer = self.mlp2(x)  # [B,N,H]
        x = self.bin_alpha * x + x_crosslayer  # [B,N,H]
        weight = self.Softmax(x)  # [B, N, H]
        # print('weight', weight.shape, weight, torch.sum(weight, 2))

        bin_num_idx = self.bin_num_idx.to(x.device)  # [H,]
        # print('bin_num_idx', bin_num_idx.shape)

        token_emb = self.emb(bin_num_idx)  # [H, D]
        # print('token_emb', token_emb.shape)
        x = torch.matmul(weight, token_emb)  # [B, N, D]

        # print("x_emb",x.shape,x)

        # tensor0 = torch.tensor(0, dtype=torch.long, device=x.device)

        # mask_token_emb = self.emb_mask(tensor0).to(x.device).type(x.dtype)
        # print(mask_token_emb.dtype)
        # print("x", x.dtype)
        # x[x_mask_idx[:,0],x_mask_idx[:,1],:] = mask_token_emb.repeat(x_mask_idx.shape[0],1)
        # print("x_emb",x.shape,x)

        # pad_token_emb = self.emb_pad(tensor0).to(x.device).type(x.dtype)
        # x[x_pad_idx[:,0],x_pad_idx[:,1],:] = pad_token_emb.repeat(x_pad_idx.shape[0],1)

        return x
