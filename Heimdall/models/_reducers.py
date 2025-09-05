from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class Reducer(nn.Module, ABC):
    """Reduce a list of `n` tensors into a single tensor.

    Each tensor in the list must have dimensionality `(batch_size, dim_in)`. The
    reduction may be symmetric or asymmetric.

    """

    def __init__(self, dim_in: int):
        super().__init__()
        self.dim_in = dim_in

    @abstractmethod
    def forward(self, tensors: list[Tensor]): ...


class SumReducer(Reducer):
    def forward(self, tensors: list[Tensor]):
        return torch.sum(torch.stack(tensors, axis=0), axis=0)


class MeanReducer(Reducer):
    def forward(self, tensors: list[Tensor]):
        return torch.mean(torch.stack(tensors, axis=0), axis=0)


class AsymmetricConcatReducer(Reducer):
    def __init__(self, dim_in: int):
        super().__init__(dim_in=dim_in)
        self.pair_embedder = nn.Linear(2 * dim_in, dim_in)

    def forward(self, tensors: list[Tensor]):
        concatenated = torch.cat(tensors, dim=-1)
        return self.pair_embedder(concatenated)


class SymmetricConcatReducer(Reducer):
    def __init__(self, dim_in: int):
        super().__init__(dim_in=dim_in)
        self.pair_embedder = nn.Linear(2 * dim_in, dim_in)

    def forward(self, tensors: list[Tensor]):
        concatenated_1 = torch.cat(tensors, dim=2)
        concatenated_2 = torch.cat(list(reversed(tensors)), dim=2)

        encoded = self.pair_embedder(concatenated_1) + self.pair_embedder(concatenated_2)
        return encoded
