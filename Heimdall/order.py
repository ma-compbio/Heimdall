from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from Heimdall.fc import Fc


class Order(ABC):
    def __init__(
        self,
        fc: Fc,
    ):
        self.fc = fc

    @abstractmethod
    def __call__(
        self,
        identity_inputs: NDArray,
        expression_inputs: NDArray,
        **kwargs,
    ) -> NDArray:
        """Order cell tokens using metadata.

        Gene tokens can be reordered based on e.g. expression level, chromosome position, etc.

        Args:
            cell_tokenization: the stacked gene identity- and gene expression-based tokenization
                of a cell.

        """


class ExpressionOrder(Order):


    def __call__(self, identity_inputs: NDArray, expression_inputs: NDArray, **kwargs,) -> NDArray:
        """Order cell tokens using metadata.

        Gene tokens are reordered based on expression level.

        Args:
            cell_tokenization: the stacked gene identity- and gene expression-based tokenization
                of a cell.

        """
        cell_index = kwargs.get("cell_index", None)
        if cell_index is None:
            raise ValueError("ExpressionOrder needs `cell_index` in **kwargs`.")
        
        cols = np.asarray(identity_inputs, dtype=int)

        n_vars = self.fc.adata.shape[1]
        if cols.size == 0 or cols.min() < 0 or cols.max() >= n_vars:
            alt = kwargs.get("identity_indices", None)
            if alt is None:
                raise ValueError(
                    "identity_inputs do not appear to be var indices. "
                    "Either ensure Fg=identity (tokens == var positions) or pass `identity_indices`."
                )
            cols = np.asarray(alt, dtype=int)

        X = self.fc.adata.X
        sub = X.getrow(cell_index)[:, cols] if hasattr(X, "getrow") else X[cell_index, cols]
        

        if hasattr(sub, "toarray"): 
            x = sub.toarray().ravel()
        elif hasattr(sub, "A"):
            x = np.asarray(sub.A).ravel()
        else:
            x = np.asarray(sub).ravel()
        
        dtype = np.dtype(getattr(self.fc, "float_dtype", "float32"))
        x = x.astype(dtype, copy=False)

        if "medians" in self.fc.adata.var:
            med = self.fc.adata.var["medians"].to_numpy()[cols].astype(dtype, copy=False)
            x = x - med

        # Sort non-zero values in descending order
        x = np.where(np.isnan(x), -np.inf, x)
        gene_order = np.argsort(x)[::-1]  # Indices for sorting descending
        return gene_order


class RandomOrder(Order):
    def __call__(self, identity_inputs: NDArray, expression_inputs: NDArray, **kwargs,) -> NDArray:
        # TODO: consider cleaning up sampling (just sample all nonzero and all zero, then concat
        (nonzero_indices,) = np.where(expression_inputs != 0)
        (zero_indices,) = np.where(expression_inputs == 0)

        # First: sample/reorder nonzero expression tokens
        # num_nonzero_to_sample = min(len(nonzero_indices), self.fc.max_input_length)
        num_nonzero = len(nonzero_indices)
        num_zero = len(zero_indices)

        # selected_nonzero = self.fc.rng.choice(nonzero_indices, num_nonzero_to_sample, replace=False)
        selected_nonzero = self.fc.rng.choice(nonzero_indices, num_nonzero, replace=False)

        # If needed: sample zero-expression tokens to fill up
        # num_remaining = self.fc.max_input_length - num_nonzero_to_sample
        # if num_remaining > 0:
        if num_zero > 0:
            selected_zero = self.fc.rng.choice(zero_indices, num_zero, replace=False)
            gene_order = np.concatenate([selected_nonzero, selected_zero])
        else:
            gene_order = selected_nonzero

        # Optionally shuffle to avoid position bias, but we dont need to because the gene ids are the position
        # self.rng.shuffle(final_indices)

        return gene_order


class ChromosomeOrder(Order):
    def __call__(self, identity_inputs: NDArray, expression_inputs: NDArray, **kwargs,) -> NDArray:
        """Order cell tokens using metadata.

        Gene tokens are reordered based on chromosome location.

        Args:
            cell_tokenization: the stacked gene identity- and gene expression-based tokenization
                of a cell.

        """

        choosen_chrom = self.fc.chroms.iloc[identity_inputs]

        unique_chromosomes = np.unique(choosen_chrom)
        self.fc.shuffled_chromosomes = self.fc.rng.permutation(unique_chromosomes)

        gene_order = np.zeros(len(identity_inputs), dtype=np.int32)
        for chromosome in self.fc.shuffled_chromosomes:
            (chromosome_index,) = np.where(choosen_chrom == chromosome)
            sort_by_start = np.argsort(
                self.fc.starts[chromosome_index],
            )  # start chromosome_indexations for this chromsome

            gene_order[chromosome_index] = chromosome_index[sort_by_start]

        return gene_order
