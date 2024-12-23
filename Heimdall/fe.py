import warnings
from abc import ABC, abstractmethod
from typing import Sequence

import anndata as ad
import awkward as ak
import numpy as np
import torch
from anndata._warnings import ExperimentalFeatureWarning
from numpy.typing import NDArray
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from scipy.sparse import csc_array, csr_array, issparse

from Heimdall.utils import searchsorted2d


class Fe(ABC):
    """Abstraction for expression-based embedding.

    Args:
        adata: input AnnData-formatted dataset, with gene names in the `.var` dataframe.
        d_embedding: dimensionality of embedding for each expression entity

    """

    def __init__(
        self,
        adata: ad.AnnData,
        vocab_size: int,
        embedding_parameters: DictConfig,
        d_embedding: int,
        pad_value: int = None,
        mask_value: int = None,
    ):
        self.adata = adata
        self.num_cells, self.num_genes = adata.shape
        self.embedding_parameters = OmegaConf.to_container(embedding_parameters, resolve=True)
        self.d_embedding = d_embedding
        self.vocab_size = vocab_size
        self.pad_value = vocab_size - 2 if pad_value is None else pad_value
        self.mask_value = vocab_size - 1 if mask_value is None else mask_value

    @abstractmethod
    def preprocess_embeddings(self, float_dtype: str = "float32"):
        """Preprocess expression embeddings and store them for use during model
        inference.

        Preprocessing may include anything from downloading gene embeddings from
        a URL to generating embeddings from scratch.

        Returns:
            Sets `self.expression_embeddings`.
            Sets the following fields of `self.adata`:
            `.obsm['processed_expression_values']` : :class:`~numpy.ndarray` (shape `(self.adata.n_obs, -1)`)
                Processed expression values, for later use in calculation of expression-based embeddings.

        """

    def __getitem__(self, cell_indices: Sequence[int]) -> NDArray:
        """Get the indices of genes in the expression embedding array.

        Args:
            cell_indices: cells for which to retrieve expression embedding indices, as stored in `self.adata`.

        Returns:
            Index of value in the expression embeddings, or `pd.NA` if the gene has no mapping.

        """

        subset = self.adata[cell_indices]
        expression_values = subset.obsm["processed_expression_values"]
        expression_indices = subset.obsm["processed_expression_indices"]
        # breakpoint()
        return expression_values, expression_indices

    def load_from_cache(
        self,
        processed_expression_values: NDArray,
        processed_expression_indices: NDArray,
        expression_embeddings: NDArray | None,
    ):
        """Load processed values from cache."""
        # TODO: add tests
        self.adata.obsm["processed_expression_values"] = processed_expression_values
        self.adata.obsm["processed_expression_indices"] = processed_expression_indices
        self.expression_embeddings = expression_embeddings
        self.prepare_embedding_parameters()

    def prepare_embedding_parameters(self):
        """Replace config placeholders with values after preprocessing."""
        args = self.embedding_parameters.get("args", {})
        for key, value in args.items():
            if value == "max_seq_length":
                value = len(self.adata.var)
            elif value == "vocab_size":
                value = self.vocab_size  # <PAD> and <MASK> TODO: data.vocab_size
            elif value == "expression_embeddings":
                expression_embeddings = torch.tensor(self.expression_embeddings)  # TODO: type is inherited from NDArray
                pad_vector = torch.zeros(1, self.d_embedding)
                mask_vector = torch.zeros(1, self.d_embedding)
                value = torch.cat((expression_embeddings, pad_vector, mask_vector), dim=0)
            else:
                continue

            self.embedding_parameters["args"][key] = value


class BinningFe(Fe):
    """Value-binning Fe from scGPT.

    Args:
        adata: input AnnData-formatted dataset, with gene names in the `.var` dataframe.
        d_embedding: dimensionality of embedding for each expression entity
        embedding_parameters: dimensionality of embedding for each expression entity
        num_bins: number of bins to generate

    """

    def __init__(
        self,
        adata: ad.AnnData,
        num_bins: int,
        **fe_kwargs,
    ):
        fe_kwargs.pop("vocab_size", None)
        vocab_size = num_bins + 2  # Accounting for mask and pad tokens
        super().__init__(adata, vocab_size=vocab_size, **fe_kwargs)
        self.num_bins = num_bins

    def preprocess_embeddings(self, float_dtype: str = "float32"):
        """Compute bin identities of expression profiles in raw data."""
        self.expression_embeddings = None

        expression = self.adata.X
        csr_expression = csr_array(expression)
        cellwise_nonzero_expression = np.split(csr_expression.data, csr_expression.indptr[1:-1])

        # Obtain the indices of non-zero entries for each row (cell)
        nonzero_indices = np.split(csr_expression.indices, csr_expression.indptr[1:-1])
        nonzero_indices = ak.Array(nonzero_indices)

        n_bins = self.num_bins
        if np.max(expression) == 0:
            binned_values = csr_array(expression.shape).astype(
                float_dtype,
            )  # TODO: add correct typing (maybe add to config...?)

        quantiles = np.linspace(0, 1, n_bins)
        bin_edges = ak.Array(
            [np.quantile(nonzero_expression, quantiles) for nonzero_expression in cellwise_nonzero_expression],
        )  # First axis is quantiles, second is cells

        binned_values = searchsorted2d(
            bin_edges,
            cellwise_nonzero_expression,
            side="left",
        )
        binned_values = binned_values + 1

        self.adata.obsm["processed_expression_values"] = binned_values
        self.adata.obsm["processed_expression_indices"] = nonzero_indices

        self.prepare_embedding_parameters()


class NonzeroIdentityFe(Fe):
    """Directly pass the continuous values. Remove zeros.

    Args:
        adata: input AnnData-formatted dataset, with gene names in the `.var` dataframe.
        d_embedding: dimensionality of embedding for each expression entity
        embedding_parameters: dimensionality of embedding for each expression entity
        num_bins: number of bins to generate

    """

    def preprocess_embeddings(self, float_dtype: str = "float32"):
        self.expression_embeddings = None

        expression = self.adata.X
        csr_expression = csr_array(expression)
        cellwise_nonzero_expression = ak.Array(np.split(csr_expression.data, csr_expression.indptr[1:-1]))
        cellwise_nonzero_indices = ak.Array(np.split(csr_expression.indices, csr_expression.indptr[1:-1]))

        self.adata.obsm["processed_expression_values"] = cellwise_nonzero_expression.astype(float_dtype)
        self.adata.obsm["processed_expression_indices"] = cellwise_nonzero_indices

        self.prepare_embedding_parameters()


class DummyFe(Fe):
    """Directly pass the continuous values. Does not remove zero expression
    elements.

    Args:
        adata: input AnnData-formatted dataset, with gene names in the `.var` dataframe.
        d_embedding: dimensionality of embedding for each expression entity
        embedding_parameters: dimensionality of embedding for each expression entity

    """

    def preprocess_embeddings(self, float_dtype: str = "float32"):
        self.expression_embeddings = None

        expression = self.adata.X.toarray() if issparse(self.adata.X) else self.adata.X

        self.adata.obsm["processed_expression_values"] = expression.astype(float_dtype)
        self.adata.obsm["processed_expression_indices"] = np.tile(np.arange(self.num_genes), (self.num_cells, 1))

        self.prepare_embedding_parameters()


class SortingFe(Fe):
    """Sorting Fe."""

    def preprocess_embeddings(self, float_dtype: str = "float32"):
        """Sort genes by expression per cell.

        Uses median normalization before sorting (Geneformer style).

        """
        warnings.filterwarnings("ignore", category=ExperimentalFeatureWarning)  # Ignore warnings for Awkward Arrays
        self.expression_embeddings = None

        expression = self.adata.X
        csc_expression = csc_array(expression)
        genewise_nonzero_expression = np.split(csc_expression.data, csc_expression.indptr[1:-1])

        gene_medians = np.array([np.median(gene_nonzeros) for gene_nonzeros in genewise_nonzero_expression])

        normalized_expression = csr_array(expression)
        normalized_expression = csr_array(normalized_expression / gene_medians)

        cellwise_nonzero_expression = np.split(normalized_expression.data, normalized_expression.indptr[1:-1])
        cellwise_nonzero_indices = np.split(normalized_expression.indices, normalized_expression.indptr[1:-1])

        argsorted_expression = ak.argsort(cellwise_nonzero_expression, axis=1)[:, ::-1]

        processed_expression_values = ak.Array(
            [
                cell_nonzero_indices[processed_cell]
                for cell_nonzero_indices, processed_cell in zip(cellwise_nonzero_indices, argsorted_expression)
            ],
        )

        self.adata.obsm["processed_expression_values"] = ak.values_astype(processed_expression_values, float_dtype)
        self.adata.obsm["processed_expression_indices"] = ak.values_astype(
            processed_expression_values,
            np.int64,
        )  # both are the same in this case
        self.prepare_embedding_parameters()
