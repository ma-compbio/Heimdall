from abc import ABC, abstractmethod
from os import PathLike
from typing import Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from Heimdall.utils import searchsorted2d


class Fe(ABC):
    """Abstraction for expression-based embedding.

    Args:
        adata: input AnnData-formatted dataset, with gene names in the `.var` dataframe.
        d_embedding: dimensionality of embedding for each expression entity
        num_embeddings: number of embeddings to generate for expression-based embedding,
            e.g. how many bins for binning, etc.
        embedding_filepath: filepath from which to load pretrained embeddings

    """

    def __init__(
        self,
        adata: ad.AnnData,
        d_embedding: int,
        num_embeddings: Optional[int] = None,
        embedding_filepath: Optional[str | PathLike] = None,
    ):
        self.adata = adata
        _, self.num_genes = adata.shape
        self.d_embedding = d_embedding
        self.num_embeddings = num_embeddings
        self.embedding_filepath = embedding_filepath

    @abstractmethod
    def preprocess_embeddings(self):
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
        embedding_indices = self.adata.obsm["processed_expression_values"][cell_indices]

        return embedding_indices

    def load_from_cache(self, processed_expression_values: NDArray, expression_embeddings: NDArray | None):
        """Load processed values from cache."""
        # TODO: add tests
        self.adata.obsm["processed_expression_values"] = processed_expression_values
        self.expression_embeddings = expression_embeddings


class DummyFe(Fe):
    """Dummy Fe for `fc`s that don't use expresssion-based embeddings."""

    def preprocess_embeddings(self):
        """Stand-in `fe` preprocessing; marks `expression_embedding_index` as
        invalid."""
        self.expression_embeddings = None
        dummy_indices = pd.array(np.full((len(self.adata), self.num_embeddings), np.nan))
        self.adata.obsm["processed_expression_values"] = dummy_indices


class BinningFe(Fe):
    """Value-binning Fe from scGPT."""

    def preprocess_embeddings(self):
        """Compute bin identities of expression profiles in raw data."""
        self.expression_embeddings = None

        valid_mask = self.adata.var["identity_valid_mask"]  # TODO: assumes that Fg is run first. Is that okay?
        expression = self.adata.X[:, valid_mask]

        n_bins = self.num_embeddings
        if np.max(expression) == 0:
            binned_values = np.zeros_like(expression)  # TODO: add correct typing (maybe add to config...?)

        masked_expression = expression.astype(np.float64)
        masked_expression[masked_expression == 0] = np.nan
        bin_edges = np.nanquantile(masked_expression, np.linspace(0, 1, n_bins - 1), axis=1).T

        binned_values = searchsorted2d(
            bin_edges,
            expression,
            side="left",
        )  # TODO: now that we do binning per cell, how to efficiently vectorize digitization???
        binned_values[expression > 0] += 1

        self.adata.obsm["processed_expression_values"] = binned_values


class SortingFe(Fe):
    """Sorting Fe."""

    def preprocess_embeddings(self):
        """Sort genes by expression per cell.

        Uses median normalization before sorting (Geneformer style).

        """
        self.expression_embeddings = None

        valid_mask = self.adata.var["identity_valid_mask"]  # TODO: assumes that Fg is run first. Is that okay?
        expression = self.adata.X[:, valid_mask]
        gene_medians = np.median(expression, axis=0)
        normalized_expression = expression / gene_medians

        argsorted_expression = np.argsort(normalized_expression, axis=1)[:, ::-1]

        self.adata.obsm["processed_expression_values"] = argsorted_expression
