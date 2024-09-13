from abc import ABC, abstractmethod
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from Heimdall.utils import searchsorted2d


class Fe(ABC):
    """Abstraction for expression-based embedding."""

    def __init__(self, adata: ad.AnnData, config: dict):
        self.adata = adata
        _, self.num_genes = adata.shape
        self.config = config

    @abstractmethod
    def preprocess_embeddings(self):
        """Preprocess expression embeddings and store them for use during model
        inference.

        Preprocessing may include anything from downloading gene embeddings from
        a URL to generating embeddings from scratch.

        """

    def __getitem__(self, cell_indices: Sequence[int]) -> NDArray:
        """Get the indices of genes in the expression embedding array.

        Args:
            cell_indices: cells for which to retrieve expression embedding indices, as stored in `self.adata`.

        Returns:
            Index of value in the expression embeddings, or `pd.NA` if the gene has no mapping.

        """
        embedding_indices = self.adata.obsm["processed_expression_values"][cell_indices]
        # if np.any(embedding_indices.isna()):
        #     raise KeyError(
        #         "At least one gene is not mapped in this Fe. Please remove such genes from consideration in the Fc.",
        #     )

        return embedding_indices


class DummyFe(Fe):
    """Dummy Fe for `fc`s that don't use expresssion-based embeddings."""

    def preprocess_embeddings(self):
        """Stand-in `fe` preprocessing; marks `expression_embedding_index` as
        invalid."""
        self.expression_embeddings = None
        dummy_indices = pd.array(np.full((len(self.adata), self.config.num_embeddings), np.nan))
        self.adata.obsm["processed_expression_values"] = dummy_indices


class BinningFe(Fe):
    """Value-binning Fe from scGPT."""

    def preprocess_embeddings(self):
        """Compute bin identities of expression profiles in raw data."""
        self.expression_embeddings = None

        valid_mask = self.adata.var["identity_valid_mask"]  # TODO: assumes that Fg is run first. Is that okay?
        expression = self.adata.X[:, valid_mask]

        n_bins = self.config.num_embeddings
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
