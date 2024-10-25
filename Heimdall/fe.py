from abc import ABC, abstractmethod
from typing import Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from scipy.sparse import issparse
from tqdm import tqdm

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
        embedding_parameters: DictConfig,
        d_embedding: int,
    ):
        self.adata = adata
        _, self.num_genes = adata.shape
        self.embedding_parameters = OmegaConf.to_container(embedding_parameters, resolve=True)
        self.d_embedding = d_embedding

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
        padding_mask = self.adata.obsm["padding_mask"][cell_indices]

        return embedding_indices, padding_mask

    def load_from_cache(
        self,
        processed_expression_values: NDArray,
        expression_padding_mask: NDArray,
        expression_embeddings: NDArray | None,
    ):
        """Load processed values from cache."""
        # TODO: add tests
        self.adata.obsm["processed_expression_values"] = processed_expression_values
        self.adata.obsm["padding_mask"] = expression_padding_mask
        self.expression_embeddings = expression_embeddings
        self.replace_placeholders()

    def replace_placeholders(self):
        """Replace config placeholders with values after preprocessing."""
        for key, value in self.embedding_parameters["args"].items():
            if value == "max_seq_length":
                value = len(self.adata.var)
            elif value == "vocab_size":
                value = len(self.adata.var) + 2  # <PAD> and <MASK> TODO: data.vocab_size
            elif value == "expression_embeddings":
                value = self.expression_embeddings
            else:
                continue

            self.embedding_parameters["args"][key] = value


class DummyFe(Fe):
    """Dummy Fe for `fc`s that don't use expresssion-based embeddings."""

    def preprocess_embeddings(self):
        """Stand-in `fe` preprocessing; marks `expression_embedding_index` as
        invalid.

        TODO: maybe we should have the `None` value for `self.expression_embeddings` marked
        with some other value to indicate that no embedding layer should be instantiated

        """
        self.expression_embeddings = None
        dummy_indices = pd.array(np.full((len(self.adata), self.num_embeddings), np.nan))
        self.adata.obsm["processed_expression_values"] = dummy_indices

        self.replace_placeholders()


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
        embedding_parameters: OmegaConf,
        d_embedding: int,
        num_bins: Optional[int],
    ):
        super().__init__(adata, embedding_parameters, d_embedding)
        self.num_bins = num_bins

    def preprocess_embeddings(self):
        """Compute bin identities of expression profiles in raw data."""
        self.expression_embeddings = None

        valid_mask = self.adata.var["identity_valid_mask"]  # TODO: assumes that Fg is run first. Is that okay?
        expression = self.adata.X[:, valid_mask]

        n_bins = self.num_bins
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

        self.replace_placeholders()


class SortingFe(Fe):
    """Sorting Fe."""

    ## TODO: make it such that we mask out values with zero expression, because right now we don't
    ## I think I can make a function in the Fe for it.
    def preprocess_embeddings(self):
        """Sort genes by expression per cell.

        Uses median normalization before sorting (Geneformer style).

        """
        self.expression_embeddings = None

        valid_mask = self.adata.var["identity_valid_mask"]  # TODO: assumes that Fg is run first. Is that okay?
        self.adata = self.adata[:, valid_mask]

        nonzero_mask = self.adata.layers["nonzero_mask"].toarray()

        if issparse(self.adata.X):
            expression = self.adata.X.toarray()  ## not needed if it is scaled
        else:
            expression = self.adata.X

        # expression = self.adata.X[:, valid_mask].toarray()

        # gene_medians = np.median(expression, axis=0)
        # normalized_expression = expression / gene_medians

        ## Taking the nonzero medians
        gene_medians = np.array([np.median(gene[gene != 0]) for gene in expression.T])
        normalized_expression = expression / gene_medians
        argsorted_expression = np.argsort(normalized_expression, axis=1)[:, ::-1]

        # breakpoint()

        # a = nonzero_mask
        # b = argsorted_expression
        # pmask = np.vstack([a[i, b[i]] for i in range(a.shape[0])]) == False

        padding_mask = (
            np.vstack([nonzero_mask[i, argsorted_expression[i]] for i in range(nonzero_mask.shape[0])]) == False
        )

        # breakpoint()

        # # Iterate over each cell (row)
        # padding_mask = np.zeros_like(argsorted_expression)
        # for i in tqdm(range(nonzero_mask.shape[0])):
        #     # Get indices where nonzero_mask is False for the current row
        #     zero_indices = np.where(nonzero_mask[i] == False)[0]
        #     mask = np.isin(argsorted_expression[i], zero_indices)
        #     argsorted_expression[i][mask] = -1
        #     padding_mask[i] = mask  ## constructing the padding mask

        ## obsm is tied to the rows, but not to the columns
        self.adata.obsm["processed_expression_values"] = argsorted_expression
        self.adata.obsm["padding_mask"] = padding_mask.astype(bool)

        # breakpoint()
        self.replace_placeholders()
