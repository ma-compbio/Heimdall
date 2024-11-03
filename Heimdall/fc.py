from abc import ABC, abstractmethod
from typing import Optional

import anndata as ad
import awkward as ak
import numpy as np
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import Module

from Heimdall.fe import Fe
from Heimdall.fg import Fg


class Fc(ABC):
    """Abstraction for cell embedding.

    Args:
        fg: `Fg` used for this `Fc` implementation.
        fe: `Fe` used for this `Fe` implementation.
        adata: input AnnData-formatted dataset, with gene names in the `.var` dataframe.
        max_input_length: maximum number of identity/expression tokens to consider for each cell.
            Extra tokens are limited.

    """

    def __init__(
        self,
        fg: Fg | None,
        fe: Fe | None,
        adata: ad.AnnData,
        max_input_length: Optional[int] = None,
    ):
        self.fg = fg
        self.fe = fe
        self.adata = adata
        self.max_input_length = max_input_length

    @abstractmethod
    def preprocess_cells(self):
        """Using the `fg` and `fe`, preprocess input cells, retrieve indices of
        both gene and expression embeddings.

        This function can be deterministic, or may involve random sampling.

        Returns:
            Sets the following fields of `self.adata`:
            `.obsm['cell_identity_embedding_indices']` : :class:`~numpy.ndarray`
                (shape `(self.adata.n_obs, self.max_input_length)`)
                Gene identity embedding indices for all cells.
            `.obsm['cell_expression_embedding_indices']` : :class:`~numpy.ndarray`
                (shape `(self.adata.n_obs, self.max_input_length)`)
                Gene expression embedding indices for all cells.

        """

    def __getitem__(self, cell_index: int) -> tuple[NDArray, NDArray, NDArray]:
        """Retrieve `cell_identity_embedding_indices`,
        `cell_expression_embedding_indices` and `padding_mask`.

        Can only be called after running `self.preprocess_cells()`.

        Returns:
            A tuple of gene identity embedding indices and gene expression embedding indices for all cells.

        """

        identity_inputs = self.adata.obsm["cell_identity_embedding_indices"][cell_index]
        expression_inputs = self.adata.obsm["cell_expression_embedding_indices"][cell_index]

        # Padding and truncating
        identity_inputs = self.tailor(identity_inputs, self.fg.pad_value)

        identity_inputs = ak.to_numpy(identity_inputs)

        expression_inputs = self.tailor(expression_inputs, self.fe.pad_value)
        expression_inputs = ak.to_numpy(expression_inputs)

        padding_mask = expression_inputs == self.fe.pad_value

        return identity_inputs, expression_inputs, padding_mask

    def pad(self, cell_tokenization: ak.Array, pad_value: int | float) -> ak.Array:
        return ak.fill_none(ak.pad_none(cell_tokenization, self.max_input_length, axis=0), pad_value)

    @abstractmethod
    def limit(self, cell_tokenization: ak.Array) -> ak.Array:
        """Ensure that none of the cell tokenizations exceed the maximum length.

        Args:
            cell_tokenization: a tokenization for each cell, represented as a ragged array.

        """

    def tailor(self, cell_tokenization: ak.Array, pad_value: int | float) -> ak.Array:
        if len(cell_tokenization) > self.max_input_length:
            return self.limit(cell_tokenization)

        return self.pad(cell_tokenization, pad_value)

    @abstractmethod
    def embed_cells(
        self,
        identity_inputs: Tensor,
        gene_embedding_layer: Module | None,
        expression_inputs: Tensor,
        expression_embedding_layer: Module | None,
    ) -> Tensor:
        """Embed cell batch using the embedding layers.

        It can be assumed that both the identity inputs and the expression inputs have been padded/
        limitd at this stage, i.e. they are square tensors.

        Args:
            identity_inputs: batched gene identity embedding indices
            gene_embedding_layer: Torch module for embedding based on gene identity.
            expression_inputs: batched gene expression embedding indices
            expression_embedding_layer: Torch module for embedding based on expression.

        Returns:
            Embeddings of cells.

        """

    def load_from_cache(
        self,
        cell_identity_embedding_indices: NDArray,
        cell_expression_embedding_indices: NDArray,
    ):
        """Load processed values from cache."""
        # TODO: add tests

        self.adata.obsm["cell_identity_embedding_indices"] = cell_identity_embedding_indices
        self.adata.obsm["cell_expression_embedding_indices"] = cell_expression_embedding_indices


class GeneformerFc(Fc):
    """Implementation of Geneformer cell embedding."""

    def preprocess_cells(self):
        # For Geneformer, we retrieve sorted indices based on expression. We then convert these to gene names.
        # We then map these names back to indices (or other identity values).
        valid_mask = self.adata.var["identity_valid_mask"]
        valid_genes = self.adata.var_names[valid_mask].values

        cell_expression_embedding_indices = self.fe[:]

        try:
            gene_lists = [
                valid_genes[expression_embedding_indices]
                for expression_embedding_indices in cell_expression_embedding_indices
            ]
        except IndexError:
            raise IndexError(
                "It seems like you are using an `Fe` that indexes more outputs than are available in the `Fg` embedding"
                "layer, which is not compatible with Geneformer. Please use a valid combination of `Fe` and `Fg`.",
            )

        cell_identity_embedding_indices = ak.Array([self.fg[gene_list] for gene_list in gene_lists])

        self.adata.obsm["cell_identity_embedding_indices"] = cell_identity_embedding_indices
        self.adata.obsm["cell_expression_embedding_indices"] = cell_expression_embedding_indices

    def embed_cells(
        self,
        identity_inputs: Tensor,
        gene_embedding_layer: Module | None,
        expression_inputs: Tensor,
        expression_embedding_layer: Module | None,
    ) -> Tensor:
        """Geneformer cell embedding function.

        Ignores expression embedding layer; uses indices based on indices into gene embedding.

        Args:
            gene_embedding_layer:  # TODO: fill out
            expression_embedding_layer: # TODO fill out

        """

        embeddings = gene_embedding_layer(identity_inputs)

        return embeddings

    def limit(self, cell_tokenization: ak.Array) -> ak.Array:
        return cell_tokenization[: self.max_input_length]


class ScGPTFc(Fc):
    """Implementation of scGPT cell embedding."""

    def __init__(
        self,
        fg: Fg | None,
        fe: Fe | None,
        adata: ad.AnnData,
        max_input_length: Optional[int] = None,
    ):
        super().__init__(fg, fe, adata, max_input_length)
        seed = 0  # TODO: make this configurable???
        self.rng = np.random.default_rng(seed)

    def preprocess_cells(self):
        valid_mask = self.adata.var["identity_valid_mask"]
        valid_genes = self.adata.var_names[valid_mask].values

        cell_identity_embedding_indices = np.array(
            [self.fg[valid_genes] for _ in range(len(self.adata))],
        )
        cell_expression_embedding_indices = self.fe[:]

        self.adata.obsm["cell_identity_embedding_indices"] = cell_identity_embedding_indices
        self.adata.obsm["cell_expression_embedding_indices"] = cell_expression_embedding_indices
        # self.adata.obsm["cell_expression_padding_mask"] = padding_mask

    def limit(self, cell_tokenization: ak.Array) -> ak.Array:
        return self.rng.choice(cell_tokenization, self.max_input_length)

    def embed_cells(
        self,
        identity_inputs: Tensor,
        gene_embedding_layer: Module | None,
        expression_inputs: Tensor,
        expression_embedding_layer: Module | None,
    ) -> Tensor:
        """ScGPT cell embedding callback.

        TODO: add "conditional tokens" (see Methods of https://www.nature.com/articles/s41592-024-02201-0#Sec14)

        Args:
            gene_embedding_layer:  # TODO: fill out
            expression_embedding_layer: # TODO fill out

        """

        gene_embeddings = gene_embedding_layer(identity_inputs)
        expression_embeddings = expression_embedding_layer(expression_inputs)

        return gene_embeddings + expression_embeddings


ScBERTFc = ScGPTFc  # TODO: is ScBERTFc actually the same as ScGPTFc?
