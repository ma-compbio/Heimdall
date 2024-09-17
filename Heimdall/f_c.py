from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union

import anndata as ad
import numpy as np
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import Module

from Heimdall.f_g import Fg
from Heimdall.fe import Fe


class Fc(ABC):
    """Abstraction for cell embedding.

    Args:
        fg: `Fg` used for this `Fc` implementation.
        fe: `Fe` used for this `Fe` implementation.
        adata: input AnnData-formatted dataset, with gene names in the `.var` dataframe.
        max_input_length: maximum number of identity/expression tokens to consider for each cell.
            Extra tokens are truncated.

    """

    def __init__(self, fg: Fg | None, fe: Fe | None, adata: ad.AnnData, max_input_length: Optional[int] = None):
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

    def __getitem__(self, cell_indices: Union[int, Sequence[int], slice]) -> tuple[NDArray, NDArray]:
        """Retrieve `cell_identity_embedding_indices` and
        `cell_expression_embedding_indices`.

        Can only be called after running `self.preprocess_cells()`.

        Returns:
            A tuple of gene identity embedding indices and gene expression embedding indices for all cells.

        """

        identity_inputs = self.adata.obsm["cell_identity_embedding_indices"][cell_indices].copy()
        expression_inputs = self.adata.obsm["cell_expression_embedding_indices"][cell_indices].copy()

        return identity_inputs, expression_inputs

    @abstractmethod
    def embed_cells(
        self,
        identity_inputs: Tensor,
        gene_embedding_layer: Module | None,
        expression_inputs: Tensor,
        expression_embedding_layer: Module | None,
    ) -> Tensor:
        """Embed cells using the embedding layers.

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
        cell_expression_embedding_indices: NDArray | None,
    ):
        """Load processed values from cache."""
        # TODO: add tests

        self.adata.obsm["cell_identity_embedding_indices"] = cell_identity_embedding_indices
        self.adata.obsm["cell_expression_embedding_indices"] = cell_expression_embedding_indices


class GeneformerFc(Fc):
    """Implementation of Geneformer cell embedding."""

    def preprocess_cells(self):
        # For Geneformer, we retrieve sorted indices based on expression. We then convert these to gene names.
        # We then map these names back to indices in the gene embeddings. Finally, we pass it through the embedding
        # layer to retrieve the embeddings.
        valid_mask = self.adata.var["identity_valid_mask"]
        valid_genes = self.adata.var_names[valid_mask].values

        cell_expression_embedding_indices = self.fe[:, : self.max_input_length]

        try:
            gene_lists = valid_genes[cell_expression_embedding_indices]
        except IndexError:
            raise IndexError(
                "It seems like you are using an Fe that indexes more outputs than are available in the Fg embedding"
                "layer, which is not compatible with Geneformer. Please use a valid combination of `Fe` and `Fg`.",
            )

        cell_identity_embedding_indices = np.array(
            [self.fg[gene_list][: self.max_input_length] for gene_list in gene_lists],
        )

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


class ScGPTFc(Fc):
    """Implementation of scGPT cell embedding."""

    def preprocess_cells(self):
        valid_mask = self.adata.var["identity_valid_mask"]
        valid_genes = self.adata.var_names[valid_mask].values

        cell_identity_embedding_indices = np.array(
            [self.fg[valid_genes][: self.max_input_length] for _ in range(len(self.adata))],
        )
        cell_expression_embedding_indices = self.fe[:, : self.max_input_length]

        self.adata.obsm["cell_identity_embedding_indices"] = cell_identity_embedding_indices
        self.adata.obsm["cell_expression_embedding_indices"] = cell_expression_embedding_indices

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
