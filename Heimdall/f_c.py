from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence, Union

import anndata as ad
import numpy as np
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import Module

from Heimdall.f_g import Fg
from Heimdall.fe import Fe


@dataclass
class TransformerInput:
    gene_inputs: NDArray
    expression_inputs: NDArray

    def __post_init__(self):
        pass


class Fc(ABC):
    """Abstraction for cell embedding."""

    def __init__(self, fg: Fg | None, fe: Fe | None, adata: ad.AnnData, config: dict):
        self.fg = fg
        self.fe = fe
        self.adata = adata
        self.config = config

    @abstractmethod
    def preprocess_cells(self):
        """Using the `fg` and `fe`, preprocess input cells, retrieve indices of
        both gene and expression embeddings.

        This function can be deterministic, or may involve random sampling.

        """

    def __getitem__(self, cell_indices: Union[int, Sequence[int], slice]) -> tuple[NDArray]:
        """Retrieve `cell_identity_embedding_indices` and
        `cell_expression_embedding_indices`."""

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

        # TODO: make it accept a range of cell indices...? Would have to modify `preprocess_cells` accordingly.

        Args:
            gene_embedding_layer: Torch module for embedding based on gene identity.
            expression_embedding_layer: Torch module for embedding based on expression.

        Returns:
            Embedding of an individual cell.

        """


class GeneformerFc(Fc):
    """Implementation of Geneformer cell embedding."""

    def preprocess_cells(self):
        # For Geneformer, we retrieve sorted indices based on expression. We then convert these to gene names.
        # We then map these names back to indices in the gene embeddings. Finally, we pass it through the embedding
        # layer to retrieve the embeddings.
        valid_mask = self.adata.var["identity_valid_mask"]
        valid_genes = self.adata.var_names[valid_mask].values

        cell_expression_embedding_indices = self.fe[:]

        try:
            gene_lists = valid_genes[cell_expression_embedding_indices]
        except IndexError:
            raise IndexError(
                "It seems like you are using an Fe that indexes more outputs than are available in the Fg embedding"
                "layer, which is not compatible with Geneformer. Please use a valid combination of `Fe` and `Fg`.",
            )

        cell_identity_embedding_indices = np.array([self.fg[gene_list] for gene_list in gene_lists])

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
            [self.fg[valid_genes][: self.config.max_input_length] for _ in range(len(self.adata))],
        )
        cell_expression_embedding_indices = self.fe[:, : self.config.max_input_length]

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
