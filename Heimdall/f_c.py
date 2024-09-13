from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence, Union

import anndata as ad
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import Module
from tqdm import tqdm

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
        # We then map these names back to indices in the gene embeddings. Finally, we pass it through the embedding layer to retrieve the embeddings.
        valid_mask = self.adata.var["identity_valid_mask"]
        valid_genes = self.adata.var_names[valid_mask].values

        cell_expression_embedding_indices = self.fe[:]

        try:
            gene_lists = valid_genes[cell_expression_embedding_indices]
        except IndexError:
            raise IndexError(
                "It seems like you are using an Fe that indexes more outputs than are available in the Fg embedding layer, which is not compatible with Geneformer. Please use a valid combination of `Fe` and `Fg`.",
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


def old_geneformer_fc(fg, adata):
    """geneformer_fc is a fc that will reprocess each cell by ordering them by
    their gene expression value, and replace each gene name by their
    corresponding representation, either token_id or a different vector.

    Reprocesses each cell by ordering them by their gene expression value, and
    replace each gene name by their corresponding representation, either
    ``token_id`` or a different vector.

    Note:
        Currently, this only supports ``token_id``.

    Args:
        fg: dictionary that maps gene names to token ids.
        adata: the whole, already processed, anndata object with the CellxGene
            matrix.

    Return:
        A numpy object that is dimension CellxGene where the position has the
        token denoting what gene it is.

    """

    valid_mask = adata.var["identity_valid_mask"]
    valid_genes = adata.var_names[valid_mask].values

    expression = adata.X[:, valid_mask]
    gene_medians = np.median(expression, axis=0)
    normalized_expression = expression / gene_medians

    argsorted_expression = np.argsort(normalized_expression, axis=1)[:, ::-1]

    gene_lists = valid_genes[argsorted_expression]
    dataset = np.array([fg[gene_list] for gene_list in gene_lists])

    return dataset


def geneformer_fc(fg, adata, embedding_layer=None):
    """geneformer_fc is a fc that will reprocess each cell by ordering them by
    their normalized gene expression value, and replace each gene name by their
    corresponding representation, either token_id or a different vector.

    right now this only supports token_id

    args:
        - fg: dictionary that maps gene names to token ids
        - adata: the whole, already processed, anndata object with the CellxGene Matrix

    output:
        - output: dataset, a numpy object that is dimension CellxGene where the position has the token denoting what
          gene it is

    """
    assert all(isinstance(value, (int)) for value in fg.values()), "Current geneformer_fc only supports token ids"

    print("> Performing the f_c using rank-based values, as seen in geneformer")

    # normalize by gene medians
    df = (
        pd.DataFrame(adata.X.toarray(), columns=fg.keys())
        if hasattr(adata.X, "toarray")
        else pd.DataFrame(adata.X, columns=fg.keys())
    )
    gene_medians = df.median()
    normalized_df = df.apply(lambda x: x / gene_medians[x.name])

    dataset = []
    for i in tqdm(range(len(normalized_df))):
        cell = normalized_df.iloc[i]
        sorted_cell = cell.sort_values(ascending=False).index
        # Use token ids only
        cell_w_gene_ids = [fg[gene] for gene in sorted_cell]

        dataset.append(cell_w_gene_ids)

    dataset = np.array(dataset)
    return dataset


def scgpt_fc(fg, adata, embedding_layer=None, num_bins=10):
    """scgpt_fc reprocesses each cell by binning genes based on expression
    values and replacing each gene name with their corresponding token_id.

    args:
        - fg: dictionary that maps gene names to token ids
        - adata: the whole, already processed, anndata object with the CellxGene Matrix
        - num_bins: number of bins for value binning

    output:
        - dataset: a numpy object that is dimension CellxGene where the position has the token denoting what gene it is
        - binned_values_dataset: a numpy object of binned expression values

    """
    # assert all(isinstance(value, int) for value in fg.values()), \
    #         "Current scgpt_fc only supports token ids"

    print("> Performing the f_c using rank-based values with binning, as seen in scGPT")
    df = (
        pd.DataFrame(adata.X.toarray(), columns=fg.keys())
        if hasattr(adata.X, "toarray")
        else pd.DataFrame(adata.X, columns=fg.keys())
    )
    df = df[df.columns.intersection(fg.keys())]
    binned_values_dataset = []

    for i in tqdm(range(len(df))):
        cell = df.iloc[i]
        # apply quantile-based binning to the expression values
        binned_values = value_binning(cell.values, n_bins=num_bins)

        binned_values_dataset.append(binned_values)

    binned_values_dataset = np.array(binned_values_dataset)
    return binned_values_dataset
