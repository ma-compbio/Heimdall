from abc import ABC, abstractmethod
from typing import Optional

import anndata as ad
import awkward as ak
import numpy as np
import torch
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
        float_dtype: str = "float32",
    ):
        self.fg = fg
        self.fe = fe
        self.adata = adata
        self.max_input_length = max_input_length
        self.float_dtype = float_dtype

    def __getitem__(self, cell_index: int) -> tuple[NDArray, NDArray, NDArray]:
        """Retrieve `identity_inputs`, `expression_inputs` and `padding_mask`.

        Returns:
            A tuple of gene identity embedding indices and gene expression embedding indices for all cells.

        """
        identity_indices, expression_inputs = self.fe[cell_index]

        gene_list = self.adata.var_names[identity_indices]  # convert to ENSEMBL Gene Names
        identity_inputs = self.fg[gene_list]  # convert the genes into fg

        if len(identity_inputs) != len(expression_inputs):
            raise ValueError(
                "Gene identity and expression inputs do not have the same shape; `Fg` and `Fe` are incompatible.",
            )

        # Padding and truncating
        # breakpoint()
        identity_inputs, expression_inputs = self.tailor(
            identity_inputs,
            expression_inputs,
        )

        padding_mask = expression_inputs == self.fe.pad_value
        return identity_inputs, expression_inputs, padding_mask

    def pad(self, cell_tokenization: ak.Array) -> tuple[ak.Array, ak.Array]:
        """Pad tokenization that is smaller than desired input length.

        Args:
            cell_tokenization: the stacked gene identity- and gene expression-based tokenization
                dof a cell.

        """

        _, input_length = cell_tokenization.shape
        pad_widths = ((0, 0), (0, self.max_input_length - input_length))
        padded = np.pad(
            cell_tokenization.astype(self.float_dtype),
            pad_widths,
            "constant",
            constant_values=(0, np.nan),
        )

        padded[0, np.isnan(padded[0]).nonzero()] = self.fg.pad_value
        padded[1, np.isnan(padded[1]).nonzero()] = self.fe.pad_value

        return padded

    @abstractmethod
    def limit(self, cell_tokenization: NDArray) -> NDArray:
        """Limit tokenization that exceeds the desired input length.

        Args:
            cell_tokenization: the stacked gene identity- and gene expression-based tokenization
                of a cell.

        """

    def tailor(
        self,
        gene_tokenization,
        expression_tokenization,
    ) -> NDArray | ak.Array:

        # first, drop any NaN values here
        # Assuming gene_tokenization is a pandas Series and expression_tokenization is a numpy array
        valid_mask = ~np.isnan(expression_tokenization)

        filtered_gene_tokenization = gene_tokenization[valid_mask]
        filtered_expression_tokenization = expression_tokenization[valid_mask]

        cell_tokenization = np.stack([filtered_gene_tokenization.values, filtered_expression_tokenization], axis=0)
        # cell_tokenization = np.stack([gene_tokenization, expression_tokenization], axis=0)

        _, input_length = cell_tokenization.shape

        if input_length > self.max_input_length:
            return self.limit(cell_tokenization)
        return self.pad(cell_tokenization)

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
        limited at this stage, i.e. they are regular-shaped tensors.

        Args:
            identity_inputs: batched gene identity inputs
            gene_embedding_layer: Torch module for embedding based on gene identity.
            expression_inputs: batched gene expression inputs
            expression_embedding_layer: Torch module for embedding based on expression.

        Returns:
            Embeddings of cells.

        """


class GeneformerFc(Fc):
    """Implementation of Geneformer cell embedding."""

    def limit(self, cell_tokenization: NDArray) -> NDArray:
        return cell_tokenization[:, : self.max_input_length]

    def embed_cells(
        self,
        identity_inputs: Tensor,
        gene_embedding_layer: Module | None,
        expression_inputs: Tensor,
        expression_embedding_layer: Module | None,
    ) -> Tensor:
        """Geneformer cell embedding function.

        Ignores expression embedding layer; uses embeddings based on identity embeddings.

        Args:
            gene_embedding_layer:  # TODO: fill out
            expression_embedding_layer: # TODO fill out

        """

        embeddings = gene_embedding_layer(identity_inputs)
        return embeddings


class DummyFc(Fc):
    """Dummy `Fc` that does not tailor the size of the input."""

    def tailor(
        self,
        gene_tokenization,
        expression_tokenization,
    ) -> NDArray | ak.Array:

        cell_tokenization = np.stack([gene_tokenization, expression_tokenization], axis=0)
        _, input_length = cell_tokenization.shape

        return cell_tokenization

    def __getitem__(self, cell_index: int) -> tuple[NDArray, NDArray, NDArray]:
        """Dummy `__getitem__` for model that does not need an `Fc`.

        Returns:
            A tuple of gene identity embedding indices and gene expression embedding indices for all cells.

        """
        identity_indices, expression_inputs = self.fe[cell_index]
        padding_mask = np.zeros(self.max_input_length)

        return identity_indices, expression_inputs, padding_mask

    def limit(self, cell_tokenization: NDArray) -> NDArray:
        pass

    def embed_cells(
        self,
        identity_inputs: Tensor,
        gene_embedding_layer: Module | None,
        expression_inputs: Tensor,
        expression_embedding_layer: Module | None,
    ) -> Tensor:

        pass


class ScGPTFc(Fc):
    """Implementation of scGPT cell embedding."""

    def __init__(
        self,
        fg: Fg | None,
        fe: Fe | None,
        adata: ad.AnnData,
        max_input_length: Optional[int] = None,
        float_dtype: str = "float32",
    ):
        super().__init__(fg, fe, adata, max_input_length, float_dtype)
        seed = 0  # TODO: make this configurable???
        self.rng = np.random.default_rng(seed)

    def limit(self, cell_tokenization: NDArray) -> NDArray:
        # Shape: (2, N)
        expression_values = cell_tokenization[1]

        # Separate indices
        nonzero_indices = np.where(expression_values != 0)[0]
        zero_indices = np.where(expression_values == 0)[0]

        # First: sample nonzero expression tokens
        num_nonzero_to_sample = min(len(nonzero_indices), self.max_input_length)
        selected_nonzero = self.rng.choice(nonzero_indices, num_nonzero_to_sample, replace=False)

        # If needed: sample zero-expression tokens to fill up
        num_remaining = self.max_input_length - num_nonzero_to_sample
        if num_remaining > 0:
            selected_zero = self.rng.choice(zero_indices, num_remaining, replace=False)
            final_indices = np.concatenate([selected_nonzero, selected_zero])
        else:
            final_indices = selected_nonzero

        # Optionally shuffle to avoid position bias, but we dont need to because the gene ids are the position
        # self.rng.shuffle(final_indices)

        return cell_tokenization[:, final_indices]

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
        # Convert str float_dtype -> actual torch dtype
        # torch_dtype = getattr(torch, self.float_dtype)

        # Cast expression_inputs to float_dtype
        expression_inputs = expression_inputs.to(torch.float32)

        gene_embeddings = gene_embedding_layer(identity_inputs)
        expression_embeddings = expression_embedding_layer(expression_inputs)

        return gene_embeddings + expression_embeddings


class ChromosomeAwareFc(Fc):
    """Chromosome-aware implementation of cell embedding."""

    def __init__(
        self,
        fg: Fg | None,
        fe: Fe | None,
        adata: ad.AnnData,
        chroms: Optional[NDArray] = None,
        starts: Optional[NDArray] = None,
        max_input_length: Optional[int] = None,
    ):
        """
        Args:
            chroms: Chromosome IDs for each gene.
            starts: Genomic start positions of genes on their chromosomes.
        """
        super().__init__(fg, fe, adata, max_input_length)
        self.chroms = chroms
        self.starts = starts

    def preprocess_cells(self):
        """Using the `fg` and `fe`, preprocess input cells with chromosome-aware
        sorting."""
        gene_names = self.adata.var_names
        processed_expression_values, processed_expression_indices = self.fe[:]

        gene_lists = ak.Array(
            [gene_names[cell_indices] for cell_indices in processed_expression_indices],
        )

        cell_identity_inputs = []
        for cell_idx, gene_indices in enumerate(processed_expression_indices):
            if self.chroms is not None and self.starts is not None:
                # Perform chromosome-aware sorting
                sorted_indices = self._chromosome_sort(gene_indices)
                cell_identity_inputs.append(self.fg[gene_names[sorted_indices]])
            else:
                # Default behavior
                cell_identity_inputs.append(self.fg[gene_lists[cell_idx]])

        # Store processed values
        self.adata.obsm["cell_identity_inputs"] = ak.Array(cell_identity_inputs)
        self.adata.obsm["cell_expression_inputs"] = processed_expression_values

    def _chromosome_sort(self, gene_indices):
        """Sort genes by chromosome and genomic start positions."""
        chroms = self.chroms[gene_indices]
        starts = self.starts[gene_indices]

        # Group by chromosome
        chrom_sort_order = np.argsort(chroms)
        sorted_indices = gene_indices[chrom_sort_order]
        sorted_chroms = chroms[chrom_sort_order]
        sorted_starts = starts[chrom_sort_order]

        # Within each chromosome, sort by start position
        chrom_groups = np.split(sorted_indices, np.unique(sorted_chroms, return_index=True)[1][1:])
        sorted_sequence = []
        for group in chrom_groups:
            group_starts = sorted_starts[group]
            sorted_sequence.extend(group[np.argsort(group_starts)])

        return np.array(sorted_sequence)

    def embed_cells(
        self,
        identity_inputs: Tensor,
        gene_embedding_layer: Module | None,
        expression_inputs: Tensor,
        expression_embedding_layer: Module | None,
    ) -> Tensor:
        """Embed cells using chromosome-aware sequences."""
        gene_embeddings = gene_embedding_layer(identity_inputs)
        expression_embeddings = expression_embedding_layer(expression_inputs)

        return gene_embeddings + expression_embeddings


class ScBERTFc(ScGPTFc):
    """Implementation of scBERT cell embedding."""

    # TODO: is ScBERTFc actually the same as ScGPTFc?
