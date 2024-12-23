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
        float_dtype: str = "float32",
    ):
        self.fg = fg
        self.fe = fe
        self.adata = adata
        self.max_input_length = max_input_length
        self.float_dtype = float_dtype

    def preprocess_cells(self):
        """Using the `fg` and `fe`, preprocess input cells, retrieve indices of
        both gene and expression embeddings.

        This function can be deterministic, or may involve random sampling.

        Returns:
            Sets the following fields of `self.adata`:
            `.obsm['cell_identity_inputs']` : :class:`~numpy.ndarray`
                (shape `(self.adata.n_obs, self.max_input_length)`)
                Gene identity embedding indices for all cells.
            `.obsm['cell_expression_inputs']` : :class:`~numpy.ndarray`
                (shape `(self.adata.n_obs, self.max_input_length)`)
                Gene expression embedding indices for all cells.

        """

        gene_names = self.adata.var_names

        processed_expression_values, processed_expression_indices = self.fe[:]

        gene_lists = ak.Array(
            [gene_names[cell_indices] for cell_indices in processed_expression_indices],
        )

        cell_identity_inputs = ak.Array(
            [self.fg[gene_list] for gene_list in gene_lists],
        )

        self.adata.obsm["cell_identity_inputs"] = ak.values_astype(cell_identity_inputs, self.float_dtype)
        self.adata.obsm["cell_expression_inputs"] = ak.values_astype(processed_expression_values, self.float_dtype)

    def __getitem__(self, cell_index: int) -> tuple[NDArray, NDArray, NDArray]:
        """Retrieve `cell_identity_inputs`, `cell_expression_inputs` and
        `padding_mask`.

        Can only be called after running `self.preprocess_cells()`.

        Returns:
            A tuple of gene identity embedding indices and gene expression embedding indices for all cells.

        """

        identity_inputs = self.adata.obsm["cell_identity_inputs"][cell_index]
        expression_inputs = self.adata.obsm["cell_expression_inputs"][cell_index]

        assert len(identity_inputs) == len(
            expression_inputs,
        ), "Gene identity and expression inputs do not have the same shape"  # TODO: do we really want to enforce this?

        # Padding and truncating
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
        gene_tokenization: ak.Array,
        expression_tokenization: ak.Array,
    ) -> NDArray | ak.Array:

        cell_tokenization = np.stack(
            [ak.to_numpy(gene_tokenization), ak.to_numpy(expression_tokenization)],
            axis=0,
        )  # This returns a ak.Array
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

    def load_from_cache(
        self,
        cell_identity_inputs: NDArray,
        cell_expression_inputs: NDArray,
    ):
        """Load processed values from cache."""
        # TODO: add tests

        self.adata.obsm["cell_identity_inputs"] = cell_identity_inputs
        self.adata.obsm["cell_expression_inputs"] = cell_expression_inputs


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
        gene_tokenization: ak.Array,
        expression_tokenization: ak.Array,
    ) -> NDArray | ak.Array:

        cell_tokenization = np.stack(
            [ak.to_numpy(gene_tokenization), ak.to_numpy(expression_tokenization)],
            axis=0,
        )  # This returns a ak.Array
        _, input_length = cell_tokenization.shape

        return cell_tokenization

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
    ):
        super().__init__(fg, fe, adata, max_input_length)
        seed = 0  # TODO: make this configurable???
        self.rng = np.random.default_rng(seed)

    def limit(self, cell_tokenization: NDArray) -> NDArray:
        _, input_length = cell_tokenization.shape
        sample_indices = self.rng.choice(input_length, self.max_input_length, replace=False)

        return cell_tokenization[:, sample_indices]

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
