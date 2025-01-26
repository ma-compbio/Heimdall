from abc import ABC, abstractmethod
from typing import Union

import anndata as ad
import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from scipy.sparse import csr_array, issparse


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
        self.embedding_parameters = OmegaConf.to_container(embedding_parameters, resolve=True)
        self.d_embedding = d_embedding
        self.vocab_size = vocab_size
        self.pad_value = vocab_size - 2 if pad_value is None else pad_value
        self.mask_value = vocab_size - 1 if mask_value is None else mask_value

        if not issparse(self.adata.X):
            print(
                "> Data was provided dense format, converting to CSR."
                " Please consider pre-computing it to save memory.",
            )

        self.adata.X = csr_array(self.adata.X)

    def _get_inputs_from_csr(self, cell_index: int):
        """Get expression values and gene indices from internal CSR
        representation.

        Args:
            cell_index: cell for which to process expression values and get indices, as stored in `self.adata`.

        """

        expression = self.adata.X
        start = expression.indptr[cell_index]
        end = expression.indptr[cell_index + 1]
        cell_expression_inputs = expression.data[start:end]
        cell_identity_inputs = expression.indices[start:end]

        return cell_identity_inputs, cell_expression_inputs

    def preprocess_embeddings(self, float_dtype: str = "float32"):
        """Preprocess expression embeddings and store them for use during model
        inference.

        Preprocessing may include anything from downloading gene embeddings from
        a URL to generating embeddings from scratch.

        Returns:
            TODO: Update this docstring

        """
        self.expression_embeddings = None
        self.prepare_embedding_parameters()

    @abstractmethod
    def __getitem__(self, cell_index: int) -> tuple[NDArray, NDArray]:
        """Get the indices of genes in the expression embedding array.

        Args:
            cell_index: cell for which to process expression values and get indices, as stored in `self.adata`.

        Returns:
            Index of value in the expression embeddings, or `pd.NA` if the gene has no mapping.

        """

    def load_from_cache(
        self,
        expression_embeddings: NDArray | None,
    ):
        """Load processed values from cache."""
        self.expression_embeddings = expression_embeddings
        self.prepare_embedding_parameters()

    def prepare_embedding_parameters(self):
        """Replace config placeholders with values after preprocessing."""
        args = self.embedding_parameters.get("args", {})
        for key, value in args.items():
            if value == "max_seq_length":
                value = self.adata.n_vars
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
        vocab_size = num_bins + 3  # Accounting for mask, pad tokens and empty bin (zero expr.)
        super().__init__(adata, vocab_size=vocab_size, **fe_kwargs)
        self.num_bins = num_bins

    def _digitize(self, x: np.ndarray, bins: np.ndarray, side="both") -> np.ndarray:
        """
        https://github.com/bowang-lab/scGPT/blob/7301b51a72f5db321fccebb51bc4dd1380d99023/scgpt/preprocess.py#L239
        Digitize the data into bins. This method spreads data uniformly when bins
        have same values.

        Args:

        x (:class:`np.ndarray`):
            The data to digitize.
        bins (:class:`np.ndarray`):
            The bins to use for digitization, in increasing order.
        side (:class:`str`, optional):
            The side to use for digitization. If "one", the left side is used. If
            "both", the left and right side are used. Default to "one".

        Returns:

        :class:`np.ndarray`:
            The digitized data.
        """
        assert x.ndim == 1 and bins.ndim == 1

        left_digits = np.digitize(x, bins)
        if side == "one":
            return left_digits

        right_digits = np.digitize(x, bins, right=True)

        rands = np.random.rand(len(x))  # uniform random numbers

        digits = rands * (right_digits - left_digits) + left_digits
        digits = np.ceil(digits).astype(np.int64)
        return digits

    def binning(self, row, n_bins) -> Union[np.ndarray, torch.Tensor]:
        """Binning the row into n_bins.

        https://github.com/bowang-lab/scGPT/blob/7301b51a72f5db321fccebb51bc4dd1380d99023/scgpt/preprocess.py#L274

        """
        dtype = row.dtype
        return_np = False if isinstance(row, torch.Tensor) else True
        row = row.cpu().numpy() if isinstance(row, torch.Tensor) else row
        # TODO: use torch.quantile and torch.bucketize

        if row.max() == 0:
            return np.zeros_like(row, dtype=dtype) if return_np else torch.zeros_like(row, dtype=dtype)

        if row.min() <= 0:
            non_zero_ids = row.nonzero()
            non_zero_row = row[non_zero_ids]
            bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
            non_zero_digits = self._digitize(non_zero_row, bins)
            binned_row = np.zeros_like(row, dtype=np.int64)
            binned_row[non_zero_ids] = non_zero_digits
        else:
            bins = np.quantile(row, np.linspace(0, 1, n_bins - 1))
            binned_row = self._digitize(row, bins)

        return torch.from_numpy(binned_row) if not return_np else binned_row.astype(dtype)

    def __getitem__(self, cell_index: int):
        """Input is an adata indexed at cell [idx]

        returns two vectors of cell_expression_inputs and cell_identity_inputs
        where cell_expression_inputs is a vector of cell expression values and
        cell_identity_inputs is a vector of corresponding gene indices

        """

        cell_identity_inputs, cell_expression_inputs = self._get_inputs_from_csr(cell_index)

        # Bin the cell expression values
        cell_expression_inputs_binned = self.binning(cell_expression_inputs, self.num_bins + 1)

        return cell_identity_inputs, cell_expression_inputs_binned


class NonzeroIdentityFe(Fe):
    """Directly pass the continuous values. Remove zeros.

    Args:
        adata: input AnnData-formatted dataset, with gene names in the `.var` dataframe.
        d_embedding: dimensionality of embedding for each expression entity
        embedding_parameters: dimensionality of embedding for each expression entity
        num_bins: number of bins to generate

    """

    def __getitem__(self, cell_index: int):
        return self._get_inputs_from_csr(cell_index)


class DummyFe(Fe):
    """Directly pass the continuous values. Does not remove zero expression
    elements.

    Args:
        adata: input AnnData-formatted dataset, with gene names in the `.var` dataframe.
        d_embedding: dimensionality of embedding for each expression entity
        embedding_parameters: dimensionality of embedding for each expression entity

    """

    def __getitem__(self, cell_index: int):
        """Input is an adata indexed at cell [idx]

        returns two vectors of cell_expression_inputs and cell_identity_inputs
        where cell_expression_inputs is a vector of cell expression values and
        cell_identity_inputs is a vector of corresponding gene indices

        """
        cell_expression_inputs = self.adata.X[[cell_index], :].toarray()
        cell_identity_inputs = np.arange(self.adata.n_vars)

        return cell_identity_inputs, cell_expression_inputs


class SortingFe(Fe):
    """Sorting Fe."""

    def __getitem__(self, cell_index: int) -> tuple[NDArray, NDArray]:
        """Returns two vectors of cell_expression_inputs and
        cell_identity_inputs where cell_expression_inputs is a vector of cell
        expression values and cell_identity_inputs is a vector of corresponding
        gene indices."""

        nonzero_indices, nonzero_values = self._get_inputs_from_csr(cell_index)

        if "medians" in self.adata.var:
            nonzero_values = nonzero_values - self.adata.var["medians"].iloc[nonzero_indices].values

        # Sort non-zero values in descending order
        sorted_order = np.argsort(nonzero_values)[::-1]  # Indices for sorting descending
        cell_expression_inputs = nonzero_values[sorted_order]
        cell_identity_inputs = nonzero_indices[sorted_order]

        return cell_identity_inputs, cell_expression_inputs
